"""Figures 5-7: effect of each curation threshold on retention and downstream Macro-F1.

For every threshold value the bulk candidate pool (data/filter_runs/curation_v2/<model>/
scored_candidates.csv, candidates that passed the non-NLI quality checks) is re-selected with
one rule changed and the others at their defaults:
  Figure 5  minimum intended-role entailment for the four roles      (default 0.50)
  Figure 6  maximum positive-role entailment allowed for Otherwise   (default 0.75)
  Figure 7  lexical-marker fraction above which Otherwise is dropped (v2 default: no exclusion;
            the 0.03 warning level of the draft is marked)
Selection then applies the curation quotas (500 per role, 2,000 Otherwise per model x disease)
in the curation's stable SHA-256 order. Top-ups are not used: they exist only at the default
thresholds, so including them would mask the threshold effect.

Downstream score: TF-IDF (1-2 grams) + linear SVM, trained on two diseases and tested on the
third, Macro-F1 averaged over the three folds; Δ is relative to the default threshold.
Panels: retained share of eligible candidates, selected corpus size, Macro-F1 and Δ vs default.

Run from PROJECT_ROOT:  python -m healthrolebench.curation.threshold_f1 --project-root .
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from healthrolebench.common.data import DISEASES, MODELS, digest

OUT = Path('data/experiments/threshold_f1_v1')
SWEEPS = {
    5: ('positive_min_entailment', (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90), 0.50),
    6: ('neutral_max_positive_entailment', (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90), 0.75),
    7: ('otherwise_max_lexical_fraction', (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, None), None),
}
DEFAULTS = dict(positive_min_entailment=.50, neutral_min_entailment=.50, neutral_max_positive_entailment=.75,
                otherwise_max_lexical_fraction=None)


def stable_rank(model, sample_id):    # identical to scripts/filter_v2/rules.stable_rank
    return hashlib.sha256(('healthrolebench-v2|42|' + model + '|' + sample_id).encode()).hexdigest()


def load_pool(root):
    pool = []
    csv.field_size_limit(1 << 26)
    for model in MODELS:
        path = root / 'data/filter_runs/curation_v2' / model / 'scored_candidates.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            for r in csv.DictReader(handle):
                if r['curation_decision'] not in ('pass', 'role_context_mismatch'):
                    continue                          # failed quality checks before the NLI stage
                pool.append(dict(model=model, sample_id=r['sample_id'], disease=r['disease'], role=r['intended_role'],
                                 text=r['text'], scores=json.loads(r['nli_scores']),
                                 lexical=float(r['lexical_marker_fraction']), rank=stable_rank(model, r['sample_id'])))
    return pool


def select(pool, rule):
    eligible = []
    for c in pool:
        s = c['scores']
        if c['role'] == 'Otherwise':
            ok = (s['Otherwise'] >= rule['neutral_min_entailment'] and
                  max(v for k, v in s.items() if k != 'Otherwise') < rule['neutral_max_positive_entailment'])
            if ok and rule['otherwise_max_lexical_fraction'] is not None:
                ok = c['lexical'] <= rule['otherwise_max_lexical_fraction']
        else:
            ok = s[c['role']] >= rule['positive_min_entailment']
        if ok:
            eligible.append(c)
    counts, chosen, seen = Counter(), [], set()
    for c in sorted(eligible, key=lambda c: c['rank']):
        text = ' '.join(c['text'].casefold().split())
        if text in seen:
            continue
        seen.add(text)
        key = c['model'], c['disease'], c['role']
        if counts[key] < (2000 if c['role'] == 'Otherwise' else 500):
            counts[key] += 1
            chosen.append(c)
    return eligible, chosen


def downstream(chosen):
    text = np.array([c['text'] for c in chosen], dtype=object)
    label = np.array([c['role'] for c in chosen])
    disease = np.array([c['disease'] for c in chosen])
    folds = []
    for held in DISEASES:
        train, test = disease != held, disease == held
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                              LinearSVC(class_weight='balanced', random_state=42))
        model.fit(text[train], label[train])
        folds.append(f1_score(label[test], model.predict(text[test]), average='macro'))
    return mean(folds)


def plot(out, figure, parameter, records, retention):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = [r[1] for r in records]
    labels = ['none' if x is None else f'{x:g}' for x in xs]
    pos = list(range(len(xs)))
    default = next(i for i, r in enumerate(records) if r[-1])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), layout='constrained')
    for model, values in retention.items():
        axes[0].plot(pos, [100 * v for v in values], marker='o', label=model)
    axes[0].set(title='Retained of eligible pool (%)', ylim=(0, 100))
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(pos, [r[3] for r in records], marker='o', color='0.3')
    axes[1].set(title='Selected corpus (posts)')
    axes[2].plot(pos, [100 * r[4] for r in records], marker='o', color='C3')
    axes[2].set(title='Macro-F1 (%)')
    axes[3].bar(pos, [100 * r[5] for r in records], color='C2')
    axes[3].set(title='Δ Macro-F1 vs default (pp)')
    for axis in axes:
        axis.set_xticks(pos, labels, rotation=45, fontsize=7)
        axis.axvline(default, color='0.5', linestyle=':')
        axis.set_xlabel(parameter.replace('_', ' '), fontsize=7)
    fig.savefig(out / f'figure_{figure}.pdf')
    plt.close(fig)


def main(root):
    out = root / OUT
    out.mkdir(parents=True, exist_ok=False)
    pool = load_pool(root)
    quality_pool = Counter(c['model'] for c in pool)
    records = []
    for figure, (parameter, values, default) in SWEEPS.items():
        rows_for_figure, retention = [], {m: [] for m in MODELS}
        base_f1 = None
        for value in values:
            rule = dict(DEFAULTS, **{parameter: value})
            eligible, chosen = select(pool, rule)
            f1 = downstream(chosen)
            for m in MODELS:
                retention[m].append(sum(c['model'] == m for c in chosen) / quality_pool[m])
            rows_for_figure.append([figure, value, len(eligible), len(chosen), f1, 0.0, value == default])
            print('figure', figure, parameter, value, 'eligible', len(eligible), 'selected', len(chosen), 'macro-F1', round(f1, 4), flush=True)
        base_f1 = next(r[4] for r in rows_for_figure if r[-1])
        for r in rows_for_figure:
            r[5] = r[4] - base_f1
        records.extend(rows_for_figure)
        plot(out, figure, parameter, rows_for_figure, retention)
        with (out / f'figure_{figure}_retention.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(('generator', 'threshold', 'retained_fraction_of_quality_pool'))
            writer.writerows((m, v, x) for m, xs in retention.items() for v, x in zip(values, xs))
    with (out / 'figures_5_7_f1.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('figure', 'threshold', 'eligible', 'selected', 'macro_f1', 'delta_vs_default', 'is_default'))
        writer.writerows(records)
    (out / 'manifest.json').write_text(json.dumps(dict(
        sweeps={k: dict(parameter=p, values=v, default=d) for k, (p, v, d) in SWEEPS.items()},
        pool='bulk candidates passing non-NLI quality checks', pool_sizes=quality_pool,
        downstream='TF-IDF(1-2)+LinearSVC, 3 held-out-disease folds, Macro-F1 mean',
        runner_sha256=digest(Path(__file__))), indent=2) + chr(10))
    print('Figures 5-7 written:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figures 5-7: threshold sensitivity with downstream Macro-F1.')
    parser.add_argument('--project-root', required=True, type=Path)
    main(parser.parse_args().project_root.resolve())
