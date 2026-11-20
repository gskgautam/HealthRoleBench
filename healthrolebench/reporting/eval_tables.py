"""Tables 8, 9, 10, 13 (prediction overlap) and 18, and Figure 8, from the evaluation set.

Inputs (all under data/experiments/):
  eval_set_v1/                      frozen 1,500-post evaluation set and rewrites
  zero_shot_eval_v1/<condition>/    zero-shot completions: llama2, mistral, qwen, gemma, gpt4o
  probe_full_v1/                    full-data frozen-backbone heads (table_08_full.csv, eval_predictions.csv)

Conventions (documented for the paper):
- Completions are parsed with labels-v2 (λ = 2); invalid answers count as wrong.
- Diagonal cells (a model classifying posts it generated) are excluded from means.
- Table 8: open models use the frozen-backbone head tested on all 12,000 posts of the held-out
  generator; GPT-4o (no weights) enters zero-shot on that generator's 300 evaluation posts.
- Table 10: zero-shot and head are compared on the same evaluation posts (off-diagonal generators).
- Table 13: per-role F1 agreement between two classifiers, on posts where both gave a valid label.
- Table 18: per-role F1 averaged over the five classifiers. Paraphrased/adversarial use only
  rewrites that kept their role (NLI check); the paired original score on the same posts is
  reported alongside so the drop is not confounded by the subset.

Run from PROJECT_ROOT:  python -m healthrolebench.reporting.eval_tables --project-root .
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from healthrolebench.common.data import digest, rows
from healthrolebench.curation.eval_set import load_eval
from healthrolebench.common.items import CONDITIONS
from healthrolebench.common.labels import INVALID, PARSER_VERSION, ROLES, parse

CLASSIFIERS = ('llama2', 'mistral', 'qwen', 'gemma', 'gpt4o')
OUT = Path('data/experiments/eval_tables_v1')
RUNS = Path('data/experiments/zero_shot_eval_v1')


def scores(true, pred, labels=ROLES):
    """Macro-F1, accuracy, macro-precision, macro-recall and per-role F1 (invalid counts as wrong)."""
    per_role, precision, recall = {}, [], []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true, pred))
        fp = sum(t != label and p == label for t, p in zip(true, pred))
        fn = sum(t == label and p != label for t, p in zip(true, pred))
        per_role[label] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
    present = [l for l in labels if l in true]
    return dict(macro_f1=mean(per_role[l] for l in present), accuracy=mean(t == p for t, p in zip(true, pred)),
                precision=mean(precision), recall=mean(recall), per_role=per_role)


def load_predictions(root, condition, max_distance=2):
    """{classifier: {item_id: label}} for every completed run of a condition."""
    result = {}
    for model in CLASSIFIERS:
        path = root / RUNS / condition / f'{model}.csv'
        if path.is_file():
            result[model] = {r['item_id']: parse(r['raw_completion'], ROLES, max_distance) for r in rows(path)}
    return result


def write(path, header, records):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(records)


def export(root):
    out = root / OUT
    out.mkdir(parents=True, exist_ok=False)
    key, _ = load_eval(root)
    gold = {item: row['intended_role'] for item, row in key.items()}
    source = {item: row['generator'] for item, row in key.items()}
    generators = sorted(set(source.values()))
    original = load_predictions(root, 'original')
    for model, pred in original.items():
        if pred.keys() != gold.keys():
            raise ValueError(f'{model} original run does not cover the evaluation set')
    inputs = {}

    # Table 9: zero-shot, generator x classifier
    table9 = []
    for model, pred in original.items():
        for generator in generators:
            ids = [i for i in gold if source[i] == generator]
            s = scores([gold[i] for i in ids], [pred[i] for i in ids])
            table9.append((generator, model, len(ids), s['macro_f1'], s['accuracy'], s['precision'], s['recall'],
                           sum(pred[i] == INVALID for i in ids)))
    write(out / 'table_09_zero_shot.csv', ('generator', 'classifier', 'n', 'macro_f1', 'accuracy', 'macro_precision',
                                           'macro_recall', 'invalid'), table9)

    # Table 8: frozen heads on full held-out generators + GPT-4o zero-shot
    table8 = []
    probe = root / 'data/experiments/probe_full_v1'
    if (probe / 'table_08_full.csv').is_file():
        inputs['probe'] = digest(probe / 'table_08_full.csv')
        for r in rows(probe / 'table_08_full.csv'):
            table8.append((r['held_generator'], r['classifier'], 'frozen_head', r['n'], r['macro_f1'], r['macro_f1_sd'],
                           r['accuracy'], r['macro_precision'], r['macro_recall']))
    for (generator, model, n, f1, acc, prec, rec, _) in table9:
        if model == 'gpt4o' and generator != 'gpt4o':
            table8.append((generator, model, 'zero_shot', n, f1, '', acc, prec, rec))
    write(out / 'table_08_cross_model.csv', ('held_generator', 'classifier', 'regime', 'n', 'macro_f1', 'macro_f1_sd',
                                             'accuracy', 'macro_precision', 'macro_recall'), table8)

    # Table 10: zero-shot vs head on the same evaluation posts
    table10 = []
    if (probe / 'eval_predictions.csv').is_file():
        head = defaultdict(lambda: defaultdict(dict))
        for r in rows(probe / 'eval_predictions.csv'):
            head[r['classifier']][r['seed']][r['item_id']] = r['prediction']
        for model in (m for m in CLASSIFIERS if m in head and m in original):
            zero, trained = [], []
            for generator in generators:
                if generator == model:
                    continue
                ids = [i for i in gold if source[i] == generator]
                true = [gold[i] for i in ids]
                zero.append(scores(true, [original[model][i] for i in ids])['macro_f1'])
                trained.append(mean(scores(true, [head[model][seed][i] for i in ids])['macro_f1'] for seed in head[model]))
            table10.append((model, len(zero), mean(zero), mean(trained), 100 * (mean(trained) - mean(zero))))
    write(out / 'table_10_zero_shot_vs_head.csv', ('classifier', 'generator_folds', 'zero_shot_macro_f1',
                                                   'head_macro_f1', 'delta_pp'), table10)

    # Table 13 (overlap): pairwise agreement on posts where both classifiers answered
    overlap = []
    for a, b in itertools.combinations(original, 2):
        shared = [i for i in gold if original[a][i] != INVALID and original[b][i] != INVALID]
        agreement = scores([original[a][i] for i in shared], [original[b][i] for i in shared])['per_role']
        overlap.extend((a, b, role, len(shared), agreement[role]) for role in ROLES)
    write(out / 'table_13_prediction_overlap.csv', ('classifier_1', 'classifier_2', 'role', 'shared_valid_n', 'agreement_f1'), overlap)

    # Table 18: conditions, with paired originals for the rewrite subsets
    table18 = []
    for condition in CONDITIONS:
        pred = original if condition == 'original' else load_predictions(root, condition)
        for model, labels in pred.items():
            ids = sorted(labels)
            s = scores([gold[i] for i in ids], [labels[i] for i in ids])
            paired = scores([gold[i] for i in ids], [original[model][i] for i in ids]) if model in original else None
            for role in ROLES:
                support = sum(gold[i] == role for i in ids)
                table18.append((condition, model, len(ids), role, support, s['per_role'][role],
                                paired['per_role'][role] if paired else '', s['macro_f1'], paired['macro_f1'] if paired else ''))
    write(out / 'table_18_stress.csv', ('condition', 'classifier', 'n', 'role', 'role_support', 'f1', 'paired_original_f1',
                                        'macro_f1', 'paired_original_macro_f1'), table18)

    # Figure 8: parser tolerance λ
    sweep = []
    for distance in range(5):
        for model, pred in load_predictions(root, 'original', distance).items():
            ids = sorted(gold)
            s = scores([gold[i] for i in ids], [pred[i] for i in ids])
            sweep.append((model, distance, mean(pred[i] != INVALID for i in ids), s['accuracy'], s['macro_f1']))
    write(out / 'figure_8.csv', ('classifier', 'edit_distance', 'valid_label_rate', 'accuracy', 'macro_f1'), sweep)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout='constrained', sharey=True)
    for model in original:
        points = [r for r in sweep if r[0] == model]
        for j, (axis, title) in enumerate(zip(axes, ('Valid label rate', 'Accuracy', 'Macro-F1')), start=2):
            axis.plot([r[1] for r in points], [100 * r[j] for r in points], label=model, marker='o')
            axis.set(title=title, xlabel='Label-match tolerance λ (edits)', xticks=range(5), ylim=(0, 100))
    for axis in axes:
        axis.axvline(2, color='0.5', linestyle=':')
    axes[0].set_ylabel('Percent')
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(out / 'figure_8.pdf')
    plt.close(fig)
    (out / 'manifest.json').write_text(json.dumps(dict(
        parser=PARSER_VERSION, classifiers=sorted(original), conditions={c: sorted(load_predictions(root, c)) for c in CONDITIONS},
        eval_manifest_sha256=digest(root / 'data/experiments/eval_set_v1/manifest.json'), inputs=inputs,
        runner_sha256=digest(Path(__file__))), indent=2) + chr(10))
    print('Evaluation-set tables written:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tables 8, 9, 10, 13 (overlap), 18 and Figure 8 from the evaluation set.')
    parser.add_argument('--project-root', required=True, type=Path)
    export(parser.parse_args().project_root.resolve())
