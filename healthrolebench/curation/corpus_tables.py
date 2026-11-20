import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from healthrolebench.common.artifacts import private_text_artifact

ROOT = Path('data')
OUT = ROOT / 'experiments' / 'corpus_tables_v1'
MODELS = ('gpt4o', 'qwen', 'llama2', 'mistral', 'gemma')
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)


def read(path):
    with path.open(encoding='utf-8', newline='') as source:
        return list(csv.DictReader(source))


def write(name, header, rows, private=False):
    path = OUT / name
    output = private_text_artifact(path) if private else path.open('x', encoding='utf-8', newline='')
    with output as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def plot(number, title, x_label, series, reference):
    fig, ax = plt.subplots(figsize=(6.2, 3.6), layout='constrained')
    for name, points in series.items():
        ax.plot(*zip(*points), label=name, marker='o', linewidth=1.6)
    ax.set(title=title, xlabel=x_label, ylabel='Candidates (%)', ylim=(0, 100))
    ax.axvline(reference, color='0.3', linestyle=':', linewidth=1)
    ax.grid(axis='y', alpha=.22)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(OUT / f'figure_{number}_v2.pdf')
    plt.close(fig)
    if len({tuple(x for x, y in p) for p in series.values()}) != 1:
        raise ValueError('Series thresholds differ')
    write(f'figure_{number}_v2.csv', ('series', 'threshold', 'percent'),
          ((name, x, y) for name, points in series.items() for x, y in points))


def main():
    inputs = {}
    selected = []
    scored = defaultdict(list)
    flow = []
    for model in MODELS:
        folder = ROOT / 'filter_runs' / 'curation_v2' / model
        base = folder / 'selected_machine_curated_draft.csv'
        topup = ROOT / 'topups_v2' / f'{model}_candidates.csv'
        candidate = folder / 'scored_candidates.csv'
        for path in (base, topup, candidate):
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        bulk_rows, topup_rows, candidates = map(read, (base, topup, candidate))
        decisions = Counter(row['curation_decision'] for row in candidates)
        assert len(bulk_rows) + len(topup_rows) == 12000
        flow.append((model, len(candidates), decisions['pass'], len(bulk_rows), len(topup_rows),
                     len(bulk_rows) + len(topup_rows)))
        selected.extend((model, stage, row) for stage, rows in (('bulk', bulk_rows), ('topup', topup_rows)) for row in rows)
        for row in candidates:
            if row['curation_decision'] not in ('pass', 'role_context_mismatch'):
                continue
            scores = json.loads(row['nli_scores'])
            role = row['intended_role']
            assert len(scores) == (5 if role == 'Otherwise' else 1)
            largest_positive = max(v for k, v in scores.items() if k != 'Otherwise')
            accepted = (scores['Otherwise'] >= .5 and largest_positive < .75) if role == 'Otherwise' else scores[role] >= .5
            assert accepted == (row['curation_decision'] == 'pass')
            scored[model].append((role, scores[role], largest_positive, float(row['lexical_marker_fraction'])))

    OUT.mkdir(parents=True)
    write('table_02_curation.csv', ('generator', 'bulk_scored_input', 'bulk_eligible', 'bulk_selected', 'accepted_topups', 'selected'), flow)
    cells = defaultdict(list)
    stats = defaultdict(list)
    features = defaultdict(Counter)
    all_features = Counter()
    for model, stage, row in selected:
        role = row['intended_role']
        cells[model, row['disease'], role].append((stage, row))
        text = row['text']
        words = re.findall(r'\b[\w’]+\b', text.casefold())
        whitespace_tokens = text.split()
        stats[role].append((len(whitespace_tokens), len([s for s in re.split(r'[.!?]+\s+|[.!?]+$', text) if s.strip()]),
                            len(set(whitespace_tokens)) / len(whitespace_tokens)))
        ngrams = set(words) | set(zip(words, words[1:]))
        features[role].update(ngrams)
        all_features.update(ngrams)
    assert len(cells) == 75 and len(selected) == 60000
    write('table_05_linguistic.csv', ('role', 'n', 'mean_whitespace_tokens', 'mean_sentence_segments', 'mean_document_ttr'),
          ((role, len(values), *(sum(col) / len(values) for col in zip(*values))) for role, values in sorted(stats.items())))
    write('table_06_distribution.csv', ('generator', 'disease', 'intended_role', 'bulk', 'topup', 'selected'),
          ((*cell, sum(stage == 'bulk' for stage, _ in values), sum(stage == 'topup' for stage, _ in values), len(values))
           for cell, values in sorted(cells.items())))
    top50 = {}
    for role, counts in features.items():
        size, outside = len(stats[role]), len(selected) - len(stats[role])
        def score(term):
            inside_count = counts[term]
            outside_count = all_features[term] - inside_count
            return math.log((inside_count + .5) / (size - inside_count + .5)) - math.log((outside_count + .5) / (outside - outside_count + .5))
        ranked = sorted((term for term in counts if counts[term] >= 5), key=lambda term: (-score(term), repr(term)))
        top50[role] = set(ranked[:50])
    write('table_16_lexical_overlap.csv', ('role_1', 'role_2', 'jaccard_percent'),
          ((a, b, 100 * len(x & y) / len(x | y)) for a, x in sorted(top50.items()) for b, y in sorted(top50.items())))

    positive = {}
    neutral = {}
    lexical = {}
    for model, rows in scored.items():
        pos = [r for r in rows if r[0] != 'Otherwise']
        oth = [r for r in rows if r[0] == 'Otherwise']
        positive[model] = [(t, 100 * sum(r[1] >= t for r in pos) / len(pos)) for t in THRESHOLDS]
        neutral[model] = [(t, 100 * sum(r[1] >= .5 and r[2] < t for r in oth) / len(oth)) for t in THRESHOLDS]
        lexical[model] = [(t, 100 * sum(r[3] > t for r in oth) / len(oth)) for t in (0.0, .01, .02, .03, .04, .05, .08, .10)]
    plot(5, 'Intended-role screening sensitivity', 'Minimum intended-role entailment', positive, .50)
    plot(6, 'Otherwise screening sensitivity', 'Maximum positive-role entailment (exclusive)', neutral, .75)
    plot(7, 'Otherwise lexical warning sensitivity', 'Lexical marker fraction threshold', lexical, .03)

    rng = random.Random(42)
    sample = []
    for cell, values in sorted(cells.items()):
        bulk = [row for stage, row in values if stage == 'bulk']
        topups = [row for stage, row in values if stage == 'topup']
        n_bulk = min(len(bulk), max(3, 6 - len(topups)))
        sample.extend((*cell, 'bulk', row) for row in rng.sample(bulk, n_bulk))
        sample.extend((*cell, 'topup', row) for row in rng.sample(topups, 6 - n_bulk))
    rng.shuffle(sample)
    audit = [(f'HRB-{i:04d}', model, disease, role, stage, row) for i, (model,disease,role,stage,row) in enumerate(sample, 1)]
    assert len(audit) == 450
    write('audit_blinded.csv', ('audit_id', 'post', 'primary_role', 'secondary_roles', 'ambiguous_or_unsuitable', 'notes'),
          ((audit_id, row['text'], '', '', '', '') for audit_id, _, _, _, _, row in audit), private=True)
    write('audit_key_private.csv', ('audit_id', 'generator', 'disease', 'intended_role', 'stage', 'sample_id'),
          ((audit_id, model, disease, role, stage, row['sample_id']) for audit_id, model, disease, role, stage, row in audit), private=True)
    manifest = {'status': 'machine-curated, pending human validation and near-duplicate review', 'input_sha256': inputs,
                'audit_seed': 42, 'figure_denominator': 'Bulk records passing non-NLI quality checks, separately by category and generator',
                'table_16_method': 'Jaccard of top 50 smoothed document log-odds unigrams/bigrams, minimum 5 in-class documents'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10), encoding='utf-8')
    print('Completed:', OUT)
    for name in ('table_02_curation.csv', 'table_05_linguistic.csv'):
        print(name, (OUT / name).read_text(), sep=chr(10))


if __name__ == '__main__':
    main()
