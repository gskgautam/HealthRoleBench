"""Tables 7, 11, 13 (confusion entropy), 14 and 15 from the standard-encoder runs.

Held-out-disease runs (encoder_transfer_v1: BERT, RoBERTa, DistilBERT x 3 diseases x 3 seeds,
the same runs as Table 12) give Tables 7, 13 (entropy), 14 and 15. Single-source runs
(encoder_source_v1: train on one disease, test on each other) give Table 11.

Aggregation (documented for the paper):
- Table 7: TP/FP/FN/TN per disease and role, averaged over the 9 runs (3 encoders x 3 seeds);
  Train (n) is the role's training count, identical across runs because every run of a disease
  uses the same group-disjoint split.
- Table 14: misclassification share of each true role into each other role, pooled over
  diseases and runs (percent of all test posts of the true role).
- Table 13: confusion entropy per role, -sum_c P(c|r) ln P(c|r) of the row-normalised confusion
  matrix (natural log), averaged over diseases and runs.
- Table 15: the five most frequent confusions of the best encoder (highest Table 12 mean),
  seed 42, pooled over diseases; one post per confusion, chosen by lowest SHA-256 of its ID.
- Table 11: Macro-F1 per (training disease, test disease), averaged over the 9 runs;
  aggregate accuracy, macro-precision and macro-recall averaged over the two test diseases.

Run from PROJECT_ROOT:  python -m healthrolebench.reporting.encoder_tables --project-root .
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.common.data import DISEASES, digest, load_corpus, rows
from healthrolebench.reporting.report_tables import verify_encoder_predictions

ENCODERS = ('bert', 'roberta', 'distilbert')
SEEDS = (42, 43, 44)
OUT = Path('data/experiments/encoder_tables_v1')


def write(path, header, records):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(records)


def held_out_runs(root):
    base = root / 'data/experiments/encoder_transfer_v1'
    runs = []
    for model in ENCODERS:
        for disease in DISEASES:
            for seed in SEEDS:
                path = base / model / disease / f'seed_{seed}'
                metrics = json.loads((path / 'metrics.json').read_text())
                verify_encoder_predictions(path, metrics)          # recomputes the matrix from predictions
                runs.append((model, disease, seed, metrics, path))
    return runs


def held_out_tables(root, out):
    runs = held_out_runs(root)
    labels = runs[0][3]['labels']
    role_of = {(r['generator'], r['sample_id']): r['role'] for r in rows(root / 'data/experiments/full_transfer_pilot_v1/groups.csv')}
    # Table 7
    table7 = []
    for disease in DISEASES:
        mats = [m['confusion_matrix'] for (_, d, _, m, _) in runs if d == disease]
        first = next(p for (_, d, _, _, p) in runs if d == disease)
        train = Counter(role_of[r['generator'], r['sample_id']] for r in rows(first / 'split.csv') if r['partition'] == 'train')
        n = sum(map(sum, mats[0]))
        for i, role in enumerate(labels):
            tp = mean(m[i][i] for m in mats)
            fn = mean(sum(m[i]) - m[i][i] for m in mats)
            fp = mean(sum(row[i] for row in m) - m[i][i] for m in mats)
            support = sum(mats[0][i])
            table7.append((disease, role, train[role], support, tp, fp, fn, n - tp - fp - fn))
    write(out / 'table_07_confusion_counts.csv', ('disease', 'role', 'train_n', 'test_n', 'tp', 'fp', 'fn', 'tn'), table7)
    # Table 14 and Table 13 (entropy)
    errors, support = Counter(), Counter()
    entropy = defaultdict(list)
    for (_, _, _, m, _) in runs:
        matrix = m['confusion_matrix']
        for i, role in enumerate(labels):
            total = sum(matrix[i])
            support[role] += total
            for j, other in enumerate(labels):
                if j != i:
                    errors[role, other] += matrix[i][j]
            entropy[role].append(-sum((c / total) * math.log(c / total) for c in matrix[i] if c))
    write(out / 'table_14_error_distribution.csv', ('true_role', 'predicted_role', 'n', 'percent_of_true_role'),
          [(a, b, errors[a, b], 100 * errors[a, b] / support[a]) for a in labels for b in labels if a != b])
    write(out / 'table_13_confusion_entropy.csv', ('role', 'confusion_entropy_nats', 'runs'),
          [(role, mean(entropy[role]), len(entropy[role])) for role in labels])
    # Table 15
    summary = {r['model']: float(r['mean']) for r in rows(root / 'data/experiments/encoder_transfer_v1/table_12_encoder_macro_f1.csv')
               if r['held_out'] == 'unweighted mean across diseases'}
    best = max(summary, key=summary.get)
    records, _ = load_corpus(root)
    text = {(r.generator, r.sample_id): (r.text, r.disease) for r in records}
    wrong = defaultdict(list)
    for (model, disease, seed, _, path) in runs:
        if model == best and seed == 42:
            for r in rows(path / 'predictions.csv'):
                if r['label'] != r['prediction']:
                    wrong[r['label'], r['prediction']].append((r['generator'], r['sample_id']))
    top = sorted(wrong, key=lambda pair: (-len(wrong[pair]), pair))[:5]
    cases = []
    for pair in top:
        chosen = min(wrong[pair], key=lambda k: hashlib.sha256('|'.join(k).encode()).hexdigest())
        post, disease = text[chosen]
        cases.append((disease, post, pair[0], pair[1], len(wrong[pair]), *chosen))
    with private_text_artifact(out / 'table_15_qualitative_errors.csv') as handle:
        writer = csv.writer(handle)
        writer.writerow(('disease', 'sample', 'true_role', 'predicted_role', 'pair_count', 'generator', 'sample_id'))
        writer.writerows(cases)
    return dict(table_15_encoder=best, held_out_runs=len(runs))


def source_tables(root, out):
    base = root / 'data/experiments/encoder_source_v1'
    paths = {(m, s, seed): base / m / s / f'seed_{seed}' / 'metrics.json' for m in ENCODERS for s in DISEASES for seed in SEEDS}
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        print(f'Table 11 skipped: {len(missing)} of {len(paths)} single-source runs missing (first: {missing[0]})')
        return dict(table_11='incomplete', missing=len(missing))
    cells = defaultdict(list)
    for (model, source, seed), path in paths.items():
        metrics = json.loads(path.read_text())
        for target, result in metrics['per_target'].items():
            report = result['classification_report']['macro avg']
            cells[source, target].append((model, seed, report['f1-score'], result['accuracy'], report['precision'], report['recall']))
    table = [(s, t, len(v), mean(x[2] for x in v), mean(x[3] for x in v), mean(x[4] for x in v), mean(x[5] for x in v))
             for (s, t), v in sorted(cells.items())]
    write(out / 'table_11_cross_disease.csv', ('train_disease', 'test_disease', 'runs', 'macro_f1', 'accuracy', 'macro_precision', 'macro_recall'), table)
    write(out / 'table_11_by_encoder.csv', ('train_disease', 'test_disease', 'model', 'seed', 'macro_f1', 'accuracy', 'macro_precision', 'macro_recall'),
          [(s, t, *x) for (s, t), v in sorted(cells.items()) for x in sorted(v)])
    return dict(table_11='complete', single_source_runs=len(paths), missing=0)


def main(root):
    """Write each table once; tables already present are kept (re-run to add Table 11 later)."""
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    info = json.loads(manifest.read_text()) if manifest.exists() else {}
    if not (out / 'table_07_confusion_counts.csv').exists():
        info.update(held_out_tables(root, out))
    if not (out / 'table_11_cross_disease.csv').exists():
        info.update(source_tables(root, out))
    info.update(runner_sha256=digest(Path(__file__)), entropy_log='natural', aggregation='mean over 3 encoders x 3 seeds')
    manifest.write_text(json.dumps(info, indent=2) + chr(10))
    print('Encoder tables written:', out, info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tables 7, 11, 13 (entropy), 14, 15 from encoder runs.')
    parser.add_argument('--project-root', required=True, type=Path)
    main(parser.parse_args().project_root.resolve())
