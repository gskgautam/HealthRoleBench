import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from healthrolebench.common.data import DISEASES, digest, rows


def write_csv(path, header, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def validate_confusion(matrix, expected):
    size = len(matrix)
    if not size or any(len(row) != size or min(row) < 0 for row in matrix):
        raise ValueError('Invalid confusion matrix')
    if sum(map(sum, matrix)) != expected:
        raise ValueError('Confusion count differs from test support')
    true = [sum(row) for row in matrix]
    predicted = [sum(column) for column in zip(*matrix)]
    if any(group <= 0 for group in true):
        raise ValueError('Confusion matrix is missing a true role')
    fp = [predicted[i] - matrix[i][i] for i in range(size)]
    fn = [true[i] - matrix[i][i] for i in range(size)]
    if sum(fp) != sum(fn):
        raise ValueError('Total false positives must equal total false negatives')
    return true, fp, fn


def full_transfer(root):
    source = root / 'data/experiments/full_transfer_pilot_v1'
    saved = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
    labels = saved['classes']
    confusion, errors, entropy, transfer = [], [], [], []
    for fold in saved['results']:
        key = fold['protocol'], fold['source'], fold['target']
        matrix = fold['confusion_matrix']
        n = fold['test_n']
        true, fp, fn = validate_confusion(matrix, n)
        transfer.append((*key, fold['train_n'], n, fold['excluded_train'], fold['accuracy'],
                         fold['report']['macro avg']['f1-score']))
        for i, label in enumerate(labels):
            tp = matrix[i][i]
            tn = n - tp - fp[i] - fn[i]
            confusion.append((*key, label, true[i], tp, fp[i], fn[i], tn))
            proportions = [j / true[i] for j in matrix[i]]
            entropy.append((*key, label, -sum(p*math.log2(p) for p in proportions if p)))
            errors.extend((*key, label, other, matrix[i][j], proportions[j])
                          for j, other in enumerate(labels) if j != i)
    keys = ('protocol','source','target')
    write_csv(source/'table_07_grouped_confusion_pilot.csv', (*keys,'role','support','tp','fp','fn','tn'), confusion)
    write_csv(source/'table_11_grouped_transfer_pilot.csv', (*keys,'train_n','test_n','excluded_train','accuracy','macro_f1'), transfer)
    write_csv(source/'table_13_grouped_entropy_pilot.csv', (*keys,'role','entropy_bits'), entropy)
    write_csv(source/'table_14_grouped_errors_pilot.csv', (*keys,'role','incorrect_prediction','n','fraction_all_true_role'), errors)
    print('Full transfer pilot tables written from', len(saved['results']), 'verified folds')


def fold_coverage(root):
    """Supplementary disclosure: training coverage per held-out disease after leakage-group purging.

    Exemplar links and near-duplicate pairs are merged transitively, so a group that touches the
    held-out disease is removed from training entirely. This lists, for every generator x role,
    how many training-disease records remain available for training in each disease fold.
    """
    source = root / 'data/experiments/full_transfer_pilot_v1'
    records = list(rows(source / 'groups.csv'))
    out = []
    for held in DISEASES:
        test_groups = {r['group'] for r in records if r['disease'] == held}
        candidates = Counter((r['generator'], r['role']) for r in records if r['disease'] != held)
        kept = Counter((r['generator'], r['role']) for r in records if r['disease'] != held and r['group'] not in test_groups)
        out.extend((held, generator, role, n, kept[generator, role], kept[generator, role] / n)
                   for (generator, role), n in sorted(candidates.items()))
    write_csv(source / 'table_s1_fold_training_coverage.csv',
              ('held_out_disease', 'generator', 'role', 'candidates', 'kept_for_training', 'kept_fraction'), out)
    low = [r for r in out if r[-1] < .5]
    print('Fold coverage written;', len(low), 'generator-role cells keep <50% in some fold')


# Prompt-template text that survived curation (REVIEW.md A6; kept in the corpus and disclosed).
ECHO_MARKERS = ('new post:', 'write the requested post', 'examples of the intended writing style', 'disease context:',
                'character and rhetoric', 'output only the post', 'use 45-100 words', 'academic research collection',
                'fictional character')


def prompt_echo(root):
    """Supplementary Table S2: selected posts containing prompt-template text, with the first marker found."""
    import re
    csv.field_size_limit(1 << 26)
    pattern = re.compile('|'.join(re.escape(m) for m in ECHO_MARKERS), re.I)
    found = []
    for model in ('gpt4o', 'qwen', 'llama2', 'mistral', 'gemma'):
        for stage, path in (('bulk', root / 'data/filter_runs/curation_v2' / model / 'selected_machine_curated_draft.csv'),
                            ('topup', root / 'data/topups_v2' / f'{model}_candidates.csv')):
            for r in rows(path):
                hit = pattern.search(r['text'])
                if hit:
                    found.append((model, stage, r['sample_id'], r['disease'], r['intended_role'], hit.group(0).lower(),
                                  r['text'].lower().count('new post:')))
    out = root / 'data/experiments/supplementary_v1'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'table_s2_prompt_echo.csv', ('generator', 'stage', 'sample_id', 'disease', 'intended_role',
                                                 'first_marker', 'new_post_count'), found)
    by_generator = Counter(f[0] for f in found)
    print('Prompt-echo posts:', len(found), dict(by_generator))


def verify_encoder_predictions(path, metrics):
    labels = metrics['labels']
    if len(labels) != 5 or len(set(labels)) != len(labels):
        raise ValueError(f'Invalid class inventory: {path}')
    split = {}
    group_partitions = {}
    for row in rows(path / 'split.csv'):
        key = row['generator'], row['sample_id']
        partition, group = row['partition'], row['group']
        if key in split or partition not in ('train', 'validation', 'test'):
            raise ValueError(f'Invalid split row: {path}')
        if group_partitions.setdefault(group, partition) != partition:
            raise ValueError(f'An exemplar or near-duplicate group crosses partitions: {path}')
        split[key] = partition
    support = Counter(split.values())
    for partition in ('train', 'validation', 'test'):
        if support[partition] != metrics[f'{partition}_n']:
            raise ValueError(f'Split counts disagree with metrics: {path}')
    label_index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    seen = set()
    for row in rows(path / 'predictions.csv'):
        key = row['generator'], row['sample_id']
        if key in seen or split.get(key) != 'test':
            raise ValueError(f'Repeated or non-test prediction: {path}')
        seen.add(key)
        matrix[label_index[row['label']]][label_index[row['prediction']]] += 1
    if len(seen) != support['test'] or matrix != metrics['confusion_matrix']:
        raise ValueError(f'Prediction coverage or confusion matrix mismatch: {path}')
    true, fp, fn = validate_confusion(matrix, support['test'])
    scores = [2 * matrix[i][i] / (2 * matrix[i][i] + fp[i] + fn[i]) for i in range(len(true))]
    macro_f1 = statistics.mean(scores)
    if not math.isclose(macro_f1, metrics['classification_report']['macro avg']['f1-score'], abs_tol=1e-12):
        raise ValueError(f'Macro-F1 differs from the saved predictions: {path}')
    return macro_f1


def encoder_summary(root):
    source = root / 'data/experiments/encoder_transfer_v1'
    models = ('bert','roberta','distilbert')
    seeds = (42,43,44)
    directories = {(m,d,s): source / m / d / f'seed_{s}' for m in models for d in DISEASES for s in seeds}
    missing = [key for key, path in directories.items() if not (path / 'metrics.json').is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(directories)} runs are incomplete. First pending run: {missing[0]}')
    split_hash, scores = {}, {}
    provenance = None
    for (m,d,s), path in directories.items():
        metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        if (metrics['disease'], metrics['train_seed']) != (d,s):
            raise ValueError(f'Run identity differs from folder: {path}')
        if provenance is None:
            provenance = metrics['provenance']
        if metrics['provenance'] != provenance:
            raise ValueError(f'Encoder runs use different data or group manifests: {path}')
        audit = digest(path / 'split.csv')
        if split_hash.setdefault(d,audit) != audit:
            raise ValueError(f'Models or seeds used different partitions for {d}')
        scores[m,d,s] = verify_encoder_predictions(path, metrics)
    rows = []
    for model in models:
        for domain in (*DISEASES,'unweighted mean across diseases'):
            values = [scores[model,domain,seed] if domain in DISEASES else
                      statistics.mean(scores[model,d,seed] for d in DISEASES) for seed in seeds]
            rows.append((model,domain,*values,statistics.mean(values),statistics.stdev(values)))
    write_csv(source/'table_12_encoder_macro_f1.csv',
              ('model','held_out','seed_42','seed_43','seed_44','mean','sample_sd'), rows)
    print('Table 12 exported from all 27 complete, split-matched runs')


def main():
    parser = argparse.ArgumentParser(description='Export manuscript tables from verified evaluation outputs.')
    parser.add_argument('report', choices=('transfer','encoders','coverage','echo'))
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args()
    {'transfer':full_transfer, 'encoders':encoder_summary, 'coverage':fold_coverage, 'echo':prompt_echo}[args.report](args.project_root.resolve())


if __name__ == '__main__':
    main()
