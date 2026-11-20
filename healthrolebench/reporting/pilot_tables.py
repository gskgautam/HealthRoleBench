import csv
import itertools
import json
import math
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

ROOT = Path('data/experiments')
PILOT = ROOT / 'bulk_transfer_pilot_v1'
AUDIT = ROOT / 'corpus_tables_v1'
from healthrolebench.common.labels import INVALID as REJECT, PARSER_VERSION, ROLES, parse


def read(path):
    with path.open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def write(folder, name, columns, rows):
    with (folder / name).open('x', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(columns)
        writer.writerows(rows)


def transfer_tables():
    metadata = json.loads((PILOT / 'metrics.json').read_text())
    labels = metadata['labels']
    results = metadata['results']
    confusion_rows, error_rows, entropy_rows = [], [], []
    for fold in results:
        matrix = fold['confusion_matrix']
        support = [sum(row) for row in matrix]
        predicted = [sum(col) for col in zip(*matrix)]
        fp_total, fn_total = 0, 0
        for i, label in enumerate(labels):
            tp = matrix[i][i]
            fp, fn = predicted[i] - tp, support[i] - tp
            tn = fold['test_n'] - tp - fp - fn
            assert min(tp, fp, fn, tn) >= 0
            fp_total, fn_total = fp_total + fp, fn_total + fn
            confusion_rows.append((fold['protocol'], fold['held_out'], label, support[i], tp, fp, fn, tn))
            rates = [cell / support[i] for cell in matrix[i]]
            error_rows.extend((fold['protocol'], fold['held_out'], label, target, matrix[i][j], rates[j])
                              for j, target in enumerate(labels) if j != i)
            entropy_rows.append((fold['protocol'], fold['held_out'], label,
                                 -sum(p * math.log2(p) for p in rates if p)))
        assert fp_total == fn_total
    write(PILOT, 'table_07_confusion_pilot.csv', ('protocol','held_out','role','support','tp','fp','fn','tn'), confusion_rows)
    write(PILOT, 'table_11_transfer_pilot.csv', ('protocol','held_out','train_n','test_n','f1_macro','accuracy'),
          ((f['protocol'], f['held_out'], f['train_n'], f['test_n'], f['report']['macro avg']['f1-score'], f['accuracy'])
           for f in results))
    write(PILOT, 'table_13_confusion_entropy_pilot.csv', ('protocol','held_out','role','entropy_bits'), entropy_rows)
    write(PILOT, 'table_14_error_distribution_pilot.csv', ('protocol','held_out','role','incorrect_prediction','n','fraction_all_true_role'), error_rows)
    examples = {}
    for row in read(PILOT / 'predictions.csv'):
        pair = (row['protocol'], row['held_out'], row['intended_role'], row['prediction'])
        if row['intended_role'] != row['prediction']:
            examples.setdefault(pair, (row['generator'], row['sample_id'], row['disease']))
    write(PILOT, 'table_15_error_case_ids_pilot.csv', ('protocol','held_out','intended_role','prediction','generator','sample_id','disease'),
          ((*pair, *metadata) for pair, metadata in sorted(examples.items())))
    print('Verified FP = FN in all eight transfer folds; wrote five pilot tables.')


def clean(text):
    text = next((line.strip() for line in text.splitlines() if line.strip()), '')
    text = re.sub(r'^\s*(?:category|role|answer)\s*:\s*', '', text, flags=re.I)
    canonical = text.casefold().strip(string.punctuation + string.whitespace)
    return ' '.join(re.sub(r'[^\w\s-]', '', canonical).split())


def edit_distance(a, b):
    last = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, last[j] + 1, last[j-1] + (char != other)))
        last = current
    return last[-1]


def zero_shot_tables(run='zero_shot_audit_v2'):
    """Score saved zero-shot completions with the shared v2 label parser.

    Reads data/experiments/<run>/<model>.csv and writes derived tables to
    data/experiments/<run>/labels-v2/, leaving earlier scorings untouched.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = ROOT / run
    out = source / PARSER_VERSION
    out.mkdir(exist_ok=True)
    truth = {row['audit_id']: row for row in read(AUDIT / 'audit_key_private.csv')}
    assert len(truth) == 450
    labels = sorted({r['intended_role'] for r in truth.values()})
    assert tuple(labels) == ROLES
    domains = ['all', *sorted({r['generator'] for r in truth.values()})]
    predictions, sweep, by_generator = {}, [], []
    # Open models are required; the GPT-4o API run is included when present.
    classifiers = ['llama2', 'mistral', 'qwen', 'gemma'] + (['gpt4o'] if (source / 'gpt4o.csv').is_file() else [])
    for model in classifiers:
        values = read(source / f'{model}.csv')
        assert len(values) == 450 and {row['audit_id'] for row in values} == set(truth), model
        completions = {row['audit_id']: row['raw_completion'] for row in values}
        for threshold in range(5):
            pred = {k: parse(text, labels, threshold) for k, text in completions.items()}
            gold = [truth[k]['intended_role'] for k in pred]
            coverage = sum(label != REJECT for label in pred.values()) / len(gold)
            accuracy = sum(a == b for a, b in zip(gold, pred.values())) / len(gold)
            f1 = f1_score(gold, list(pred.values()), labels=labels, average='macro', zero_division=0)
            sweep.append((model, threshold, coverage, accuracy, f1))
            if threshold == 2:
                predictions[model] = pred
                for domain in domains:
                    ids = [k for k, row in truth.items() if domain == 'all' or row['generator'] == domain]
                    actual = [truth[k]['intended_role'] for k in ids]
                    guessed = [pred[k] for k in ids]
                    value = f1_score(actual, guessed, labels=labels, average='macro', zero_division=0)
                    by_generator.append((domain, model, len(ids), value, sum(x == REJECT for x in guessed)))
    write(out, 'table_09_zero_shot_pilot.csv', ('generator','classifier','n','macro_f1','invalid'), by_generator)
    overlap = []
    for (a, p), (b, q) in itertools.combinations(predictions.items(), 2):
        # Agreement is measured only where both classifiers returned a valid label.
        shared = [k for k in truth if p[k] != REJECT and q[k] != REJECT]
        left, right = [p[k] for k in shared], [q[k] for k in shared]
        per_role = f1_score(left, right, labels=labels, average=None, zero_division=0) if shared else [0.0] * len(labels)
        overlap.extend((a, b, role, len(shared), score) for role, score in zip(labels, per_role))
    write(out, 'table_13_pairwise_overlap_pilot.csv', ('classifier_1','classifier_2','role','shared_valid_n','prediction_overlap_f1'), overlap)
    write(out, 'figure_8_pilot.csv', ('classifier','edit_distance','valid_label_rate','accuracy','macro_f1'), sweep)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout='constrained', sharey=True)
    for model in predictions:
        rows = [r for r in sweep if r[0] == model]
        for j, (axis, title) in enumerate(zip(axes, ('Valid label rate','Accuracy','Macro F1')), start=2):
            axis.plot([r[1] for r in rows], [100*r[j] for r in rows], label=model, marker='o')
            axis.set(title=title, xlabel='Maximum edit distance (leading label)', xticks=range(5), ylim=(0,100))
    for axis in axes:
        axis.axvline(2, color='0.5', linestyle=':')
    axes[0].set_ylabel('Percent')
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(out / 'figure_8_pilot.pdf')
    plt.close(fig)
    print('Scored zero-shot run', run, 'with', PARSER_VERSION, '->', out)
    for domain, model, n, value, invalid in by_generator:
        if domain == 'all':
            print(model, n, round(value,4), 'invalid', invalid)


if __name__ == '__main__':
    if sys.argv[1:2] == ['zero']:
        zero_shot_tables(*sys.argv[2:3])
    elif sys.argv[1:2] == ['transfer']:
        transfer_tables()
    else:
        raise SystemExit('Usage: python -m healthrolebench.reporting.pilot_tables zero [RUN_DIR] | transfer')
