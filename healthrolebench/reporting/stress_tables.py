import argparse
import csv
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from healthrolebench.common.data import rows
from healthrolebench.curation.definition_tables import read_lexicons
from healthrolebench.common.labels import INVALID as REJECT, PARSER_VERSION, parse

MODELS = ('llama2', 'mistral', 'qwen', 'gemma')
API_MODELS = ('gpt4o',)  # zero-shot only; included in Table 18 when both conditions exist


def masker(protocol):
    markers = {term for terms in read_lexicons(protocol).values() for term in terms}
    patterns = [re.escape(term.replace(chr(8217), chr(39))).replace(r"\ ", r"\s+").replace(chr(39), "[" + chr(39) + chr(8217) + "]")
                for term in sorted(markers, key=lambda term: (-len(term), term))]
    return re.compile(r'(?<!\w)(?:' + '|'.join(patterns) + r')(?!\w)', flags=re.IGNORECASE)


def decode(text, labels):
    return parse(text, tuple(labels))


def read_predictions(path, expected, labels):
    results = {}
    for row in rows(path):
        item = row['audit_id']
        if item in results:
            raise ValueError(f'Duplicate prediction ID in {path}')
        results[item] = decode(row['raw_completion'], labels)
    if results.keys() != expected:
        raise ValueError(f'Prediction IDs differ from the frozen audit sample: {path}')
    return results


def report(root, run='zero_shot_audit_v2', stress_run='zero_shot_cue_mask_v2'):
    base = root/'data/experiments'/run
    stress = root/'data/experiments'/stress_run
    sample = root/'data/experiments/corpus_tables_v1'
    gold = {r['audit_id']:r['intended_role'] for r in rows(sample/'audit_key_private.csv')}
    mask = masker(root/'configs/filter_v1/protocol.json')
    hits = {r['audit_id']:bool(mask.search(r['post'])) for r in rows(sample/'audit_blinded.csv')}
    if len(gold) != 450 or gold.keys() != hits.keys():
        raise ValueError('Audit labels and text IDs differ')
    labels = sorted(set(gold.values()))
    results, role_results, unchanged = [], [], []
    classifiers = [*MODELS, *(m for m in API_MODELS if (base/f'{m}.csv').is_file() and (stress/f'{m}.csv').is_file())]
    for model in classifiers:
        original = read_predictions(base/f'{model}.csv', gold.keys(), labels)
        masked = read_predictions(stress/f'{model}.csv', gold.keys(), labels)
        controls = [item for item in gold if not hits[item]]
        unchanged.append((model,len(controls),sum(original[i] != masked[i] for i in controls)))
        for subset, ids in (('all', list(gold)), ('changed_text', [item for item in gold if hits[item]])):
            if not ids:
                continue
            true = [gold[i] for i in ids]
            estimates = [[prediction[i] for i in ids] for prediction in (original, masked)]
            for condition, guessed in zip(('original','cue_masked'), estimates):
                results.append((model,subset,condition,len(ids),sum(hits[i] for i in ids),
                                f1_score(true,guessed,labels=labels,average='macro',zero_division=0),
                                accuracy_score(true,guessed),sum(p == REJECT for p in guessed)))
                f1_by_role = f1_score(true,guessed,labels=labels,average=None,zero_division=0)
                for role, f1 in zip(labels, f1_by_role):
                    support = sum(label == role for label in true)
                    if support:
                        role_results.append((model,subset,condition,role,support,float(f1)))
    exports = (
        ('table_18_cue_masking_pilot.csv', ('classifier','subset','condition','n','changed_n','macro_f1','accuracy','invalid_n'), results),
        ('table_18_cue_masking_by_role.csv', ('classifier','subset','condition','intended_role','support','f1'), role_results),
        ('table_18_unchanged_controls.csv', ('classifier','unchanged_n','label_disagreements'), unchanged),
    )
    out = stress/PARSER_VERSION
    out.mkdir(exist_ok=True)
    for filename, columns, data in exports:
        target = out/filename
        if target.exists():
            existing = [tuple(row.values()) for row in rows(target)]
            if existing != [tuple(map(str,row)) for row in data]:
                raise ValueError(f'Existing output disagrees with the paired scoring: {target}')
            continue
        with target.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(data)
    print('Saved Table 18 paired cue-masking pilot. Changed texts:', sum(hits.values()), '/', len(hits))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Report the paired lexical cue-masking evaluation.')
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--run', default='zero_shot_audit_v2', help='Original-condition zero-shot run directory')
    parser.add_argument('--stress-run', default='zero_shot_cue_mask_v2', help='Cue-masked zero-shot run directory')
    args = parser.parse_args()
    report(args.project_root.resolve(), args.run, args.stress_run)
