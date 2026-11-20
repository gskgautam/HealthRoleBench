import argparse
import csv
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score, hamming_loss, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

from healthrolebench.common.artifacts import private_text_artifact
from healthrolebench.common.data import rows
from healthrolebench.curation.definition_tables import TAXONOMY

ROLES = tuple(role for role, _, _ in TAXONOMY) + ('Otherwise',)
UNSURE = 'UNSURE'


def index(path):
    result = {}
    for row in rows(path):
        item = row['audit_id']
        if not item or item in result:
            raise ValueError(f'Blank or duplicate audit ID: {path}')
        result[item] = row
    if not result:
        raise ValueError(f'No rows in {path}')
    return result


def read_ratings(path, expected):
    result = index(path)
    if set(result) != set(expected):
        raise ValueError(f'Audit IDs differ from the expected sample: {path}')
    return {item: labels_from_row(row) for item, row in result.items()}


def labels_from_row(row):
    primary = row['primary_role'].strip()
    secondary = [label.strip() for label in row.get('secondary_roles', '').split('|') if label.strip()]
    if (primary not in (*ROLES, UNSURE) or any(label not in ROLES for label in secondary)
            or len(set(secondary)) != len(secondary) or primary in secondary
            or ('Otherwise' in secondary) or (primary in ('Otherwise', UNSURE) and secondary)):
        raise ValueError(f'Invalid or missing role label for {row["audit_id"]}')
    return primary, tuple(secondary)


def write(path, header, records):
    with private_text_artifact(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(records)


def templates(root):
    audit = root/'data/experiments/corpus_tables_v1/audit_blinded.csv'
    sample = index(audit)
    if len(sample) != 450:
        raise ValueError('Expected the 450-row frozen audit sample')
    target = root/'annotations/healthrolebench_audit_v1'
    target.mkdir(parents=True)
    header = ('audit_id','post','primary_role','secondary_roles','notes')
    for name in ('annotator_a.csv', 'annotator_b.csv'):
        write(target/name, header, ((item, row['post'], '', '', '') for item,row in sample.items()))
    write(target/'adjudicated.csv', ('audit_id','primary_role','secondary_roles','notes'), ((item,'','','') for item in sample))
    print('Prepared two blinded rating files and one adjudication file:', target)


def score(root, a, b, adjudicated, predictions):
    key = index(root/'data/experiments/corpus_tables_v1/audit_key_private.csv')
    ratings_a, ratings_b, consensus = (read_ratings(path, key) for path in (a,b,adjudicated))
    table = []
    for bucket in (*ROLES, 'Overall'):
        ids = [item for item, row in key.items() if bucket == 'Overall' or row['intended_role'] == bucket]
        resolved = [item for item in ids if consensus[item][0] != UNSURE]
        left, right = ([ratings[item][0] for item in ids] for ratings in (ratings_a, ratings_b))
        agreement = sum(a == b for a,b in zip(left,right))
        kapp = None if len(set(left + right)) == 1 else cohen_kappa_score(left, right, labels=(*ROLES, UNSURE))
        label_matches = sum(consensus[item][0] == key[item]['intended_role'] for item in resolved)
        table.append((bucket,len(ids),len(resolved),len(ids)-len(resolved),agreement/len(ids),kapp,
                      label_matches/len(resolved) if resolved else None))
    reports = []
    if predictions:
        resolved = {item: label for item,label in consensus.items() if label[0] != UNSURE}
        if not resolved:
            raise ValueError('At least one resolved example is needed to score predictions')
        binarizer = MultiLabelBinarizer(classes=ROLES)
        actual = binarizer.fit_transform([(first,*secondary) for first,secondary in resolved.values()])
        settings = set()
        for setting, filename in predictions:
            if setting in settings:
                raise ValueError(f'Repeated evaluation setting: {setting}')
            settings.add(setting)
            pred = read_ratings(Path(filename), resolved)
            if any(label[0] == UNSURE for label in pred.values()):
                raise ValueError('Model predictions must specify a role')
            guessed = binarizer.transform([(pred[item][0],*pred[item][1]) for item in resolved])
            reports.append((setting,len(resolved),f1_score(actual,guessed,average='macro',zero_division=0),
                            f1_score(actual,guessed,average='micro',zero_division=0),hamming_loss(actual,guessed),
                            jaccard_score(actual,guessed,average='samples',zero_division=0)))
    destination = root/'data/experiments/human_audit_v1'
    destination.mkdir(exist_ok=True)
    table_03 = destination/'table_03_human_validation.csv'
    if not table_03.exists():
        write(table_03, ('role','n','resolved_n','unresolved_n','raw_agreement','cohen_kappa','adjudicated_intended_agreement'), table)
    else:
        existing = [tuple(row.values()) for row in rows(table_03)]
        if existing != [tuple('' if v is None else str(v) for v in row) for row in table]:
            raise ValueError('The human ratings differ from the existing Table 3 run')
    if reports:
        write(destination/'table_17_multilabel_evaluation.csv', ('setting','n','macro_f1','micro_f1','hamming_loss','samples_jaccard'), reports)
    print('Saved human audit metrics:', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare blinded ratings and score completed independent annotations.')
    parser.add_argument('command', choices=('templates','score'))
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--annotator-a', type=Path)
    parser.add_argument('--annotator-b', type=Path)
    parser.add_argument('--adjudicated', type=Path)
    parser.add_argument('--predictions', action='append', nargs=2, metavar=('SETTING','CSV'))
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.command == 'templates':
        templates(root)
    else:
        if not (args.annotator_a and args.annotator_b and args.adjudicated):
            parser.error('score requires two completed ratings files and adjudication')
        score(root,args.annotator_a,args.annotator_b,args.adjudicated,args.predictions)
