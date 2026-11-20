import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from healthrolebench.common.data import digest, rows

# Conceptual assignments from the original taxonomy table, separate from empirical counts.
TAXONOMY = (
    ('Religious Conspiracy Theorist', 'Moral–Ideological', 'Limited (Distinct Moral Framing)'),
    ('Anti-Vaccine Advocate', 'Ideological', 'Fear Monger'),
    ('Fear Monger', 'Emotional', 'Anti-Vaccine Advocate'),
    ('Misinformation Spreader', 'Low-Intent', 'Fear Monger'),
)


def read_lexicons(protocol):
    lexicons = json.loads(protocol.read_text(encoding='utf-8'))['F3']['lexicons']
    if set(lexicons) != {role for role, _, _ in TAXONOMY}:
        raise ValueError('Curation lexicon role names differ from the approved taxonomy')
    if any(not terms or len(set(terms)) != len(terms) or any(not t.strip() for t in terms) for terms in lexicons.values()):
        raise ValueError('A lexicon contains missing or duplicate entries')
    return lexicons


def write_csv(path, header, records):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(records)


def export(root):
    data = root / 'data/experiments'
    distribution = data / 'corpus_tables_v1/table_06_distribution.csv'
    protocol = root / 'configs/filter_v1/protocol.json'
    counts = Counter()
    for row in rows(distribution):
        counts[row['intended_role']] += int(row['selected'])
    if set(counts) != {role for role, _, _ in TAXONOMY} | {'Otherwise'}:
        raise ValueError('Unexpected role inventory in the corpus CSV')
    lexicons = read_lexicons(protocol)
    output = data / 'definition_tables_v1'
    output.mkdir()
    write_csv(output / 'table_01_taxonomy.csv', ('role','alignment','potential_overlap','machine_curated_n'),
              ((*definition, counts[definition[0]]) for definition in TAXONOMY))
    write_csv(output / 'table_04_lexical_markers.csv', ('role','marker_order','lexical_marker'),
              ((role, i, term) for role, _, _ in TAXONOMY for i, term in enumerate(lexicons[role], 1)))
    info = dict(inputs={str(p.relative_to(root)): digest(p) for p in (distribution, protocol)},
                taxonomy_definition='Original manuscript taxonomy; counts joined to the reconstructed corpus CSV',
                otherwise_n=counts['Otherwise'], lexical_marker_use='Diagnostic warning in v2',
                lexical_marker_counts={role:len(terms) for role, terms in lexicons.items()})
    (output / 'manifest.json').write_text(json.dumps(info, indent=2) + chr(10), encoding='utf-8')
    print('Wrote Tables 1 and 4', dict(counts), info['lexical_marker_counts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the approved taxonomy and actual curation lexicon.')
    parser.add_argument('--project-root', type=Path, required=True)
    export(parser.parse_args().project_root.resolve())
