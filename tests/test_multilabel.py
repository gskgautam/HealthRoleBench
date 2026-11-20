import csv
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from healthrolebench.curation.human_audit import ROLES
from healthrolebench.analysis.multilabel import SETTINGS, main

WORDS = {'Misinformation Spreader': 'cousin forwarded rumor heard', 'Religious Conspiracy Theorist': 'god punishment sin prayer',
         'Fear Monger': 'deadly panic collapse dying', 'Anti-Vaccine Advocate': 'pharma toxins injury natural',
         'Otherwise': 'recommended clinic schedule evidence'}


def write(path, header, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        csv.writer(handle).writerows([header, *records])


class MultilabelTest(unittest.TestCase):
    def test_four_settings_are_scored_from_synthetic_annotations(self):
        rng = random.Random(0)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = [(f'HRB-{i:04d}', ROLES[i % 5]) for i in range(100)]
            posts = [(k, f'post {WORDS[r]} {rng.choice(list(WORDS.values()))} vaccine') for k, r in items]
            write(root / 'data/experiments/corpus_tables_v1/audit_blinded.csv', ('audit_id', 'post'), posts)
            write(root / 'data/experiments/corpus_tables_v1/audit_key_private.csv',
                  ('audit_id', 'generator', 'disease', 'intended_role', 'stage', 'sample_id'),
                  [(k, 'gpt4o', 'HPV', r, 'bulk', k) for k, r in items])
            audit = root / 'annotations/healthrolebench_audit_v1'
            second = lambda r: '' if r == 'Otherwise' else ('Fear Monger' if r == 'Anti-Vaccine Advocate' else '')
            for name in ('annotator_a.csv', 'annotator_b.csv'):
                write(audit / name, ('audit_id', 'post', 'primary_role', 'secondary_roles', 'notes'),
                      [(k, '', r, second(r) if name.endswith('a.csv') else '', '') for k, r in items])
            write(audit / 'adjudicated.csv', ('audit_id', 'primary_role', 'secondary_roles', 'notes'),
                  [(k, r, second(r), '') for k, r in items])
            main(root)
            with (root / 'data/experiments/human_audit_v1/table_17_multilabel_evaluation.csv').open() as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual([r['setting'] for r in table], list(SETTINGS))
            self.assertTrue(all(0 <= float(r['macro_f1']) <= 1 for r in table))
            self.assertGreater(float(table[0]['macro_f1']), .5)


if __name__ == '__main__':
    unittest.main()
