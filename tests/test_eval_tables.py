"""End-to-end check of the evaluation-set tables on a small synthetic project (no private data)."""
import csv
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from healthrolebench.reporting.draft_tables import markdown_rows
from healthrolebench.reporting.eval_tables import CLASSIFIERS, export
from healthrolebench.common.labels import ROLES

GENERATORS = ('gpt4o', 'qwen', 'llama2', 'mistral', 'gemma')


def write(path, header, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        csv.writer(handle).writerows([header, *records])


def build(root, rng):
    items = [(f'EVAL-{i:04d}', GENERATORS[i % 5], ROLES[(i // 5) % 5]) for i in range(250)]
    folder = root / 'data/experiments/eval_set_v1'
    write(folder / 'eval_key_private.csv', ('item_id', 'generator', 'sample_id', 'disease', 'intended_role', 'group'),
          [(i, g, i, 'HPV', r, n) for n, (i, g, r) in enumerate(items)])
    write(folder / 'eval_posts_private.csv', ('item_id', 'post'), [(i, f'post {i}') for i, _, _ in items])
    (folder / 'manifest.json').write_text('{}')
    answer = lambda role, accuracy: (role + '. Because.') if rng.random() < accuracy else rng.choice(ROLES + ('Not sure',))
    for condition, accuracy, subset in (('original', .8, 250), ('cue-masked', .75, 250), ('paraphrased', .7, 200), ('adversarial', .6, 180)):
        for model in CLASSIFIERS:
            write(root / 'data/experiments/zero_shot_eval_v1' / condition / f'{model}.csv', ('item_id', 'raw_completion'),
                  [(i, answer(r, accuracy)) for i, _, r in items[:subset]])
    probe = root / 'data/experiments/probe_full_v1'
    write(probe / 'table_08_full.csv', ('held_generator', 'classifier', 'n', 'macro_f1', 'macro_f1_sd', 'accuracy', 'macro_precision', 'macro_recall'),
          [(g, m, 12000, .85, .01, .86, .85, .85) for m in CLASSIFIERS[:4] for g in GENERATORS if g != m])
    write(probe / 'eval_predictions.csv', ('classifier', 'held_generator', 'seed', 'item_id', 'prediction'),
          [(m, g, s, i, r if rng.random() < .9 else 'Otherwise') for m in CLASSIFIERS[:4] for s in (42, 43, 44)
           for i, g, r in items if g != m])
    return items


class EvalTablesTest(unittest.TestCase):
    def test_export_and_render(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root, random.Random(0))
            export(root)
            out = root / 'data/experiments/eval_tables_v1'
            for name in ('table_08_cross_model.csv', 'table_09_zero_shot.csv', 'table_10_zero_shot_vs_head.csv',
                         'table_13_prediction_overlap.csv', 'table_18_stress.csv', 'figure_8.csv', 'figure_8.pdf'):
                self.assertTrue((out / name).is_file(), name)
            with (out / 'table_10_zero_shot_vs_head.csv').open() as handle:
                table10 = list(csv.DictReader(handle))
            self.assertEqual(len(table10), 4)
            self.assertTrue(all(float(r['head_macro_f1']) > float(r['zero_shot_macro_f1']) for r in table10))
            with (out / 'table_18_stress.csv').open() as handle:
                stress = list(csv.DictReader(handle))
            self.assertEqual({r['condition'] for r in stress}, {'original', 'cue-masked', 'paraphrased', 'adversarial'})
            self.assertEqual({r['n'] for r in stress if r['condition'] == 'adversarial'}, {'180'})
            self.assertTrue(all(int(r['role_support']) > 0 for r in stress))
            for table in (8, 9, 10, 13, 18):     # every manuscript table renders with numbers, not N/A placeholders
                rendered = markdown_rows(table, root / 'data/experiments')
                body = [line for line in rendered if line.startswith('| ') and '---' not in line][1:]
                self.assertTrue(body and not all('N/A' in line for line in body), table)
            self.assertEqual(json.loads((out / 'manifest.json').read_text())['parser'], 'labels-v2')


if __name__ == '__main__':
    unittest.main()
