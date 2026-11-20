import unittest

from healthrolebench.common.data import Record, split_disease
from healthrolebench.reporting.pilot_tables import clean, edit_distance


class GroupedSplitTests(unittest.TestCase):
    def test_holdout_and_related_training_rows_are_disjoint(self):
        records = [Record('generator', str(i), 'HPV', str(i % 5), '', i) for i in range(100)]
        records.extend(Record('generator', str(i), 'Influenza', str(i % 5), '', i) for i in range(100, 160))
        records.extend(Record('generator', str(i + 160), 'COVID-19', str(i), '', i) for i in range(5))
        train, val, test, removed = split_disease(records, 'COVID-19', 42)
        self.assertEqual(removed, 5)
        self.assertEqual(len(test), 5)
        self.assertEqual(set(r.group for r in test), set(range(5)))
        self.assertEqual(len(train) + len(val) + removed, 160)
        for partition in (train, val, test):
            self.assertEqual({r.label for r in partition}, set('01234'))
        self.assertFalse({r.group for r in train} & {r.group for r in val})
        self.assertFalse({r.group for r in train + val} & {r.group for r in test})
        self.assertEqual((train, val, test, removed), split_disease(records, 'COVID-19', 42))

    def test_missing_test_disease_is_rejected(self):
        with self.assertRaises(ValueError):
            split_disease([Record('g', '0', 'HPV', 'Otherwise', 'text', 0)], 'COVID-19', 42)


class ParserTests(unittest.TestCase):
    def test_case_punctuation_and_explanations_on_other_lines(self):
        self.assertEqual(clean('  Role: Fear Monger.' + chr(10) + 'Explanation'), 'fear monger')
        self.assertEqual(clean(''), '')

    def test_edit_distance(self):
        self.assertEqual(edit_distance('otherwise', 'otherwise'), 0)
        self.assertEqual(edit_distance('otherwise', 'otherwize'), 1)
        self.assertEqual(edit_distance('fear monger', ''), 11)
        self.assertEqual(edit_distance('fear', 'gear'), edit_distance('gear', 'fear'))


if __name__ == '__main__':
    unittest.main()
