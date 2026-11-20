import unittest

import numpy as np
from sklearn.metrics import f1_score

from healthrolebench.analysis.significance import ci, holm, macro_f1, majority, paired


class SignificanceTest(unittest.TestCase):
    def test_macro_f1_matches_sklearn_and_treats_invalid_as_wrong(self):
        rng = np.random.default_rng(0)
        y, p = rng.integers(0, 5, 400), rng.integers(-1, 5, 400)
        expected = f1_score(y, p, labels=range(5), average='macro', zero_division=0)
        self.assertAlmostEqual(float(macro_f1(y, p, 5)), expected, places=12)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_paired_detects_a_clear_difference_and_ci_brackets_estimate(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 2000)
        good = np.where(rng.random(2000) < .9, y, 1 - y)
        bad = np.where(rng.random(2000) < .6, y, 1 - y)
        diff, low, high, p = paired(y, good, bad, 2)
        self.assertTrue(low > 0 and p <= .002 and low <= diff <= high)
        value, lo, hi = ci(y, good, 2)
        self.assertTrue(lo <= value <= hi and hi - lo < .05)
        self.assertGreater(paired(y, good, good, 2)[3], .5)

    def test_majority_vote_breaks_ties_by_first_seed(self):
        self.assertEqual(majority({'42': 'A', '43': 'B', '44': 'B'}, ('42', '43', '44')), 'B')
        self.assertEqual(majority({'42': 'A', '43': 'B', '44': 'C'}, ('42', '43', '44')), 'A')


if __name__ == '__main__':
    unittest.main()
