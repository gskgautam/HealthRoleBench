import unittest

from healthrolebench.common.labels import INVALID, parse


class LabelParserTest(unittest.TestCase):
    def test_label_followed_by_explanation_on_same_line(self):
        self.assertEqual(parse('Fear Monger. The post predominantly uses alarm.'), 'Fear Monger')
        self.assertEqual(parse('Otherwise. The quoted post does not contain misinformation.'), 'Otherwise')

    def test_prefixes_case_spelling_and_markup(self):
        self.assertEqual(parse('Category: anti-vaccine advocate'), 'Anti-Vaccine Advocate')
        self.assertEqual(parse('**Religious Conspiracy Theorist**'), 'Religious Conspiracy Theorist')
        self.assertEqual(parse('Fearmonger'), 'Fear Monger')
        self.assertEqual(parse('Misinformation Spreder'), 'Misinformation Spreader')

    def test_single_label_named_inside_a_sentence(self):
        self.assertEqual(parse('Based on the post, the rhetorical role is Anti-Vaccine Advocate.'), 'Anti-Vaccine Advocate')

    def test_truncated_ambiguous_or_empty_answers_are_invalid(self):
        self.assertEqual(parse('Based on the content of the quoted post, the rhetor'), INVALID)
        self.assertEqual(parse('Not a Fear Monger but an Anti-Vaccine Advocate.'), INVALID)
        self.assertEqual(parse(''), INVALID)

    def test_zero_tolerance_requires_exact_leading_label(self):
        self.assertEqual(parse('Fear Mongr', max_distance=0), INVALID)
        self.assertEqual(parse('Fear Mongr', max_distance=1), 'Fear Monger')


if __name__ == '__main__':
    unittest.main()
