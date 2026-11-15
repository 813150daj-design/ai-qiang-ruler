import random
import unittest

from aiqiang.markers import CHECKS, _UPLIFT, markers
from common import GOLDEN_PATH, ROOT, read_json
from make_toy_model import load_reference


class MarkerTests(unittest.TestCase):
    def test_regexes_match_authoritative_reference_exactly(self):
        ref = load_reference()
        self.assertEqual(len(CHECKS), len(ref.CHECKS))
        self.assertEqual(_UPLIFT.pattern, ref._UPLIFT.pattern)
        self.assertEqual(_UPLIFT.flags, ref._UPLIFT.flags)
        for (name, regex), (ref_name, ref_regex) in zip(CHECKS, ref.CHECKS):
            self.assertEqual(name, ref_name)
            if regex is None:
                self.assertIsNone(ref_regex)
            else:
                self.assertEqual(regex.pattern, ref_regex.pattern)
                self.assertEqual(regex.flags, ref_regex.flags)

    def test_reference_counts_and_matched_spans_on_entire_ai_corpus(self):
        ref = load_reference()
        paths = sorted((ROOT / 'data' / 'ai').glob('*/*.txt'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                text = path.read_text(encoding='utf-8-sig')
                rows = markers(text)
                self.assertEqual({row['name']: row['count'] for row in rows}, ref.markers(text))
                for row, (_, regex) in zip(rows, ref.CHECKS):
                    self.assertEqual(row['count'], len(row['spans']))
                    for start, end in row['spans']:
                        self.assertTrue(0 <= start < end <= len(text))
                        excerpt = text[start:end]
                        self.assertTrue(excerpt.strip())
                        if regex is not None:
                            match = regex.match(text, start)
                            self.assertIsNotNone(match)
                            self.assertEqual(match.end(), end)
                        elif row['name'] == '三连排比':
                            self.assertEqual(ref.parallel(excerpt), 1)
                        else:
                            self.assertEqual(excerpt.replace('\n', ''), ref.sentences(text)[-1].strip())
                            self.assertIsNotNone(ref._UPLIFT.search(excerpt.replace('\n', '')))

    def test_golden_counts_and_spans(self):
        for case in read_json(GOLDEN_PATH)['cases']:
            with self.subTest(name=case['name']):
                text = case['text']
                rows = markers(text)
                self.assertEqual([row['name'] for row in rows], [name for name, _ in CHECKS])
                self.assertEqual({row['name']: row['count'] for row in rows}, case['markers'])
                for row, (_, regex) in zip(rows, CHECKS):
                    self.assertEqual(len(row['spans']), row['count'])
                    for start, end in row['spans']:
                        self.assertTrue(0 <= start < end <= len(text))
                        self.assertTrue(text[start:end].strip())
                        if regex:
                            # Match against the original string to preserve multiline anchors.
                            match = regex.match(text, start)
                            self.assertIsNotNone(match)
                            self.assertEqual(match.end(), end)

    def test_parallel_extends_one_span_and_maps_newlines(self):
        text = '😀。\n  我来， 我\n看，我想，我走，我回。\n  人生很温柔。 \r\n'
        rows = markers(text)
        self.assertEqual(rows[5]['count'], 1)
        start, end = rows[5]['spans'][0]
        self.assertEqual(text[start:end], '我来， 我\n看，我想，我走，我回。')
        start, end = rows[6]['spans'][0]
        self.assertEqual(text[start:end], '人生很温柔。')

    def test_reference_equivalence_on_generated_clause_sequences(self):
        ref = load_reference()
        randomizer = random.Random(19)
        for _ in range(150):
            parts = [randomizer.choice(['他走', '他来', '她来', ' 我看 ', '', '\r\n他跑']) for _ in range(16)]
            text = ''.join(part + randomizer.choice(['，', '、', '；', ',', '。', '\n']) for part in parts)
            self.assertEqual({row['name']: row['count'] for row in markers(text)}, ref.markers(text))


if __name__ == '__main__':
    unittest.main()
