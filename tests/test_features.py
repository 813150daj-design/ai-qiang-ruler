import copy
import unicodedata
import unittest

from aiqiang import features
from common import GOLDEN_PATH, MODEL_PATH, ROOT, assert_nested, read_json
from make_toy_model import load_reference


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = read_json(MODEL_PATH)
        cls.cases = read_json(GOLDEN_PATH)['cases']
        cls.ref = load_reference()

    def test_golden(self):
        self.assertGreaterEqual(len(self.cases), 30)
        for case in self.cases:
            with self.subTest(name=case['name']):
                text = case['text']
                self.assertEqual(features.normalize(text), case['normalized'])
                self.assertEqual(features.distort(features.normalize(text)), case['distorted'])
                self.assertEqual(features.cjk_count(text), case['chars'])
                for i, fs in enumerate(self.model['feature_sets']):
                    assert_nested(self, features.featurize(text, fs), case['features'][i])
                prediction = features.predict(self.model, text)
                for key in ('logits', 'probs', 'ai_score', 'ai_prob', 'which'):
                    assert_nested(self, prediction[key], case[key])
                assert_nested(self, features.logits(self.model, text), case['logits'])
                assert_nested(self, features.softmax(case['logits']), case['probs'])
                assert_nested(self, features.split_pieces(text, self.model.get('long_text')), case['pieces'])
                assert_nested(self, features.predict_doc(self.model, text), case['doc'])

    @unittest.skipUnless(
        unicodedata.unidata_version == '12.1.0',
        'Full table validation requires Python 3.8 with Unicode 12.1.0',
    )
    def test_frozen_unicode_table_matches_every_code_point(self):
        table = read_json(ROOT / 'aiqiang' / 'unicode_12_1.json')
        self.assertEqual(table['unicode'], '12.1.0')
        for key, live in [
            ('assigned', lambda ch: unicodedata.category(ch) != 'Cn'),
            ('alnum', str.isalnum),
        ]:
            included = bytearray(0x110000)
            for start, end in table[key]:
                included[start:end + 1] = b'\x01' * (end - start + 1)
            for cp in range(0x110000):
                if bool(included[cp]) != live(chr(cp)):
                    self.fail('{} differs from Unicode 12.1 at U+{:04X}'.format(key, cp))

    def test_unicode_12_1_literal_results(self):
        # Literal normalize/distort results from train/ref_features.py on Python 3.8.
        cases = [
            ('x䶿一鿿y', 'a䶿一鿿a', 'x䶿一鿿x'),
            ('@鿿用户 你好', '你好', '你x'),
            ('编号 \U0001fbf0\U0001fbf1 12', '编号\U0001fbf0\U0001fbf10',
             'x\U0001fbf0\U0001fbf1x'),
            ('ΣΑΣ ΟΔΟΣ', 'σαςοδος', 'x'),
            ('@\U0001fbf0\U0001fbf1 你好', '@\U0001fbf0\U0001fbf1你好',
             '@\U0001fbf0\U0001fbf1你x'),
            ('Σ\U0001e290Σ', 'σ\U0001e290σ', 'x\U0001e290x'),
            ('\U00010570\U0001e290\U0001fbf0䶿鿿',
             '\U00010570\U0001e290\U0001fbf0䶿鿿',
             '\U00010570\U0001e290\U0001fbf0䶿鿿'),
            ('Å䶿\u0301', 'å䶿\u0301', 'x䶿\u0301'),
        ]
        for text, normalized, distorted in cases:
            with self.subTest(text=text):
                self.assertEqual(features.normalize(text), normalized)
                self.assertEqual(features.distort(normalized), distorted)
                self.assertEqual(features.source_string(text, {'source': 'plain'}), normalized)
                self.assertEqual(features.source_string(text, {'source': 'distorted'}), distorted)

    def test_split_boundaries_and_short_remainder(self):
        for size in (0, 499, 500):
            text = '中。' * (size // 2) + '中' * (size % 2)
            self.assertEqual(features.split_pieces(text), [(0, size)])
        self.assertEqual(features.split_pieces('中。' * 251), [(0, 300), (300, 502)])
        self.assertEqual(features.split_pieces('中' * 1000), [(0, 1000)])
        text = ('春' * 149 + '。') * 4 + '尾' * 40
        self.assertEqual(features.split_pieces(text), [(0, 300), (300, 640)])
        unit = '中' * 299 + '。'
        text = ' \r\n' + unit + '\r\n\r\n' + unit + '\r\n '
        self.assertEqual(features.split_pieces(text), [(3, 303), (307, 607)])

    def test_long_text_options_match_reference_and_fall_back_per_key(self):
        texts = [case['text'] for case in self.cases if case['name'].startswith('long')]
        params = [
            None, {}, {'min_len': 1000}, {'target': 200}, {'min_piece': 10}, {'min_cjk': 500},
            {'min_len': 100, 'target': 80, 'min_piece': 25, 'min_cjk': 5},
        ]
        for options in params:
            model = dict(self.model, long_text=options)
            before = copy.deepcopy(model)
            for text in texts:
                with self.subTest(options=options, text=text[:20]):
                    self.assertEqual(features.split_pieces(text, options), self.ref.split_pieces(text, options))
                    assert_nested(self, features.predict_doc(model, text), self.ref.predict_doc(model, text))
            self.assertEqual(model, before)

    def test_document_averages_scores_not_probabilities_or_piece_lengths(self):
        model = dict(
            classes=['人', 'AI'], bias=[0.0, 0.0], ai_calibration={'a': 1.7, 'b': -0.4},
            feature_sets=[dict(source='plain', n=[1], vocab={'甲': [1, 0, 4], '乙': [1, 0, -1]})],
        )
        text = '甲' * 299 + '。' + '乙' * 200 + '。'
        doc = features.predict_doc(model, text)
        self.assertEqual(doc['ai_score'], 1.5)
        self.assertEqual(doc['logits'], [0.0, 1.5])
        for key in ('probs', 'which'):
            expected = [(a + b) / 2 for a, b in zip(features.predict(model, text[:300])[key],
                                                   features.predict(model, text[300:])[key])]
            assert_nested(self, doc[key], expected)
        self.assertEqual(doc['ai_prob'], features.sigmoid(1.7 * 1.5 - 0.4))
        self.assertGreater(abs(doc['ai_prob'] - sum(p[2] for p in doc['pieces']) / 2), 0.1)

    def test_document_filters_low_cjk_pieces_and_falls_back_if_all_skipped(self):
        unit = '中' * 299 + '。'
        for text, expected in [
            ('a' * 299 + '。' + unit, [(300, 600)]),
            ('a' * 299 + '。' + unit + 'b' * 299 + '。' + unit, [(300, 600), (900, 1200)]),
            (('a' * 299 + '。') * 3, [(0, 900)]),
        ]:
            with self.subTest(text=text[:20]):
                doc = features.predict_doc(self.model, text)
                self.assertEqual([(a, b) for a, b, _ in doc['pieces']], expected)
                assert_nested(self, doc, self.ref.predict_doc(self.model, text))

    def test_custom_distortion_and_missing_defaults(self):
        texts = ['ABC猫咪你好——我在这里', '我爱猫猫猫，猫爱我。', 'éé🌱你123', '']
        variants = [
            {'source': 'plain'}, {'source': 'distorted'},
            {'source': 'distorted', 'keep': '我你', 'punct': '，', 'mask': '?'},
            {'source': 'distorted', 'keep': '', 'punct': '', 'mask': 'xx'},
        ]
        for text in texts:
            for fs in variants:
                with self.subTest(text=text, fs=fs):
                    self.assertEqual(features.source_string(text, fs), self.ref.source_string(text, fs))

    def test_markdown_rule_line_normalization(self):
        for mark in '-*_':
            for rule in (mark * 3, mark * 5, ' \t' + ' \t'.join([mark] * 3) + '\t '):
                with self.subTest(rule=rule):
                    text = '前文\n' + rule + '\n后文'
                    self.assertEqual(features.normalize(text), '前文后文')
                    self.assertEqual(features.normalize(text), self.ref.normalize(text))
                    self.assertEqual(features.normalize(rule), '')
        for text, expected in [
            ('前文\n－－－\n＊＊＊\n＿＿＿\n后文', '前文后文'),
            ('---\n# 标题\n* * *\n- 正文\n___\n**结尾**', '标题正文结尾'),
            ('正文---结尾', '正文—结尾'),
            ('--\n正文', '—正文'),
            ('-*-\n正文', '-*-正文'),
            ('---文字\n正文', '—文字正文'),
        ]:
            with self.subTest(text=text):
                self.assertEqual(features.normalize(text), expected)
                self.assertEqual(features.normalize(text), self.ref.normalize(text))

    def test_ngram_count_duplicate_ns_and_short_inputs(self):
        for text in ('', 'a', '猫咪猫咪', '😀猫😀'):
            for ns in ([1, 2, 3], [2, 2], [4]):
                self.assertEqual(features.ngram_counts(text, ns), self.ref.ngram_counts(text, ns))

    def test_unknown_source_and_empty_vocabulary(self):
        with self.assertRaises(ValueError):
            features.source_string('中文', {'source': 'unknown'})
        self.assertEqual(features.featurize('中文', {'source': 'plain', 'n': [1], 'vocab': {}}), {})

    def test_extreme_calibrated_scores_do_not_overflow(self):
        for bias, expected in [([1.0, 0.0], 0.0), ([0.0, 1.0], 1.0)]:
            model = dict(
                classes=['人', 'AI'], bias=bias, feature_sets=[],
                ai_calibration={'a': 1000.0, 'b': 0.0},
            )
            with self.subTest(bias=bias):
                self.assertEqual(features.predict(model, '中文')['ai_prob'], expected)
                self.assertEqual(features.predict(model, '中文'), self.ref.predict(model, '中文'))


if __name__ == '__main__':
    unittest.main()
