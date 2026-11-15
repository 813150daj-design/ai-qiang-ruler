import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiqiang import analyze, load_model
from aiqiang import analyzer
from aiqiang.features import cjk_count, featurize, predict_doc
from common import MODEL_PATH, ROOT, assert_nested, read_json


class VerdictTests(unittest.TestCase):
    def test_five_tier_boundaries(self):
        limits = dict(ai_high=0.9, mid_high=0.8, mid_low=0.5, human_low=0.2)
        for p, expected in [
            (0.0, 'likely_human'), (0.1999, 'likely_human'), (0.2, 'likely_human'),
            (0.2001, 'leans_human'), (0.4999, 'leans_human'),
            (0.5, 'uncertain'), (0.5001, 'uncertain'), (0.7999, 'uncertain'),
            (0.8, 'leans_ai'), (0.8001, 'leans_ai'), (0.8999, 'leans_ai'),
            (0.9, 'likely_ai'), (0.9001, 'likely_ai'), (1.0, 'likely_ai'),
        ]:
            with self.subTest(p=p):
                self.assertEqual(analyzer.verdict_for(p, limits), expected)

    def test_old_limits_produce_only_three_tiers(self):
        limits = dict(ai_high=0.9, human_low=0.2)
        codes = set()
        for p in [i / 10000 for i in range(10001)]:
            expected = 'likely_ai' if p >= 0.9 else 'likely_human' if p <= 0.2 else 'uncertain'
            code = analyzer.verdict_for(p, limits)
            self.assertEqual(code, expected)
            codes.add(code)
        self.assertEqual(codes, {'likely_ai', 'likely_human', 'uncertain'})
        resolved = analyzer.thresholds({'thresholds': limits})
        self.assertEqual(resolved['mid_high'], 0.9)
        self.assertEqual(resolved['mid_low'], 0.2)
        self.assertNotIn('mid_high', limits)

    def test_equal_thresholds_preserve_decision_order(self):
        limits = dict(ai_high=0.5, mid_high=0.5, mid_low=0.5, human_low=0.5)
        self.assertEqual(analyzer.verdict_for(0.5, limits), 'likely_ai')
        self.assertEqual(analyzer.verdict_for(0.4999, limits), 'likely_human')


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.model = read_json(MODEL_PATH)
        self.text = '我今天去街口买菜，回来发现下雨了。' * 8

    def prediction(self, probability, which=None):
        return dict(
            ai_prob=probability, probs=[0.1, 0.6, 0.2, 0.1], which=which or [0.7, 0.2, 0.1],
            pieces=[[0, len(self.text), probability]],
        )

    def test_verdict_boundaries_inclusive_and_adjacent(self):
        for probability, expected in [
            (0.0, 'likely_human'), (0.2 - 1e-10, 'likely_human'), (0.2, 'likely_human'),
            (0.2 + 1e-10, 'uncertain'), (0.9 - 1e-10, 'uncertain'),
            (0.9, 'likely_ai'), (0.9 + 1e-10, 'likely_ai'), (1.0, 'likely_ai'),
        ]:
            with self.subTest(probability=probability):
                with patch.object(analyzer, 'predict_doc', return_value=self.prediction(probability)):
                    result = analyze(self.text, self.model)
                self.assertEqual(result['verdict'], expected)
                self.assertEqual(result['ai_prob'], probability)

    def test_custom_thresholds_and_min_chars(self):
        self.model['thresholds'].update(min_chars=60, ai_high=0.8, human_low=0.1)
        for size, expected in [(59, 'too_short'), (60, 'likely_ai'), (61, 'likely_ai')]:
            with patch.object(analyzer, 'predict_doc', return_value=self.prediction(0.8)) as predict:
                result = analyze('中' * size, self.model)
            self.assertEqual(result['verdict'], expected)
            self.assertEqual(predict.call_count, int(size >= 60))
        self.assertEqual(analyze('中文', self.model)['verdict_text'], '太短了，没法判断（至少要 60 个汉字）')

    def test_tier_labels_notes_and_human_reference_text(self):
        model = copy.deepcopy(self.model)
        model['thresholds'].update(mid_high=0.8, mid_low=0.5)
        model['tiers'] = {
            code: dict(label='标签 ' + code, note='实测说明 ' + code)
            for code in analyzer.VERDICT_LABELS
        }
        for p in (0.95, 0.85, 0.6, 0.4, 0.1):
            code = analyzer.verdict_for(p, analyzer.thresholds(model))
            with self.subTest(code=code):
                with patch.object(analyzer, 'predict_doc', return_value=self.prediction(p)):
                    result = analyze(self.text, model)
                    old_result = analyze(self.text, self.model)
                self.assertEqual(result['verdict'], code)
                self.assertEqual(result['verdict_text'], model['tiers'][code]['label'])
                self.assertEqual(result['verdict_note'], model['tiers'][code]['note'])
                self.assertEqual(old_result['verdict_note'], '')
                for key in ('ai_prob', 'probs', 'which', 'evidence', 'pieces', 'notes'):
                    self.assertEqual(result[key], old_result[key])
                if code in ('likely_human', 'leans_human'):
                    self.assertEqual(
                        result['which_text'], analyzer.VERDICT_LABELS[code] + '，这一栏仅供参考',
                    )
                model['tiers'][code] = {}
                with patch.object(analyzer, 'predict_doc', return_value=self.prediction(p)):
                    fallback = analyze(self.text, model)
                self.assertEqual(fallback['verdict_text'], analyzer.VERDICT_LABELS[code])
                self.assertEqual(fallback['verdict_note'], '')

    def test_too_short_and_no_model_still_have_all_markers(self):
        for text in ['', '   ', '原来——人生很温柔。']:
            for model in (None, self.model):
                with patch.object(analyzer, 'load_model', return_value=None):
                    result = analyze(text, model)
                self.assertEqual(result['verdict'], 'no_model' if model is None else 'too_short')
                self.assertIsNone(result['ai_prob'])
                self.assertEqual(result['verdict_note'], '')
                self.assertIsNone(result['probs'])
                self.assertEqual(result['which'], [])
                self.assertEqual(result['which_text'], '')
                self.assertEqual(result['pieces'], [])
                self.assertEqual(result['evidence'], {'ai': [], 'human': []})
                self.assertEqual(len(result['markers']), 10)
                self.assertEqual(result['baigu_index'], sum(row['count'] > 0 for row in result['markers']))

    def test_which_text_sorting_and_threshold(self):
        for probability, which, expected in [
            (0.1, [0.8, 0.1, 0.1], '更像人写的，这一栏仅供参考'),
            (0.7, [0.4, 0.3, 0.3], '三家都有点像，分不清'),
            (0.7, [0.25, 0.5, 0.25], '最像 GPT'),
            (0.95, [0.1, 0.2, 0.7], '最像 Claude'),
        ]:
            with patch.object(analyzer, 'predict_doc', return_value=self.prediction(probability, which)):
                result = analyze(self.text, self.model)
            self.assertEqual(result['which_text'], expected)
            self.assertEqual([r['p'] for r in result['which']], sorted(which, reverse=True))

    def test_evidence_is_plain_contribution_deduplicated_and_cjk(self):
        positive = ['原来如此', '原来', '如此', '此刻', '宇宙', '星辰', '成长', '人生', '温柔', '意义']
        negative = ['哈哈大笑', '哈哈', '大笑', '饭糊', '锅没洗', '忘带伞', '菜凉', '面坨', '鞋湿']
        vocab = {}
        for i, gram in enumerate(positive):
            vocab[gram] = [1.0, 0, 10 - i * 0.4, -5, 0]
        for i, gram in enumerate(negative):
            vocab[gram] = [1.0, 0, -9 + i * 0.4, 5, 0]
        unreadable = ['原', 'aa', '。！', ',而是', '。它', 'a中', '中0', '中😀', '中𠀀']
        vocab.update({gram: [1, 0, 100, 0, 0] for gram in unreadable})
        plain = dict(name='plain', source='plain', n=[1, 2, 3, 4], vocab=vocab)
        distorted = dict(name='skeleton', source='distorted', n=[2], vocab={'xx': [1, 0, 9999, 0, 0]})
        self.model['feature_sets'] = [plain, distorted]
        text = '，'.join(positive + negative) + '，A B。！,而是。它a中0中😀中𠀀' + '中' * 50
        with patch.object(analyzer, 'predict_doc', return_value=self.prediction(0.8)):
            result = analyze(text, self.model)
        values = featurize(text, plain)
        for key, sign in [('ai', 1), ('human', -1)]:
            chosen = result['evidence'][key]
            self.assertEqual(len(chosen), 6)
            self.assertEqual([abs(value) for _, value in chosen], sorted([abs(v) for _, v in chosen], reverse=True))
            for index, (gram, value) in enumerate(chosen):
                self.assertGreaterEqual(len(gram), 2)
                self.assertEqual(cjk_count(gram), len(gram))
                self.assertGreater(value * sign, 0)
                self.assertEqual(value, round(values[gram] * vocab[gram][2], 4))
                self.assertFalse(any(gram in old or old in gram for old, _ in chosen[:index]))

    def test_evidence_includes_basic_and_extension_a_boundaries(self):
        grams = ['㐀䶿', '一鿿']
        self.model['feature_sets'] = [dict(
            source='plain', n=[2], vocab={gram: [1, 0, 1, 0, 0] for gram in grams},
        )]
        with patch.object(analyzer, 'predict_doc', return_value=self.prediction(0.8)):
            result = analyze('，'.join(grams) + '中' * 50, self.model)
        self.assertEqual({gram for gram, _ in result['evidence']['ai']}, set(grams))

    def test_notes_order_and_ratio(self):
        with patch.object(analyzer, 'load_model', return_value=None):
            result = analyze('中' * 3001 + 'a ' * 8000)
            self.assertEqual(result['notes'][:2], analyzer.BASE_NOTES)
            self.assertEqual(len(result['notes']), 3)
            self.assertIn('汉字不多', result['notes'][2])
            self.assertEqual(len(analyze('中' * 3000)['notes']), 2)
            self.assertEqual(len(analyze('中文中abcdefg')['notes']), 2)
            self.assertEqual(len(analyze('中文abcdefg')['notes']), 3)
            self.assertEqual(len(analyze('  \n\t')['notes']), 2)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        text = ('中' * 40 + 'a' * 259 + '。') * 3
        result = analyze(text, self.model)
        self.assertEqual(len(result['notes']), 4)
        self.assertEqual(result['notes'][:2], analyzer.BASE_NOTES)
        self.assertEqual(result['notes'][2], '文字较长，分成 3 段分别量，再取平均。')
        self.assertIn('汉字不多', result['notes'][3])

    def test_document_pieces_and_whole_text_evidence(self):
        unit = '中' * 299 + '。'
        for text in (self.text, self.text * 10, '中' * 1000, 'a' * 299 + '。' + unit):
            with self.subTest(text=text[:20]):
                doc = predict_doc(self.model, text)
                with patch.object(analyzer, '_evidence', wraps=analyzer._evidence) as evidence:
                    result = analyze(text, self.model)
                self.assertEqual(result['pieces'], [dict(start=a, end=b, ai_prob=p) for a, b, p in doc['pieces']])
                self.assertEqual(result['ai_prob'], doc['ai_prob'])
                assert_nested(self, result['probs'], dict(zip(self.model['classes'], doc['probs'])))
                expected = [dict(name=n, p=p) for n, p in zip(self.model['classes'][1:], doc['which'])]
                self.assertEqual(result['which'], sorted(expected, key=lambda row: row['p'], reverse=True))
                top_class = self.model['classes'].index(result['which'][0]['name'])
                evidence.assert_called_once_with(text, self.model, top_class)
                extra = []
                if len(result['pieces']) > 1:
                    extra.append('文字较长，分成 {} 段分别量，再取平均。'.format(len(result['pieces'])))
                if result['chars'] < sum(not ch.isspace() for ch in text) * 0.3:
                    extra.append('这段文字里汉字不多，结果不可靠。')
                self.assertEqual(result['notes'], analyzer.BASE_NOTES + extra)
                json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_model_cache_missing_then_created_and_dynamic_classes(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            path = Path(directory) / '模型.json'
            self.assertIsNone(load_model(path))
            for count in (2, 5):
                model = copy.deepcopy(self.model)
                model['classes'] = ['人'] + ['模型{}'.format(i) for i in range(1, count)]
                model['bias'] = [0.0] * count
                for fs in model['feature_sets']:
                    for gram, weights in fs['vocab'].items():
                        fs['vocab'][gram] = [weights[0]] + [0.1] * count
                current = path if count == 2 else path.with_name('模型五类.json')
                current.write_bytes(json.dumps(model, ensure_ascii=False).encode('utf-8'))
                first = load_model(current)
                self.assertIs(load_model(current), first)
                result = analyze(self.text, first)
                self.assertEqual(len(result['which']), count - 1)
                self.assertEqual(len(result['probs']), count)


if __name__ == '__main__':
    unittest.main()
