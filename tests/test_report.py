import unittest

from aiqiang.analyzer import analyze_loaded
from aiqiang.report import format_probability, format_report


class ReportTests(unittest.TestCase):
    def test_probability_display_boundaries(self):
        for probability, expected in [
            (0.0, '1% 以下'), (0.004, '1% 以下'), (0.01, '1% 以下'),
            (0.0149, '1%'), (0.5, '50%'), (0.985, '98%'),
            (0.99, '99%+'), (0.999, '99%+'), (1.0, '99%+'),
            (0.9899, '99%'),
        ]:
            with self.subTest(probability=probability):
                self.assertEqual(format_probability(probability), expected)

    def test_report_formats_probabilities_without_mutating_result(self):
        result = analyze_loaded('', None)
        result.update(
            ai_prob=1.0, verdict='likely_ai', verdict_text='很像 AI 写的',
            which=[{'name': 'DeepSeek', 'p': 0.999}, {'name': 'GPT', 'p': 0.001}],
            which_text='最像 DeepSeek',
        )
        report = format_report(result, '')
        self.assertIn('AI 可能性 99%+', report)
        self.assertIn('DeepSeek 99%+ ｜ GPT 1% 以下', report)
        self.assertEqual(result['ai_prob'], 1.0)
        self.assertEqual([row['p'] for row in result['which']], [0.999, 0.001])
        result['ai_prob'] = 0.0
        self.assertIn('AI 可能性 1% 以下', format_report(result, ''))

    def test_piece_percentage_helper_and_line_placement(self):
        result = analyze_loaded('', None)
        result['pieces'] = [dict(start=i, end=i + 1, ai_prob=p) for i, p in enumerate((0.45, 0.8, 1.0, 0.0))]
        lines = format_report(result, '').splitlines()
        self.assertTrue(lines[2].startswith('结论：'))
        self.assertEqual(lines[3], '分段：第1段 45% ｜ 第2段 80% ｜ 第3段 99%+ ｜ 第4段 1% 以下')
        self.assertEqual([p['ai_prob'] for p in result['pieces']], [0.45, 0.8, 1.0, 0.0])


if __name__ == '__main__':
    unittest.main()
