import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ruler
from aiqiang.report import format_probability
from common import MODEL_PATH, ROOT, read_json


class CLITests(unittest.TestCase):
    def run_cli(self, arguments, data=None, model_path=MODEL_PATH, module=False):
        entry = ['-m', 'aiqiang'] if module else [str(ROOT / 'ruler.py')]
        command = [sys.executable] + entry + ['--model', str(model_path)] + arguments
        environment = dict(os.environ, PYTHONIOENCODING='utf-8')
        completed = subprocess.run(
            command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(ROOT), env=environment, timeout=20,
        )
        return completed.returncode, completed.stdout.decode('utf-8'), completed.stderr.decode('utf-8')

    def test_files_utf8_bom_and_gbk_and_multiple_results(self):
        text = '首先，不是远方而是此刻。原来日子这么温柔。' * 5
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            paths = []
            for encoding in ('utf-8', 'utf-8-sig', 'gbk'):
                path = Path(directory) / ('中文路径_' + encoding + '.txt')
                path.write_bytes(text.encode(encoding))
                paths.append(str(path))
                code, out, error = self.run_cli([str(path)])
                self.assertEqual(code, 0, error)
                self.assertIn('结论', out)
                self.assertIn('不是…而是 ×5', out)
            code, out, error = self.run_cli(['--json'] + paths)
            self.assertEqual(code, 0, error)
            results = json.loads(out)
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[1], results[2])

    def test_text_json_stdin_and_module_entry(self):
        text = '他让脚步变轻，他让心跳放慢，他让烦恼远去。' * 5
        for arguments, data, module in [
            (['-t', text], None, False),
            (['-'], text.encode('utf-8'), False),
            ([], text.encode('gbk'), False),
            (['-t', text], None, True),
        ]:
            code, out, error = self.run_cli(arguments, data, module=module)
            self.assertEqual(code, 0, error)
            self.assertIn('结论', out)
        code, out, error = self.run_cli(['--json', '-t', text])
        self.assertEqual(code, 0, error)
        result = json.loads(out)
        self.assertIn('verdict', result)
        self.assertIn('不是…而是', out)
        self.assertNotIn('\\u4e0d', out)

    def test_about_summary_verbatim(self):
        code, out, error = self.run_cli(['--about'])
        self.assertEqual(code, 0, error)
        for line in read_json(MODEL_PATH)['eval']['summary_lines']:
            self.assertIn(line, out)
        self.assertIn('0.1.0', out)
        self.assertIn('代码 MIT；模型权重 CC BY-SA 4.0（训练数据含维基百科）', out)

    def test_piece_line_after_verdict_and_json_offsets(self):
        text = ' \r\n' + ('原来，今天路边的花开了。' * 30 + '\r\n\r\n') * 3 + ' \r\n'
        code, out, error = self.run_cli(['--json', '-t', text])
        self.assertEqual(code, 0, error)
        result = json.loads(out)
        self.assertGreater(len(result['pieces']), 1)
        expected = '分段：' + ' ｜ '.join(
            '第{}段 {}'.format(i, format_probability(p['ai_prob'])) for i, p in enumerate(result['pieces'], 1)
        )
        code, out, error = self.run_cli(['-t', text])
        self.assertEqual(code, 0, error)
        lines = out.splitlines()
        verdict_index = next(i for i, line in enumerate(lines) if line.startswith('结论：'))
        self.assertEqual(lines[verdict_index + 1], expected)
        for piece in result['pieces']:
            self.assertEqual(text[piece['start']:piece['end']].strip(), text[piece['start']:piece['end']])
        for text in ('中。' * 30, '中' * 1000):
            code, out, error = self.run_cli(['-t', text])
            self.assertEqual(code, 0, error)
            self.assertNotIn('分段：', out)

    def test_probability_limits_and_json_raw_values(self):
        model = read_json(MODEL_PATH)
        model.update(feature_sets=[], bias=[0.0, 1000.0, -1000.0, -1000.0])
        text = '中文' * 30
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            path = Path(directory) / 'extreme_model.json'
            for calibration, probability, display in [(1000.0, 1.0, '99%+'), (-1000.0, 0.0, '1% 以下')]:
                with self.subTest(probability=probability):
                    model['ai_calibration'] = {'a': 0.0, 'b': calibration}
                    path.write_bytes(json.dumps(model, ensure_ascii=False).encode('utf-8'))
                    code, out, error = self.run_cli(['-t', text], model_path=path)
                    self.assertEqual(code, 0, error)
                    self.assertIn('AI 可能性 ' + display, out)
                    self.assertIn('DeepSeek 99%+ ｜ GPT 1% 以下 ｜ Claude 1% 以下', out)
                    code, out, error = self.run_cli(['--json', '-t', text], model_path=path)
                    self.assertEqual(code, 0, error)
                    result = json.loads(out)
                    self.assertEqual(result['ai_prob'], probability)
                    self.assertEqual([row['p'] for row in result['which']], [1.0, 0.0, 0.0])
                    self.assertEqual(result['probs'], {'人': 0.0, 'DeepSeek': 1.0, 'GPT': 0.0, 'Claude': 0.0})

    def test_missing_model_and_unreadable_input(self):
        missing = ROOT / 'tests' / 'fixtures' / 'does_not_exist.json'
        code, out, error = self.run_cli(['-t', '原来——人生很温柔。'], model_path=missing)
        self.assertEqual(code, 0, error)
        self.assertIn('还没有模型文件 aiqiang/model.json', out)
        self.assertIn('原来/忽然 ×1', out)
        self.assertNotIn('Traceback', out + error)
        code, out, error = self.run_cli(['-'], b'\x81')
        self.assertEqual(code, 1)
        self.assertIn('无法', error)
        self.assertNotIn('Traceback', error)

    def test_no_arguments_on_terminal_prints_usage(self):
        with patch.object(sys.stdin, 'isatty', return_value=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(ruler.main([]), 0)
        self.assertIn('用法', output.getvalue())


if __name__ == '__main__':
    unittest.main()
