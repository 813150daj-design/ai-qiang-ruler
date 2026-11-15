"""AI腔尺子的命令行入口。"""

import argparse
import json
import sys

from aiqiang import DEFAULT_MODEL_PATH, load_model
from aiqiang.analyzer import analyze_loaded
from aiqiang.report import about_lines, format_report
from aiqiang.textio import decode_text, read_text


class ChineseParser(argparse.ArgumentParser):
    def format_usage(self):
        return super().format_usage().replace('usage:', '用法：', 1)

    def format_help(self):
        return super().format_help().replace('usage:', '用法：', 1)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, '参数有误，请用 --help 查看用法。\n')


def _parser():
    parser = ChineseParser(
        description='AI腔尺子：离线运行，只量文风，不下结论。',
        usage='python ruler.py [选项] [文件.txt ... | -]', add_help=False,
        epilog='示例：python ruler.py -t "一段中文"\n无参数且有管道输入时，读取标准输入。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser._positionals.title = '输入'
    parser._optionals.title = '选项'
    parser.add_argument('files', nargs='*', metavar='文件', help='文字文件；- 表示标准输入')
    parser.add_argument('-t', '--text', metavar='文字', help='直接输入一段文字')
    parser.add_argument('--json', action='store_true', help='以 UTF-8 JSON 输出')
    parser.add_argument('--about', action='store_true', help='显示版本、模型评估与许可')
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH), metavar='路径', help='使用指定模型')
    parser.add_argument('-h', '--help', action='help', help='显示帮助')
    return parser


def _configure_output(force_utf8=False):
    for stream in (sys.stdout, sys.stderr):
        try:
            if force_utf8 or not stream.isatty():
                stream.reconfigure(encoding='utf-8', errors='replace')
            else:
                stream.reconfigure(errors='replace')
        except (AttributeError, OSError, ValueError):
            pass


def _stdin_text():
    if hasattr(sys.stdin, 'buffer'):
        return decode_text(sys.stdin.buffer.read())
    return sys.stdin.read().lstrip('\ufeff')


def main(argv=None):
    _configure_output()
    parser = _parser()
    args = parser.parse_args(argv)
    _configure_output(args.json)
    if not args.about and args.text is None and not args.files and sys.stdin.isatty():
        parser.print_help()
        return 0
    try:
        model = load_model(args.model)
        if args.about:
            print('\n'.join(about_lines(model)))
            return 0
        inputs = []
        if args.text is not None:
            inputs.append(('输入文字', args.text))
        stdin = None
        for filename in args.files:
            if filename == '-':
                if stdin is None:
                    stdin = _stdin_text()
                inputs.append(('标准输入', stdin))
            else:
                inputs.append((filename, read_text(filename)))
        if not inputs:
            inputs.append(('标准输入', _stdin_text()))
        results = [analyze_loaded(text, model) for _, text in inputs]
        if args.json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))
        else:
            blocks = []
            for (name, text), result in zip(inputs, results):
                block = format_report(result, text)
                if len(inputs) > 1:
                    block = '文件：' + name + '\n' + block
                blocks.append(block)
            print('\n\n'.join(blocks))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print('无法完成测量：{}'.format(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
