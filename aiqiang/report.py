"""Chinese plain-text reports, without assertions about authorship."""

from . import __version__
from .analyzer import MISSING_MODEL_MESSAGE

AFFILIATION = '非 Anthropic 官方，与 DeepSeek、OpenAI、Anthropic 官方均无关。'


def format_probability(probability):
    """Format an estimated probability without displaying absolute certainty."""
    if probability >= 0.99:
        return '99%+'
    if probability <= 0.01:
        return '1% 以下'
    return '{}%'.format(max(1, min(99, round(probability * 100))))


def about_lines(model):
    lines = ['AI腔尺子 ' + __version__, '离线运行，只量文风，不下结论。']
    if model is None:
        lines.append(MISSING_MODEL_MESSAGE)
    else:
        lines.extend(model.get('eval', {}).get('summary_lines', []))
    lines.extend([AFFILIATION, '代码 MIT；模型权重 CC BY-SA 4.0（训练数据含维基百科）'])
    return lines


def format_report(result, text):
    lines = [
        'AI腔尺子 {}（离线运行，只量文风，不下结论）'.format(result['version']),
        '字数：{} 个汉字'.format(result['chars']),
    ]
    verdict = '结论：' + result['verdict_text']
    if result['ai_prob'] is not None:
        verdict += '（AI 可能性 {}）'.format(format_probability(result['ai_prob']))
    lines.append(verdict)
    pieces = result['pieces']
    if len(pieces) > 1:
        scores = ' ｜ '.join(
            '第{}段 {}'.format(i, format_probability(piece['ai_prob'])) for i, piece in enumerate(pieces, 1)
        )
        lines.append('分段：' + scores)
    if result['verdict'] == 'no_model':
        lines.append(MISSING_MODEL_MESSAGE)
    if result['which']:
        scores = ' ｜ '.join('{} {}'.format(row['name'], format_probability(row['p'])) for row in result['which'])
        lines.append('更像哪个 AI：{} —— {}'.format(result['which_text'], scores))
    lines.append('AI 八股：{}/10'.format(result['baigu_index']))
    for row in result['markers']:
        if row['count']:
            start, end = row['spans'][0]
            excerpt = text[start:end].replace('\r', ' ').replace('\n', ' ')
            if len(excerpt) > 24:
                excerpt = excerpt[:23] + '…'
            lines.append('  ● {} ×{}  {}  「{}」'.format(
                row['name'], row['count'], row['hint'], excerpt,
            ))
        else:
            lines.append('  ○ {} ×0'.format(row['name']))
    for key, label in [('ai', '像 AI 的写法'), ('human', '像人的写法')]:
        grams = ' · '.join(gram for gram, _ in result['evidence'][key])
        lines.append(label + '：' + (grams or '暂无'))
    lines.extend('提示：' + note for note in result['notes'])
    return '\n'.join(lines)
