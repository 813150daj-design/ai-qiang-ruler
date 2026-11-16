"""The reference's ten checks, with Python character offsets in the original text."""

import re

_UPLIFT = re.compile(
    r'人生|生活|日子|岁月|时光|幸福|温柔|治愈|意义|勇气|成长|慈悲|美好|滋味|人间|福气|骄傲'
    r'|值得|珍惜'
)
CHECKS = [
    ('不是…而是', re.compile(
        r'不是[^。！？!?\n]{1,30}?而是|不是[^。！？!?\n，,]{1,25}[，,]\s*(而)?是'
        r'|与其说[^。！？!?\n]{1,30}不如说'
        r'|与其[^说。！？!?\n][^。！？!?\n]{0,20}?不如'
        r'|并非[^。！？!?\n]{1,30}?而是|不在于[^。！？!?\n]{1,30}?而在于')),
    ('不仅…更', re.compile(r'不(仅|只|单)(仅|是)?[^。！？!?\n]{1,30}?(更|还|也)')),
    ('原来/忽然', re.compile(
        r'原来|才(发现|明白|懂得|知道)|终于(明白|懂了)'
        r'|(忽然|突然|蓦然|猛然|顿时|这才)(就)?'
        r'(觉得|感到|感觉到?|发现|明白|懂得|懂了|意识到|想通)')),
    ('这一刻', re.compile(r'这一刻|那一刻|此刻|瞬间|那一瞬|这一瞬|刹那')),
    ('破折号', re.compile(r'——|—')),
    ('三连排比', None),
    ('结尾升华', None),
    ('首先/总之', re.compile(
        r'首先|其次|总之|总而言之|综上所述|值得注意的是|总的来说|换句话说'
        r'|某种意义上|不得不说|说到底|说白了|归根结底|归根到底|简而言之')),
    ('大词抒情', re.compile(
        r'量子|熵|宇宙|星河|星辰|永恒|灵魂|褶皱|缝隙|共振|维度|意义感'
        r'|烟火气|诗意|底色|和解|救赎|温柔')),
    # \d uses live Unicode data for style checks only, never model features.
    ('客服/格式', re.compile(
        r'希望(这些|以上|对你)|如果你(愿意|需要)|需要的话|好问题|很高兴|让我们|以下是'
        r'|^\s*(#{1,4} |[-*] |\d+[.、] )|\*\*', re.M)),
]
HINTS = [
    '先否定、再肯定的转折句（不是…而是、并非…而是、与其…不如）',
    '层层递进的句式',
    '“原来”“才发现”“突然感到”式的顿悟',
    '这一刻、那一刻、此刻、瞬间、刹那',
    '破折号',
    '同一个字开头的三连分句',
    '最后一句拔高到人生、意义、温柔、珍惜……',
    '首先、其次、总之、说白了一类的提纲词',
    '宇宙、灵魂、烟火气、温柔一类的大词',
    '“希望对你有帮助”、列表、加粗这类客服腔和格式',
]


def sentences(text):
    return [s for s in re.split(r'(?<=[。！？!?])', text.replace('\n', '')) if s.strip()]


def _sentence_ranges(flat):
    start = 0
    for match in re.finditer(r'(?<=[。！？!?])', flat):
        end = match.start()
        if flat[start:end].strip():
            yield start, end
        start = end
    if flat[start:].strip():
        yield start, len(flat)


def _trim_range(flat, start, end):
    while start < end and flat[start].isspace():
        start += 1
    while end > start and flat[end - 1].isspace():
        end -= 1
    return start, end


def _parallel_spans(flat, positions):
    spans = []
    for start, end in _sentence_ranges(flat):
        parts = []
        cursor = start
        for delimiter in re.finditer(r'[，、；,]', flat[start:end]):
            boundary = start + delimiter.start()
            a, b = _trim_range(flat, cursor, boundary)
            if a < b:
                parts.append((a, b))
            cursor = boundary + 1
        a, b = _trim_range(flat, cursor, end)
        if a < b:
            parts.append((a, b))
        run = 1
        for i in range(1, len(parts)):
            same = flat[parts[i - 1][0]] == flat[parts[i][0]]
            run = run + 1 if same else 1
            if run == 3:
                spans.append([positions[parts[i - 2][0]], positions[parts[i][1] - 1] + 1])
            elif run > 3:
                spans[-1][1] = positions[parts[i][1] - 1] + 1
    return spans


def markers(text):
    """Return all ten rows; spans are half-open offsets before any normalisation."""
    positions = [i for i, ch in enumerate(text) if ch != '\n']
    flat = text.replace('\n', '')
    rows = []
    for (name, regex), hint in zip(CHECKS, HINTS):
        if name == '三连排比':
            spans = _parallel_spans(flat, positions)
        elif name == '结尾升华':
            ranges = list(_sentence_ranges(flat))
            spans = []
            if ranges:
                start, end = ranges[-1]
                if _UPLIFT.search(flat[start:end]):
                    start, end = _trim_range(flat, start, end)
                    spans = [[positions[start], positions[end - 1] + 1]]
        else:
            spans = [[m.start(), m.end()] for m in regex.finditer(text)]
        rows.append(dict(name=name, count=len(spans), spans=spans, hint=hint))
    return rows


def parallel(text):
    return markers(text)[5]['count']


def baigu_index(text):
    return sum(row['count'] > 0 for row in markers(text))
