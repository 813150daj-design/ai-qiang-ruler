"""Model loading, cautious verdicts and explanations."""

import json
from functools import lru_cache
from pathlib import Path

from . import __version__
from .features import cjk_count, featurize, predict_doc
from .markers import markers

DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name('model.json')
DEFAULT_THRESHOLDS = dict(min_chars=50, ai_high=0.9, human_low=0.2, which_min=0.5)
MISSING_MODEL_MESSAGE = '还没有模型文件 aiqiang/model.json'
BASE_NOTES = [
    '这把尺子只量文风，不能证明是谁写的。人也会写出“AI 腔”，AI 也能写得不像 AI。',
    '请别拿它的结果去指控任何人，尤其是学生的作文。',
]


@lru_cache(maxsize=4)
def _read_model(path):
    with open(path, 'r', encoding='utf-8') as handle:
        model = json.load(handle)
    if model.get('format') != 'aiqiang-model' or model.get('version') != 1:
        raise ValueError('模型格式不支持，需要 aiqiang-model 第 1 版。')
    classes = model['classes']
    if len(classes) < 2 or classes[0] != '人' or len(set(classes)) != len(classes):
        raise ValueError('模型必须包含“人”和至少一个 AI 类别，且类别名称不能重复。')
    if len(model['bias']) != len(classes):
        raise ValueError('模型的偏置数量与类别数量不一致。')
    for fs in model['feature_sets']:
        if fs['source'] not in ('plain', 'distorted'):
            raise ValueError('模型含有不支持的特征来源。')
        if not fs['n'] or any(not isinstance(n, int) or n < 1 for n in fs['n']):
            raise ValueError('模型的 n-gram 长度必须是正整数。')
        if any(len(weights) != len(classes) + 1 for weights in fs['vocab'].values()):
            raise ValueError('模型的特征权重数量与类别数量不一致。')
    model['ai_calibration']['a']
    model['ai_calibration']['b']
    return model


def load_model(path=DEFAULT_MODEL_PATH):
    """Load UTF-8 JSON once per resolved path. Missing files return None, uncached."""
    try:
        return _read_model(str(Path(path).resolve()))
    except FileNotFoundError:
        return None


def thresholds(model):
    values = dict(DEFAULT_THRESHOLDS)
    if model:
        values.update(model.get('thresholds', {}))
    return values


def _evidence(text, model, top_class):
    contributions = {}
    for fs in model['feature_sets']:
        if fs['source'] != 'plain':
            continue
        for gram, value in featurize(text, fs).items():
            if len(gram) < 2 or cjk_count(gram) != len(gram):
                continue
            weights = fs['vocab'][gram]
            delta = value * (weights[1 + top_class] - weights[1])
            contributions[gram] = contributions.get(gram, 0.0) + delta
    result = {'ai': [], 'human': []}
    ranked = sorted(contributions.items(), key=lambda item: (-abs(item[1]), item[0]))
    for gram, value in ranked:
        if value == 0:
            continue
        chosen = result['ai' if value > 0 else 'human']
        if len(chosen) == 6 or any(gram in old or old in gram for old, _ in chosen):
            continue
        chosen.append([gram, round(value, 4)])
    return result


def analyze_loaded(text, model):
    """Analyze an explicitly loaded model; None means missing, without a second load."""
    chars = cjk_count(text)
    rows = markers(text)
    notes = list(BASE_NOTES)
    nonspace = sum(not ch.isspace() for ch in text)
    if chars < nonspace * 0.3:
        notes.append('这段文字里汉字不多，结果不可靠。')
    result = dict(
        version=__version__, chars=chars, verdict='no_model', verdict_text='还没有模型文件',
        ai_prob=None, probs=None, which=[], which_text='', pieces=[], markers=rows,
        baigu_index=sum(row['count'] > 0 for row in rows),
        evidence={'ai': [], 'human': []}, notes=notes,
    )
    if model is None:
        return result
    limits = thresholds(model)
    if chars < limits['min_chars']:
        result.update(
            verdict='too_short',
            verdict_text='太短了，没法判断（至少要 {} 个汉字）'.format(limits['min_chars']),
        )
        return result
    prediction = predict_doc(model, text)
    pieces = [dict(start=s, end=e, ai_prob=p) for s, e, p in prediction['pieces']]
    if len(pieces) > 1:
        notes.insert(2, '文字较长，分成 {} 段分别量，再取平均。'.format(len(pieces)))
    ai_prob = prediction['ai_prob']
    if ai_prob >= limits['ai_high']:
        verdict, label = 'likely_ai', '很像 AI 写的'
    elif ai_prob <= limits['human_low']:
        verdict, label = 'likely_human', '更像人写的'
    else:
        verdict, label = 'uncertain', '不确定'
    which = [dict(name=name, p=p) for name, p in zip(model['classes'][1:], prediction['which'])]
    which.sort(key=lambda row: row['p'], reverse=True)
    if verdict == 'likely_human':
        which_text = '更像人写的，这一栏仅供参考'
    elif which[0]['p'] < limits['which_min']:
        which_text = '三家都有点像，分不清'
    else:
        which_text = '最像 ' + which[0]['name']
    top_class = model['classes'].index(which[0]['name'])
    result.update(
        verdict=verdict, verdict_text=label, ai_prob=ai_prob,
        probs=dict(zip(model['classes'], prediction['probs'])), which=which, which_text=which_text,
        evidence=_evidence(text, model, top_class), pieces=pieces,
    )
    return result


def analyze(text, model=None):
    """Return a JSON-serialisable result, loading the default model when omitted."""
    return analyze_loaded(text, load_model() if model is None else model)
