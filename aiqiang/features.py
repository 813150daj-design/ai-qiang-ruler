"""Model mathematics, ported in operation order from train/ref_features.py."""

import math
import re
import unicodedata

_URL = re.compile(r'(https?://|www\.)\S+')
_AT = re.compile(r'@[\w一-鿿\-]{1,30}')
_EMO = re.compile(r'\[[^\[\]\s]{1,8}\]')
_TAG = re.compile(r'#[^#\n]{1,30}#')
_DQ = '“”「」『』〝〞〟"'
_SQ = '‘’\''
_DOTS = re.compile(r'(\.{2,}|。{2,}|⋯+|·{3,})')
_DASH = re.compile(r'[—–―‒\-]{2,}|[—–―‒]')
_DIG = re.compile(r'[0-9]+(?:[.,][0-9]+)*')
_LAT = re.compile(r'[a-z]+')
_WS = re.compile(r'\s+')
_MD_HR = re.compile(r'^[ \t]*([-*_])[ \t]*(?:\1[ \t]*){2,}$', re.M)
_MD_LINE = re.compile(r'^[ \t]*(#{1,6}[ \t]+|>[ \t]*|[-*+][ \t]+)', re.M)
_MD_INLINE = re.compile(r'\*\*|__|`')

KEEP = ''.join([
    '我你他她它们咱您自己人家',
    '的了着过地得之所者也吗呢吧啊呀哦嘛啦哈么嘞呗哇噢唉嗯',
    '是有在会能要想可以应该需',
    '不没别未无非',
    '很太更最都也还又就才只却再已经曾正刚总一直真挺特非常比较越尤其几乎甚至竟然居然其实果然当',
    '和与跟同及或而且但并因为所如果虽然后于是此即使哪怕既尽管否则除',
    '把被让给对向从到以按照关根据通随着',
    '这那哪每各某些个种样点次件位一二两三几多少什怎谁',
    '上下里外中前后时候今天年月日现',
    '说看来去做起开出进回成',
])
PUNCT = set('，。！？、；：,.!?;:"\'()（）《》<>…—·/')
MASK = 'x'


def normalize(text):
    """Apply exactly the reference's normalisation, in the same order."""
    t = unicodedata.normalize('NFKC', text)
    t = _MD_HR.sub('', t)
    t = _MD_LINE.sub('', t)
    t = _MD_INLINE.sub('', t)
    t = _URL.sub(' ', t)
    t = _AT.sub(' ', t)
    t = _EMO.sub(' ', t)
    t = _TAG.sub(' ', t)
    t = t.translate({ord(c): '"' for c in _DQ})
    t = t.translate({ord(c): "'" for c in _SQ})
    t = t.replace('…', '...')
    t = _DOTS.sub('…', t)
    t = _DASH.sub('—', t)
    t = t.lower()
    t = _DIG.sub('0', t)
    t = _LAT.sub('a', t)
    t = _WS.sub('', t)
    return t


def cjk_count(text):
    return sum(1 for ch in text if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿')


def distort(norm, keep=KEEP, punct=PUNCT, mask=MASK):
    out = []
    for ch in norm:
        if ch in keep or ch in punct:
            out.append(ch)
        elif ch.isalnum():
            if not out or out[-1] != mask:
                out.append(mask)
        else:
            out.append(ch)
    return ''.join(out)


def ngram_counts(s, ns):
    counts = {}
    for n in ns:
        for i in range(len(s) - n + 1):
            gram = s[i:i + n]
            counts[gram] = counts.get(gram, 0) + 1
    return counts


def source_string(text, fs):
    norm = normalize(text)
    if fs['source'] == 'plain':
        return norm
    if fs['source'] == 'distorted':
        return distort(
            norm, fs.get('keep', KEEP),
            fs.get('punct', ''.join(sorted(PUNCT))), fs.get('mask', MASK),
        )
    raise ValueError('unknown source ' + fs['source'])


def featurize(text, fs):
    counts = ngram_counts(source_string(text, fs), fs['n'])
    vocab = fs['vocab']
    vals = {}
    for gram, tf in counts.items():
        entry = vocab.get(gram)
        if entry is not None:
            vals[gram] = (1.0 + math.log(tf)) * entry[0]
    norm = math.sqrt(sum(v * v for v in vals.values()))
    if norm > 0:
        for gram in vals:
            vals[gram] /= norm
    return vals


def logits(model, text):
    k = len(model['classes'])
    z = list(model['bias'])
    for fs in model['feature_sets']:
        vocab = fs['vocab']
        for gram, value in featurize(text, fs).items():
            weights = vocab[gram]
            for c in range(k):
                z[c] += value * weights[1 + c]
    return z


def softmax(z):
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    total = sum(exps)
    return [v / total for v in exps]


def sigmoid(z):
    """Numerically stable logistic function, matching the training reference."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


LONG_TEXT = dict(min_len=500, target=300, min_piece=150, min_cjk=30)
_UNIT = re.compile(r'[^。！？!?；;…\n]*(?:[。！？!?；;…]+|\n+|$)')


def split_pieces(text, params=None):
    """Cut long texts at sentence ends or newlines once a piece reaches target stripped characters.

    Join a final remainder shorter than min_piece to the previous piece. Return whitespace-trimmed,
    half-open spans in the original text. Texts at most min_len characters, or fewer than two trimmed
    pieces, return one span covering the whole text, matching the training reference.
    """
    p = dict(LONG_TEXT, **(params or {}))
    if len(text) <= p['min_len']:
        return [(0, len(text))]
    spans, start, size = [], None, 0
    for m in _UNIT.finditer(text):
        if m.start() == m.end():
            continue
        if start is None:
            start = m.start()
        size += len(m.group().strip())
        if size >= p['target']:
            spans.append([start, m.end()])
            start, size = None, 0
    if start is not None:
        if spans and size < p['min_piece']:
            spans[-1][1] = len(text)
        else:
            spans.append([start, len(text)])
    out = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            out.append((a, b))
    if len(out) < 2:
        return [(0, len(text))]
    return out


def predict_doc(model, text):
    """Score each piece, skipping pieces below min_cjk; fall back to the whole text if none remain.

    Average ai_score, logits, probs and which equally over scored pieces, then calibrate the mean score.
    pieces contains [start, end, ai_prob] in the original text for each scored piece, including short texts.
    """
    p = dict(LONG_TEXT, **(model.get('long_text') or {}))
    spans = split_pieces(text, p)
    if len(spans) > 1:
        spans = [s for s in spans if cjk_count(text[s[0]:s[1]]) >= p['min_cjk']] or [(0, len(text))]
    preds = [predict(model, text[a:b]) for a, b in spans]
    if len(preds) == 1:
        out = dict(preds[0])
        out['pieces'] = [[spans[0][0], spans[0][1], preds[0]['ai_prob']]]
        return out
    k = len(preds)

    def mean(key):
        return [sum(x[key][i] for x in preds) / k for i in range(len(preds[0][key]))]

    score = sum(x['ai_score'] for x in preds) / k
    cal = model['ai_calibration']
    return dict(
        logits=mean('logits'), probs=mean('probs'), ai_score=score,
        ai_prob=sigmoid(cal['a'] * score + cal['b']), which=mean('which'),
        pieces=[[a, b, x['ai_prob']] for (a, b), x in zip(spans, preds)],
    )


def predict(model, text):
    z = logits(model, text)
    p = softmax(z)
    m = max(z[1:])
    lse = m + math.log(sum(math.exp(v - m) for v in z[1:]))
    score = lse - z[0]
    cal = model['ai_calibration']
    ai_prob = sigmoid(cal['a'] * score + cal['b'])
    which = softmax(z[1:])
    return dict(logits=z, probs=p, ai_score=score, ai_prob=ai_prob, which=which)
