"""Reference definitions shared by training and the program (11-14 night).

This file is the single written-down definition of:
  normalize(text)        the text clean-up every model feature is computed on
  distort(norm)          the "function-word skeleton": rare characters masked, so topic words drop out
  ngram_counts(s, ns)    character n-gram counts
  featurize(text, fs)    TF-IDF values of one feature set, L2-normalised
  predict(model, text)   logits, class probabilities, AI score, calibrated AI probability
  markers(text)          the 10 「AI 八股」 checks, same rules as 科普/AI腔/实验/ai_score.py

The program in ../aiqiang must give the same numbers (tests compare against golden.json made from this file).
Standard library only.
"""
import math
import re
import unicodedata

# ---------------------------------------------------------------- normalisation (model features only)
_URL = re.compile(r'(https?://|www\.)\S+')
_AT = re.compile(r'@[\w一-鿿\-]{1,30}')
_EMO = re.compile(r'\[[^\[\]\s]{1,8}\]')          # weibo-style emoticon codes like [哈哈]
_TAG = re.compile(r'#[^#\n]{1,30}#')
_DQ = '“”「」『』〝〞〟"'   # “ ” 「 」 『 』 〝 〞 〟 "
_SQ = '‘’\''                                       # ‘ ’ '
_DOTS = re.compile(r'(\.{2,}|。{2,}|⋯+|·{3,})')
_DASH = re.compile(r'[—–―‒\-]{2,}|[—–―‒]')
_DIG = re.compile(r'[0-9]+(?:[.,][0-9]+)*')
_LAT = re.compile(r'[a-z]+')
_WS = re.compile(r'\s+')
_MD_HR = re.compile(r'^[ \t]*([-*_])[ \t]*(?:\1[ \t]*){2,}$', re.M)           # markdown rule lines: --- *** ___
_MD_LINE = re.compile(r'^[ \t]*(#{1,6}[ \t]+|>[ \t]*|[-*+][ \t]+)', re.M)   # markdown heading/quote/bullet marks
_MD_INLINE = re.compile(r'\*\*|__|`')                                        # markdown bold / code marks


def normalize(text):
    """NFKC, drop markdown marks, urls/@names/emoticon codes/#tags#, unify quotes, ellipses and dashes,
    every digit run -> '0', every latin-letter run -> 'a', remove all whitespace.
    (Markdown marks are dropped so the model learns wording, not formatting; the 客服/格式 check still sees them.)"""
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
    t = t.replace('…', '...')          # NFKC already does this; kept for clarity
    t = _DOTS.sub('…', t)
    t = _DASH.sub('—', t)
    t = t.lower()
    t = _DIG.sub('0', t)
    t = _LAT.sub('a', t)
    t = _WS.sub('', t)
    return t


def cjk_count(text):
    """number of CJK unified ideographs (the length the program reports and gates on)"""
    return sum(1 for ch in text if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿')


# ---------------------------------------------------------------- function-word skeleton
# Characters kept as they are; every other letter/ideograph/digit becomes 'x' and runs of 'x' merge into one.
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


def distort(norm, keep=KEEP, punct=PUNCT, mask=MASK):
    """keep the characters in keep/punct; every other alphanumeric character (ideographs included)
    becomes mask, and a run of masks merges into one; any other character stays as it is."""
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


# ---------------------------------------------------------------- n-grams and TF-IDF
def ngram_counts(s, ns):
    c = {}
    for n in ns:
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            c[g] = c.get(g, 0) + 1
    return c


def source_string(text, fs):
    norm = normalize(text)
    if fs['source'] == 'plain':
        return norm
    if fs['source'] == 'distorted':
        return distort(norm, fs.get('keep', KEEP), fs.get('punct', ''.join(sorted(PUNCT))), fs.get('mask', MASK))
    raise ValueError('unknown source ' + fs['source'])


def featurize(text, fs):
    """{ngram: value} for one feature set: (1 + ln tf) * idf over the set's vocabulary, then L2-normalised."""
    counts = ngram_counts(source_string(text, fs), fs['n'])
    vocab = fs['vocab']
    vals = {}
    for g, tf in counts.items():
        e = vocab.get(g)
        if e is not None:
            vals[g] = (1.0 + math.log(tf)) * e[0]
    norm = math.sqrt(sum(v * v for v in vals.values()))
    if norm > 0:
        for g in vals:
            vals[g] /= norm
    return vals


def logits(model, text):
    k = len(model['classes'])
    z = list(model['bias'])
    for fs in model['feature_sets']:
        vocab = fs['vocab']
        for g, v in featurize(text, fs).items():
            w = vocab[g]
            for c in range(k):
                z[c] += v * w[1 + c]
    return z


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def sigmoid(z):
    """numerically stable logistic function (no overflow for very negative z)"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


LONG_TEXT = dict(min_len=500, target=300, min_piece=150, min_cjk=30)
_UNIT = re.compile(r'[^。！？!?；;…\n]*(?:[。！？!?；;…]+|\n+|$)')


def split_pieces(text, params=None):
    """Long texts are scored piece by piece, like the training texts (150-400 characters).
    Units end after a run of sentence-final marks (。！？!?；;…) or at a run of newlines. Units are added to the
    current piece until it has at least `target` characters (stripped); a last remainder shorter than `min_piece`
    joins the previous piece. Returns [(start, end)] half-open spans in the original text, trimmed of whitespace.
    Texts of at most `min_len` characters, or texts that give fewer than 2 pieces, return one span: the whole text."""
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
    """predict() for any length. A long text (see split_pieces) is cut into pieces; pieces with fewer than
    `min_cjk` CJK characters are skipped; every remaining piece gets predict(); the text's ai_score, logits,
    probs and which are the plain means over the pieces; ai_prob = sigmoid(a*mean ai_score + b).
    'pieces' lists [start, end, ai_prob] of every scored piece (one entry covering the text when it is short)."""
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
    return dict(logits=mean('logits'), probs=mean('probs'), ai_score=score,
                ai_prob=sigmoid(cal['a'] * score + cal['b']), which=mean('which'),
                pieces=[[a, b, x['ai_prob']] for (a, b), x in zip(spans, preds)])


def predict(model, text):
    """classes[0] is the human class. ai_score = log P(any AI) - log P(human); ai_prob = sigmoid(a*ai_score + b)."""
    z = logits(model, text)
    p = softmax(z)
    m = max(z[1:])
    lse = m + math.log(sum(math.exp(v - m) for v in z[1:]))
    score = lse - z[0]
    cal = model['ai_calibration']
    ai_prob = sigmoid(cal['a'] * score + cal['b'])
    which = softmax(z[1:])
    return dict(logits=z, probs=p, ai_score=score, ai_prob=ai_prob, which=which)


# ---------------------------------------------------------------- the 10 「AI 八股」 checks (raw text, counts)
_UPLIFT = re.compile(r'人生|生活|日子|岁月|时光|幸福|温柔|治愈|意义|勇气|成长|慈悲|美好|滋味|人间|福气|骄傲')
CHECKS = [
    ('不是…而是', re.compile(r'不是[^。！？!?\n]{1,30}?而是|不是[^。！？!?\n，,]{1,25}[，,]\s*(而)?是|与其说[^。！？!?\n]{1,30}不如说')),
    ('不仅…更', re.compile(r'不(仅|只|单)(仅|是)?[^。！？!?\n]{1,30}?(更|还|也)')),
    ('原来/忽然', re.compile(r'原来|才(发现|明白|懂得|知道)|(忽然|突然|蓦然|这才)(就)?(觉得|发现|明白|懂得|意识到)|终于(明白|懂了)')),
    ('这一刻', re.compile(r'这一刻|那一刻|此刻|瞬间')),
    ('破折号', re.compile(r'——|—')),
    ('三连排比', None),
    ('结尾升华', None),
    ('首先/总之', re.compile(r'首先|其次|总之|总而言之|综上所述|值得注意的是|总的来说|换句话说|某种意义上|不得不说|说到底')),
    ('大词抒情', re.compile(r'量子|熵|宇宙|星河|星辰|永恒|灵魂|褶皱|缝隙|共振|维度|意义感')),
    ('客服/格式', re.compile(r'希望(这些|以上|对你)|如果你(愿意|需要)|需要的话|好问题|很高兴|让我们|以下是'
                          r'|^\s*(#{1,4} |[-*] |\d+[.、] )|\*\*', re.M)),
]


def sentences(t):
    return [s for s in re.split(r'(?<=[。！？!?])', t.replace('\n', '')) if s.strip()]


def parallel(t):
    """three or more clauses in a row, inside one sentence, that open with the same character"""
    n = 0
    for s in sentences(t):
        parts = [p.strip() for p in re.split(r'[，、；,]', s) if p.strip()]
        run = 1
        for a, b in zip(parts, parts[1:]):
            run = run + 1 if a[:1] == b[:1] else 1
            if run == 3:
                n += 1
    return n


def markers(t):
    """{check name: count}; the 八股指数 is the number of checks with count > 0 (0-10)"""
    hits = {}
    for name, rx in CHECKS:
        if name == '三连排比':
            n = parallel(t)
        elif name == '结尾升华':
            s = sentences(t)
            n = 1 if s and _UPLIFT.search(s[-1]) else 0
        else:
            n = len(rx.findall(t))
        hits[name] = n
    return hits


def baigu_index(t):
    return sum(1 for v in markers(t).values() if v)
