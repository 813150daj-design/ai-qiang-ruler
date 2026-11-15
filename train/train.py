"""Train and evaluate the AI腔尺子 model (11-15). numpy + scipy for training only; the program needs neither.

Data
  AI:    ../data/ai/{deepseek,gpt,claude}/*.txt: prompts.json (24 everyday prompts x 3 samples x 3 models, plus the
         18 texts of the video experiment) and prompts_v2.json (58 prompts in the human genres, 1 text per model).
         A text longer than 500 characters is cut into 150-400 character pieces (at most 3).
  human: <work>/human/human.jsonl from build_human.py (texts stay outside the repo)

Model
  4 classes (人 / DeepSeek / GPT / Claude), multinomial logistic regression with L2, on TF-IDF of
  character n-grams of the normalised text ("char") and of the function-word skeleton ("skeleton").
  Sample weights: humans 1/2 of the total, each AI 1/6, so AI and human count the same.
  Defaults are the shipped settings (v1.1): char 1-3 + skeleton 2-4, lam 1e-4, min_df 5, max_fpr 0.5%.

Honest evaluation
  - AI texts: 6 folds BY PROMPT, so every AI score comes from a model that never saw that prompt (topic shift).
  - human texts usable for training: 6 folds by document (out-of-fold scores), plus unused documents.
  - human sources we may not train on (student essays, Weibo, other HC3 parts, famous PD pieces): never trained on;
    each is split in two by document: the "calib" half helps choose thresholds, the "test" half is only reported.
  - texts outside the pool are scored like the program scores them: long ones piece by piece, then averaged.
  - calibration: Platt scaling of the AI score, AI and human 50/50, every human group weighted the same.
  - thresholds: ai_high so that no human group with >= 100 texts gets more than MAX_FPR flagged
    (using out-of-fold + calib halves); human_low so that at most 5% of AI texts fall under it.

  python train.py                      train with the shipped settings, write model.json and reports/
  python train.py --auto --kinds char2+skel5,char3+skel4 --lams 3e-5,1e-4    try settings, export the best
  python train.py --small              tiny smoke test of the whole pipeline
"""
import argparse
import collections
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import scipy.optimize
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ref_features as R  # noqa: E402

CLASSES = ['人', 'DeepSeek', 'GPT', 'Claude']
AI_DIRS = {'deepseek': 1, 'gpt': 2, 'claude': 3}
MAX_FPR = 0.005
FOLDS = 6

TRAIN_SOURCES = {'clue': 3000, 'hc3_open_qa': 900, 'hc3_finance': 500, 'hc3_psychology': 900, 'cmrc': 700,
                 'pd_other': None}
EVAL_SOURCES = {'cedcc': None, 'weibo': 3000, 'hc3_baike': 1200, 'hc3_nlpcc_dbqa': None,
                'hc3_medicine': None, 'hc3_law': None, 'pd_famous': None}
HELDOUT_PER_TRAIN_SOURCE = 600   # unused documents of the training sources, scored by the final model
OLD_GPT_TRAIN = 150              # 2023 ChatGPT answers (licensed HC3 splits) added to the GPT class

SKELETON_KEEP = R.KEEP
SKELETON_PUNCT = ''.join(sorted(R.PUNCT))


def feature_sets(kind):
    sets = []
    if 'char' in kind:
        n = [1, 2, 3] if kind.get('char') == 3 else [1, 2]
        sets.append(dict(name='char', source='plain', n=n))
    if 'skeleton' in kind:
        sets.append(dict(name='skeleton', source='distorted', n=list(range(2, kind['skeleton'] + 1)),
                         keep=SKELETON_KEEP, punct=SKELETON_PUNCT, mask=R.MASK))
    return sets


# ------------------------------------------------------------------------------------------------ data
def load_ai(cut=True):
    """AI texts; a text longer than 500 characters is cut into 150-400 character pieces at sentence ends
    (at most 3), the same way the human texts are cut, so long AI answers and long human answers compare."""
    from build_human import chunks
    out = []
    for d, label in AI_DIRS.items():
        folder = os.path.join(ROOT, 'data', 'ai', d)
        for f in sorted(os.listdir(folder)):
            m = re.match(r'(old_)?([pqr]\d+)_s(\d+)\.txt$', f)
            if not m:
                continue
            t = io.open(os.path.join(folder, f), encoding='utf-8').read().strip()
            pieces = [t]
            if cut and len(t) > 500:
                pieces = [x for x in chunks(t, random.Random(f)) if R.cjk_count(x) >= 50][:3] or [t]
            for k, piece in enumerate(pieces):
                out.append(dict(id='%s:%s#%d' % (d, f[:-4], k), y=label, prompt=m.group(2), group=CLASSES[label],
                                text=piece, sample=('old' if m.group(1) else '') + m.group(3), whole=f[:-4]))
    return out


def load_human(work):
    by = collections.defaultdict(list)
    for line in io.open(os.path.join(work, 'human', 'human.jsonl'), encoding='utf-8'):
        x = json.loads(line)
        by[x['source']].append(x)
    return by


def essay_key(x):
    # CEDCC whole essays and their pieces belong to the same essay: keep them on the same side of every split
    return re.sub(r':(whole|part)_', ':', x['doc'])


def pick_docs(xs, n, rng):
    docs = sorted(set(essay_key(x) for x in xs))
    rng.shuffle(docs)
    return docs if n is None else docs[:n]


# ------------------------------------------------------------------------------------------------ features
def gram_counts(texts, fs):
    """a generator, so only one text's counts are in memory at a time"""
    for t in texts:
        yield R.ngram_counts(R.source_string(t, fs), fs['n'])


def build_matrix(counts, vocab_index):
    indptr, indices, data = [0], [], []
    for c in counts:
        for g, v in c.items():
            j = vocab_index.get(g)
            if j is not None:
                indices.append(j)
                data.append(v)
        indptr.append(len(indices))
    return sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32),
                          np.array(indptr, dtype=np.int64)), shape=(len(indptr) - 1, len(vocab_index)))


def tfidf(C, idf):
    X = C.copy().tocsr()
    X.data = (1.0 + np.log(X.data)) * idf[X.indices]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ X


class Featurizer:
    """counts once for all texts (over a global vocabulary: grams seen in >= 2 pool texts);
    a fold picks its own vocabulary (df >= min_df in its training rows) and idf from those counts."""

    def __init__(self, sets, pool_texts, all_texts):
        self.sets = sets
        self.C = []
        self.grams = []
        for fs in sets:
            df = collections.Counter()
            for c in gram_counts(pool_texts, fs):
                df.update(c.keys())
            grams = sorted(g for g, v in df.items() if v >= 2)
            index = {g: i for i, g in enumerate(grams)}
            self.grams.append(grams)
            self.C.append(build_matrix(gram_counts(all_texts, fs), index))
            print('  feature set %-8s n=%s  %d grams (df>=2 in pool)' % (fs['name'], fs['n'], len(grams)), flush=True)

    def fit_vocab(self, rows, min_df):
        self.cols, self.idf = [], []
        N = len(rows)
        for C in self.C:
            df = np.asarray((C[rows] > 0).sum(0)).ravel()
            cols = np.where(df >= min_df)[0]
            self.cols.append(cols)
            self.idf.append(np.log((1.0 + N) / (1.0 + df[cols])) + 1.0)

    def transform(self, rows):
        return sp.hstack([tfidf(C[rows][:, cols], idf) for C, cols, idf in zip(self.C, self.cols, self.idf)]).tocsr()


# ------------------------------------------------------------------------------------------------ model
def fit_lr(X, y, w, K, lam, maxiter=400):
    n, F = X.shape
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    w = w / w.sum()
    XT = X.T.tocsr()

    def f(theta):
        W = theta[:F * K].reshape(F, K)
        b = theta[F * K:]
        Z = X @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        loss = -(w * np.log(P[np.arange(n), y] + 1e-300)).sum() + 0.5 * lam * (W * W).sum()
        G = (P - Y) * w[:, None]
        gW = XT @ G + lam * W
        return loss, np.concatenate([gW.ravel(), G.sum(0)])

    res = scipy.optimize.minimize(f, np.zeros(F * K + K), jac=True, method='L-BFGS-B',
                                  options=dict(maxiter=maxiter))
    return res.x[:F * K].reshape(F, K), res.x[F * K:]


def scores(X, W, b):
    Z = X @ W + b
    m = Z[:, 1:].max(1)
    lse = m + np.log(np.exp(Z[:, 1:] - m[:, None]).sum(1))
    which = np.exp(Z[:, 1:] - m[:, None])
    which /= which.sum(1, keepdims=True)
    return lse - Z[:, 0], which, Z


OLD_SHARE = 0.3   # inside the GPT class, the 2023 ChatGPT answers get this share of the class weight


def weights(y, old=None):
    """humans 1/2 of the total weight, each AI class 1/6; inside a class that also holds old (2023) answers,
    those get OLD_SHARE of the class weight and the current models the rest"""
    y = np.asarray(y)
    old = np.zeros(len(y), bool) if old is None else np.asarray(old, bool)
    w = np.zeros(len(y))
    nh = (y == 0).sum()
    w[y == 0] = 0.5 / max(nh, 1)
    for c in (1, 2, 3):
        sel = y == c
        if not sel.any():
            continue
        s_old, s_new = sel & old, sel & ~old
        if s_old.any() and s_new.any():
            w[s_new] = (0.5 / 3) * (1 - OLD_SHARE) / s_new.sum()
            w[s_old] = (0.5 / 3) * OLD_SHARE / s_old.sum()
        else:
            w[sel] = (0.5 / 3) / sel.sum()
    return w * len(y)


def platt(s, is_ai, groups=None):
    """sigmoid(a*s+b) fitted with AI and human weighted 50/50; inside the human half every group
    (source) weighs the same, so a small hard group (student essays) counts as much as a big easy one."""
    s = np.asarray(s, float)
    t = np.asarray(is_ai, float)
    w = np.where(t == 1, 0.5 / t.sum(), 0.0)
    if groups is None:
        w = np.where(t == 1, w, 0.5 / (1 - t).sum())
    else:
        g = np.asarray(groups)
        hg = sorted(set(g[t == 0]))
        for name in hg:
            sel = (t == 0) & (g == name)
            w[sel] = 0.5 / len(hg) / sel.sum()

    def f(ab):
        z = ab[0] * s + ab[1]
        p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        loss = -(w * (t * np.log(p + 1e-12) + (1 - t) * np.log(1 - p + 1e-12))).sum()
        g = w * (p - t)
        return loss, np.array([(g * s).sum(), g.sum()])

    r = scipy.optimize.minimize(f, np.array([1.0, 0.0]), jac=True, method='L-BFGS-B')
    return float(r.x[0]), float(r.x[1])


def sigmoid(z):
    z = np.asarray(z, float)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


# ------------------------------------------------------------------------------------------------ experiment
def prepare(work, seed):
    rng = random.Random(seed)
    ai = load_ai()
    hb = load_human(work)
    pool, heldout, evalset = [], [], []
    for src, n in TRAIN_SOURCES.items():
        docs = pick_docs(hb[src], None, rng)
        if n is None:                    # small source: 5/6 for training (out-of-fold scored), 1/6 held out
            n = len(docs) - len(docs) // 6
        use, rest = set(docs[:n]), docs[n:n + HELDOUT_PER_TRAIN_SOURCE]
        rest = set(rest)
        for x in hb[src]:
            k = essay_key(x)
            if k in use:
                pool.append(dict(x, y=0, group=src))
            elif k in rest:
                heldout.append(dict(x, y=0, group=src, half='heldout'))
    for src, n in EVAL_SOURCES.items():
        docs = pick_docs(hb[src], n, rng)
        half = {d: ('calib' if i % 2 == 0 else 'test') for i, d in enumerate(docs)}
        for x in hb[src]:
            k = essay_key(x)
            if k in half:
                evalset.append(dict(x, y=0, group=src, half=half[k]))
    # older ChatGPT (gpt-3.5, early 2023) answers from HC3. The splits we may train on (open_qa MIT, finance CC BY,
    # psychology CC0) give OLD_GPT_TRAIN answers to the GPT class (one question = one prompt group); the other
    # splits are never trained on and give the "old ChatGPT" test.
    from build_human import chunks
    old_train, old_gpt = [], []
    for split in ('open_qa', 'finance', 'psychology', 'baike', 'medicine', 'nlpcc_dbqa', 'law'):
        p = os.path.join(work, 'human_raw', 'hc3', split + '.jsonl')
        for i, line in enumerate(io.open(p, encoding='utf-8')):
            d = json.loads(line)
            for a in (d.get('chatgpt_answers') or [])[:1]:
                if a and R.cjk_count(a) >= 50:
                    piece = chunks(a.strip(), random.Random(i))[0]
                    if R.cjk_count(piece) >= 50:
                        x = dict(id='hc3gpt:%s:%d' % (split, i), y=2, text=piece)
                        if split in ('open_qa', 'finance', 'psychology'):
                            old_train.append(dict(x, group='GPT', prompt='hc3:%s:%d' % (split, i), sample='hc3',
                                                  whole=x['id'], old=True))
                        else:
                            old_gpt.append(dict(x, group='old_chatgpt', half='test'))
    rng.shuffle(old_train)
    rng.shuffle(old_gpt)
    old_gpt = old_gpt[:600]
    # Gemini (gen_gemini.py): an AI the model has never seen, test only
    gdir = os.path.join(ROOT, 'data', 'ai_eval', 'gemini')
    if os.path.isdir(gdir):
        for f in sorted(os.listdir(gdir)):
            if f.endswith('.txt'):
                t = io.open(os.path.join(gdir, f), encoding='utf-8').read().strip()
                if R.cjk_count(t) >= 50:
                    old_gpt.append(dict(id='gemini:' + f[:-4], y=9, group='gemini', text=t, half='test'))
    pool = ai + old_train[:OLD_GPT_TRAIN] + pool
    # folds: AI by prompt, human by document
    prompts = sorted(set(x['prompt'] for x in ai), key=lambda p: int(p[1:]))
    rng.shuffle(prompts)
    pf = {p: i % FOLDS for i, p in enumerate(prompts)}
    for x in pool:
        if x['y'] > 0 and x['prompt'] in pf:
            x['fold'] = pf[x['prompt']]
        elif x['y'] > 0:
            x['fold'] = int(hashlib.md5(x['prompt'].encode('utf-8')).hexdigest(), 16) % FOLDS
        else:
            x['fold'] = int(hashlib.md5(essay_key(x).encode('utf-8')).hexdigest(), 16) % FOLDS
    return pool, heldout, evalset, old_gpt


def piece_texts(x):
    """the pieces the program scores for this text (ref_features.split_pieces; short texts: the text itself)"""
    if 'pieces' not in x:
        t = x['text']
        spans = R.split_pieces(t)
        if len(spans) > 1:
            spans = [s for s in spans if R.cjk_count(t[s[0]:s[1]]) >= R.LONG_TEXT['min_cjk']] or [(0, len(t))]
        x['pieces'] = [t[a:b] for a, b in spans]
    return x['pieces']


def make_featurizer(pool, others, kind):
    texts = [x['text'] for x in pool] + [p for x in others for p in piece_texts(x)]
    return Featurizer(feature_sets(kind), [x['text'] for x in pool], texts)


def run(pool, heldout, evalset, old_gpt, kind, lam, min_df, verbose=True, fz=None):
    sets = feature_sets(kind)
    others = heldout + evalset + old_gpt
    t0 = time.time()
    if fz is None:
        fz = make_featurizer(pool, others, kind)
    n = len(pool)
    y = np.array([x['y'] for x in pool])
    w = weights(y, [bool(x.get('old')) for x in pool])
    folds = np.array([x['fold'] for x in pool])
    oof_s = np.zeros(n)
    oof_which = np.zeros((n, 3))
    for k in range(FOLDS):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        fz.fit_vocab(tr, min_df)
        Xtr = fz.transform(tr)
        W, b = fit_lr(Xtr, y[tr], w[tr], 4, lam)
        s, wh, _ = scores(fz.transform(te), W, b)
        oof_s[te] = s
        oof_which[te] = wh
        if verbose:
            print('  fold %d: %d train, %d test, %d features, %.0fs' % (k, len(tr), len(te), Xtr.shape[1],
                                                                       time.time() - t0), flush=True)
    # final model on the whole pool
    allrows = np.arange(n)
    fz.fit_vocab(allrows, min_df)
    W, b = fit_lr(fz.transform(allrows), y, w, 4, lam)
    # texts not in the pool are scored the way the program scores them: long ones piece by piece, then averaged
    parent = np.array([i for i, x in enumerate(others) for _ in piece_texts(x)])
    orows = np.arange(n, n + len(parent))
    p_s, p_which, _ = scores(fz.transform(orows), W, b)
    cnt = np.bincount(parent, minlength=len(others)).astype(float)
    o_s = np.bincount(parent, weights=p_s, minlength=len(others)) / cnt
    o_which = np.stack([np.bincount(parent, weights=p_which[:, c], minlength=len(others)) for c in range(3)], 1)
    o_which /= cnt[:, None]
    return dict(fz=fz, W=W, b=b, sets=sets, oof_s=oof_s, oof_which=oof_which, o_s=o_s, o_which=o_which,
                others=others, seconds=time.time() - t0)


def evaluate(pool, res, max_fpr=None):
    max_fpr = MAX_FPR if max_fpr is None else max_fpr
    y = np.array([x['y'] for x in pool])
    others = res['others']
    # calibration on out-of-fold scores (pool) + calib halves of eval-only human sources
    cal_s = list(res['oof_s']) + [s for x, s in zip(others, res['o_s']) if x.get('half') == 'calib']
    cal_t = list((y > 0).astype(int)) + [0 for x in others if x.get('half') == 'calib']
    cal_g = [x['group'] for x in pool] + [x['group'] for x in others if x.get('half') == 'calib']
    a, bb = platt(cal_s, cal_t, cal_g)
    old_mask = np.array([bool(x.get('old')) for x in pool])   # 2023 ChatGPT answers inside the GPT class
    p_oof = sigmoid(a * res['oof_s'] + bb)
    p_oth = sigmoid(a * res['o_s'] + bb)
    # groups for threshold choice: each human group, out-of-fold (pool) or calib half
    groups = collections.defaultdict(list)
    for x, p in zip(pool, p_oof):
        if x['y'] == 0:
            groups[x['group']].append(p)
    for x, p in zip(others, p_oth):
        if x.get('half') == 'calib':
            groups[x['group']].append(p)
    q = {g: float(np.quantile(v, 1 - max_fpr)) for g, v in groups.items() if len(v) >= 100}
    ai_high = min(0.999999, max(0.5, max(q.values()) + 1e-6))
    ai_p = p_oof[(y > 0) & ~old_mask]
    human_low = float(min(0.5, np.quantile(ai_p, 0.05)))
    if human_low >= ai_high:
        human_low = ai_high * 0.5
    # report
    rep = dict(calibration=dict(a=a, b=bb), thresholds=dict(ai_high=ai_high, human_low=human_low), groups={})

    def verdicts(ps):
        ps = np.asarray(ps)
        return dict(n=int(len(ps)), likely_ai=float((ps >= ai_high).mean()),
                    uncertain=float(((ps > human_low) & (ps < ai_high)).mean()),
                    likely_human=float((ps <= human_low).mean()))

    for c in (1, 2, 3):
        sel = (y == c) & ~old_mask
        v = verdicts(p_oof[sel])
        guess = res['oof_which'][sel].argmax(1) + 1
        v['which_acc'] = float((guess == c).mean())
        v['which_acc_when_flagged'] = float((guess[p_oof[sel] >= ai_high] == c).mean()) if (
            p_oof[sel] >= ai_high).any() else None
        rep['groups'][CLASSES[c]] = dict(kind='AI (out-of-fold, unseen prompts)', **v)
    sel = (y > 0) & ~old_mask
    guess = res['oof_which'][sel].argmax(1) + 1
    conf = np.zeros((3, 3), int)
    for t, g in zip(y[sel], guess):
        conf[t - 1, g - 1] += 1
    rep['which_confusion'] = conf.tolist()
    rep['which_acc'] = float((guess == y[sel]).mean())
    v = verdicts(p_oof[sel])
    rep['ai_all'] = v
    if old_mask.any():
        og = verdicts(p_oof[old_mask])
        og['which_acc'] = float((res['oof_which'][old_mask].argmax(1) + 1 == 2).mean())
        rep['groups']['old_chatgpt_train/oof'] = dict(kind='AI (ChatGPT 2023, trained on other questions)', **og)
    # long AI texts were cut into pieces for training; the program averages the pieces of a long text, so the
    # whole-text result is the mean of the out-of-fold piece scores
    by_whole = collections.defaultdict(list)
    for x, s in zip(pool, res['oof_s']):
        if x['y'] > 0 and not x.get('old'):
            by_whole[(x['y'], x['whole'])].append(s)
    long_docs = [float(np.mean(v)) for v in by_whole.values() if len(v) > 1]
    if long_docs:
        rep['ai_long_docs'] = verdicts(sigmoid(a * np.array(long_docs) + bb))
    # AI results by genre of the prompt (out-of-fold), and by batch (v1 everyday prompts / v2 human-genre prompts)
    genre = {}
    for fn in ('prompts.json', 'prompts_v2.json', 'prompts_v3.json'):
        for pr in json.load(io.open(os.path.join(HERE, fn), encoding='utf-8'))['prompts']:
            genre[pr['id']] = pr['genre'].split('·')[0]
    rep['ai_by_genre'] = {}
    for gname in sorted(set(genre.values())):
        sel = np.array([x['y'] > 0 and genre.get(x.get('prompt')) == gname for x in pool])
        if sel.any():
            vv = verdicts(p_oof[sel])
            g2 = res['oof_which'][sel].argmax(1) + 1
            vv['which_acc'] = float((g2 == y[sel]).mean())
            rep['ai_by_genre'][gname] = vv
    for batch, pref in (('v1', 'p'), ('v2', 'q'), ('v3', 'r')):
        sel = np.array([x['y'] > 0 and x.get('prompt', '').startswith(pref) for x in pool])
        if sel.any():
            rep['ai_by_genre']['[%s]' % batch] = verdicts(p_oof[sel])
    hg = collections.defaultdict(list)
    for x, p in zip(pool, p_oof):
        if x['y'] == 0:
            hg[(x['group'], 'oof')].append(p)
    for x, p in zip(others, p_oth):
        hg[(x['group'], x.get('half'))].append(p)
    for (g, half), ps in sorted(hg.items()):
        kind = {'old_chatgpt': 'AI never seen (ChatGPT gpt-3.5, 2023)',
                'gemini': 'AI never seen (Gemini)'}.get(g, 'human')
        rep['groups']['%s/%s' % (g, half)] = dict(kind=kind, **verdicts(ps))
    # AUC per human test group vs all AI out-of-fold
    rep['auc'] = {}
    for key, gv in rep['groups'].items():
        if gv['kind'] != 'human':
            continue
        g, half = key.split('/')
        hs = np.array(hg[(g, half)])
        rep['auc'][key] = float(auc(ai_p, hs))
    # length buckets for the human test halves + held-out
    rep['by_length'] = {}
    for lo, hi in ((50, 100), (100, 200), (200, 400), (400, 100000)):
        hp = [p for x, p in zip(others, p_oth) if x['y'] == 0 and x.get('half') in ('test', 'heldout')
              and lo <= R.cjk_count(x['text']) < hi]
        apx = [p for x, p in zip(pool, p_oof) if x['y'] > 0 and not x.get('old')
               and lo <= R.cjk_count(x['text']) < hi]
        rep['by_length']['%d-%d' % (lo, hi)] = dict(human=verdicts(hp) if hp else None,
                                                     ai=verdicts(apx) if apx else None)
    rep['p_oof'] = p_oof
    rep['p_oth'] = p_oth
    return rep


def auc(pos, neg):
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def summary_row(name, rep):
    hum = [v for k, v in rep['groups'].items() if v['kind'] == 'human' and not k.endswith('/calib')]
    worst = max(v['likely_ai'] for v in hum)
    tot = sum(v['n'] for v in hum)
    fpr = sum(v['likely_ai'] * v['n'] for v in hum) / tot
    og = rep['groups'].get('old_chatgpt/test', {})
    gm = rep['groups'].get('gemini/test', {})
    ce = rep['groups'].get('cedcc/test', {})
    return ('%-30s AI flagged %5.1f%% (->human %4.1f%%)  which %5.1f%%  human flagged %.2f%% (worst %.2f%%)'
            '  essays flag %.1f%% unsure %.0f%%  oldGPT flag %.0f%% human %.0f%%  gemini flag %.0f%% human %.0f%%'
            '  thr %.3f/%.4f' % (
                name, 100 * rep['ai_all']['likely_ai'], 100 * rep['ai_all']['likely_human'],
                100 * rep['which_acc'], 100 * fpr, 100 * worst, 100 * ce.get('likely_ai', 0),
                100 * ce.get('uncertain', 0), 100 * og.get('likely_ai', 0), 100 * og.get('likely_human', 0),
                100 * gm.get('likely_ai', 0), 100 * gm.get('likely_human', 0),
                rep['thresholds']['human_low'], rep['thresholds']['ai_high']))


def main():
    global MAX_FPR, HELDOUT_PER_TRAIN_SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.join(ROOT, '..', 'AI腔尺子_工作'))
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--seed', type=int, default=1115)
    ap.add_argument('--kind', default='char3+skel4')
    ap.add_argument('--lam', type=float, default=1e-4)
    ap.add_argument('--min-df', type=int, default=5)
    ap.add_argument('--kinds', default='char3,skel5,char3+skel4,char2+skel5')
    ap.add_argument('--lams', default='3e-5,1e-4,3e-4,1e-3')
    ap.add_argument('--small', action='store_true', help='tiny smoke test of the pipeline')
    ap.add_argument('--auto', action='store_true', help='try --kinds x --lams, export the best')
    ap.add_argument('--max-fpr', type=float, default=MAX_FPR,
                    help='largest share of any human group allowed over the 「很像 AI」 line (calib halves)')
    a = ap.parse_args()
    MAX_FPR = a.max_fpr
    if a.small:
        for k in TRAIN_SOURCES:
            TRAIN_SOURCES[k] = 60
        for k in EVAL_SOURCES:
            EVAL_SOURCES[k] = 60
        HELDOUT_PER_TRAIN_SOURCE = 30
    pool, heldout, evalset, old_gpt = prepare(a.work, a.seed)
    print('pool %d (AI %d), heldout %d, eval-only %d, old ChatGPT %d' % (
        len(pool), sum(1 for x in pool if x['y'] > 0), len(heldout), len(evalset), len(old_gpt)), flush=True)
    kinds = {'char3': dict(char=3), 'char2': dict(char=2), 'skel4': dict(skeleton=4), 'skel5': dict(skeleton=5),
             'char3+skel4': dict(char=3, skeleton=4), 'char2+skel5': dict(char=2, skeleton=5),
             'char2+skel4': dict(char=2, skeleton=4)}
    if a.grid:
        rows = []
        for kn in a.kinds.split(','):
            fz = make_featurizer(pool, heldout + evalset + old_gpt, kinds[kn])
            for lam in [float(v) for v in a.lams.split(',')]:
                res = run(pool, heldout, evalset, old_gpt, kinds[kn], lam, a.min_df, verbose=False, fz=fz)
                rep = evaluate(pool, res)
                line = summary_row('%s lam=%g (%.0fs)' % (kn, lam, res['seconds']), rep)
                print(line, flush=True)
                rows.append(line)
        io.open(os.path.join(a.work, 'grid_%s.txt' % time.strftime('%m%d_%H%M')), 'w', encoding='utf-8').write(
            '\n'.join(rows) + '\n')
        return
    if a.auto:
        # try the given settings, keep the one that flags the most AI texts (out-of-fold, unseen prompts) at the safe
        # threshold (ties: which-AI accuracy). Thresholds already keep every human calib group <= 1%, and the test
        # halves are only reported, never used to choose.
        best = None
        rows = []
        for kn in a.kinds.split(','):
            fz = make_featurizer(pool, heldout + evalset + old_gpt, kinds[kn])
            for lam in [float(v) for v in a.lams.split(',')]:
                res = run(pool, heldout, evalset, old_gpt, kinds[kn], lam, a.min_df, verbose=False, fz=fz)
                rep = evaluate(pool, res)
                line = summary_row('%s lam=%g (%.0fs)' % (kn, lam, res['seconds']), rep)
                print(line, flush=True)
                rows.append(line)
                key = (round(rep['ai_all']['likely_ai'], 3), rep['which_acc'])
                if best is None or key > best[0]:
                    best = (key, kn, lam, res, rep)
                else:
                    del res
            fz = None
        io.open(os.path.join(a.work, 'grid_%s.txt' % time.strftime('%m%d_%H%M')), 'w', encoding='utf-8').write(
            '\n'.join(rows) + '\n')
        _, a.kind, a.lam, res, rep = best
        print('chosen:', a.kind, a.lam, flush=True)
    else:
        res = run(pool, heldout, evalset, old_gpt, kinds[a.kind], a.lam, a.min_df)
        rep = evaluate(pool, res)
    print(summary_row('%s lam=%g' % (a.kind, a.lam), rep))
    export(a, pool, res, rep)


# ------------------------------------------------------------------------------------------------ export
def export(a, pool, res, rep):
    fz = res['fz']
    W, b = res['W'], res['b']
    model = dict(format='aiqiang-model', version=1, created=time.strftime('%Y-%m-%d'), classes=CLASSES,
                 feature_sets=[], bias=[round(float(v), 6) for v in b],
                 ai_calibration=dict(a=round(rep['calibration']['a'], 6), b=round(rep['calibration']['b'], 6)),
                 thresholds=dict(min_chars=50, ai_high=round(rep['thresholds']['ai_high'], 4),
                                 human_low=round(rep['thresholds']['human_low'], 4), which_min=0.5),
                 eval=dict(summary_lines=summary_lines(rep)),
                 long_text=dict(R.LONG_TEXT),
                 training=dict(kind=a.kind, lam=a.lam, min_df=a.min_df, seed=a.seed, max_fpr=MAX_FPR,
                               pool=len(pool), ai=int(sum(1 for x in pool if x['y'] > 0))))
    off = 0
    for fs, grams, cols, idf in zip(res['sets'], fz.grams, fz.cols, fz.idf):
        vocab = {}
        for j, (c, v) in enumerate(zip(cols, idf)):
            row = W[off + j]
            vocab[grams[c]] = [round(float(v), 5)] + [round(float(x), 5) for x in row]
        off += len(cols)
        d = dict(fs)
        d['vocab'] = vocab
        model['feature_sets'].append(d)
    path = os.path.join(ROOT, 'aiqiang', 'model.json')
    io.open(path, 'w', encoding='utf-8').write(json.dumps(model, ensure_ascii=False, separators=(',', ':')))
    print('model.json %.1f MB' % (os.path.getsize(path) / 1e6))
    # rounded weights: check the exported model against the in-memory scores on a few pool texts
    m2 = json.load(io.open(path, encoding='utf-8'))
    diffs = []
    sample = [x['text'] for x in pool[:40]]
    rows = np.arange(40)
    s_mem, _, _ = scores(fz.transform(rows), W, b)
    for t, sm in zip(sample, s_mem):
        diffs.append(abs(R.predict(m2, t)['ai_score'] - sm))
    print('exported vs in-memory ai_score max diff %.2e' % max(diffs))
    # results for the report (no human text in it)
    out = {k: v for k, v in rep.items() if k not in ('p_oof', 'p_oth')}
    out['training'] = model['training']
    out['created'] = time.strftime('%Y-%m-%d %H:%M')
    os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
    io.open(os.path.join(ROOT, 'reports', 'eval_results.json'), 'w', encoding='utf-8').write(
        json.dumps(out, ensure_ascii=False, indent=1))
    # per-text out-of-fold results for the AI samples (ours, may be shown) -> for checking misses
    ai_rows = []
    for x, p, wh in zip(pool, rep['p_oof'], res['oof_which']):
        if x['y'] > 0:
            ai_rows.append(dict(id=x['id'], model=CLASSES[x['y']], prompt=x['prompt'], ai_prob=round(float(p), 4),
                                which={CLASSES[c + 1]: round(float(wh[c]), 3) for c in range(3)}))
    io.open(os.path.join(ROOT, 'reports', 'ai_samples_oof.json'), 'w', encoding='utf-8').write(
        json.dumps(ai_rows, ensure_ascii=False, indent=1))
    # human misses go to the WORK folder only (they quote human texts)
    miss = []
    for x, p in zip(pool, rep['p_oof']):
        if x['y'] == 0 and p >= rep['thresholds']['ai_high'] * 0.9:
            miss.append(dict(group=x['group'], half='oof', ai_prob=round(float(p), 4), text=x['text'][:300]))
    for x, p in zip(res['others'], rep['p_oth']):
        if x['y'] == 0 and p >= rep['thresholds']['ai_high'] * 0.9:
            miss.append(dict(group=x['group'], half=x.get('half'), ai_prob=round(float(p), 4), text=x['text'][:300]))
    miss.sort(key=lambda r: -r['ai_prob'])
    io.open(os.path.join(a.work, 'human_high_scores.json'), 'w', encoding='utf-8').write(
        json.dumps(miss, ensure_ascii=False, indent=1))
    print('human texts near/over the line: %d (in work folder)' % len(miss))
    # every score (no text) for threshold studies, in the work folder
    sc = [dict(kind='ai', group=x['group'], half='oof', p=float(p), n=R.cjk_count(x['text']))
          for x, p in zip(pool, rep['p_oof']) if x['y'] > 0]
    sc += [dict(kind='human', group=x['group'], half='oof', p=float(p), n=R.cjk_count(x['text']))
           for x, p in zip(pool, rep['p_oof']) if x['y'] == 0]
    sc += [dict(kind='human' if x['y'] == 0 else 'ai_unseen', group=x['group'], half=x.get('half'), p=float(p),
                n=R.cjk_count(x['text'])) for x, p in zip(res['others'], rep['p_oth'])]
    io.open(os.path.join(a.work, 'scores_latest.json'), 'w', encoding='utf-8').write(json.dumps(sc))


GROUP_NAMES = {'clue': '社区问答、点评、新闻', 'hc3_open_qa': '社区问答', 'hc3_finance': '理财问答',
               'hc3_psychology': '心理问答', 'cmrc': '维基百科段落', 'pd_other': '民国散文小说',
               'pd_famous': '民国课本名篇', 'cedcc': '中学生作文', 'weibo': '微博', 'hc3_baike': '百科问答',
               'hc3_nlpcc_dbqa': '问答', 'hc3_medicine': '医疗问答', 'hc3_law': '法律问答'}


def summary_lines(rep):
    """the lines the program shows under 「关于」 (Chinese, plain numbers)"""
    g = rep['groups']
    hum = {k: v for k, v in g.items() if v['kind'] == 'human' and not k.endswith('/calib')}
    tot = sum(v['n'] for v in hum.values())
    flagged = sum(round(v['likely_ai'] * v['n']) for v in hum.values())
    worst_k = max(hum, key=lambda k: hum[k]['likely_ai'])
    ced = [v for k, v in hum.items() if k.startswith('cedcc/')]
    ai = load_ai(cut=False)
    per = collections.Counter(x['y'] for x in ai)
    prompts = len(set(x['prompt'] for x in ai))
    lines = [
        '测试用的人类文字：%d 段，都写于 2022 年以前（问答、点评、百科、微博、中学生作文、民国名篇等）。' % tot,
        '其中被判成“很像 AI 写的”：%d 段（%.2f%%）；最高的一类（%s）是 %.2f%%。' % (
            flagged, 100.0 * flagged / tot, GROUP_NAMES.get(worst_k.split('/')[0], worst_k.split('/')[0]),
            100 * hum[worst_k]['likely_ai']),
    ]
    if ced:
        lines.append('中学生作文（2021 年，从没拿来训练）：%.2f%% 被判成“很像 AI”，%.0f%% 判成“不确定”。' % (
            100 * ced[0]['likely_ai'], 100 * ced[0]['uncertain']))
    lines += [
        'AI 文字（训练时没见过的题目）：%.0f%% 判成“很像 AI”，%.0f%% “不确定”，%.0f%% 被当成“更像人写的”。' % (
            100 * rep['ai_all']['likely_ai'], 100 * rep['ai_all']['uncertain'], 100 * rep['ai_all']['likely_human']),
        '猜“更像哪个 AI”（DeepSeek / GPT / Claude 三选一）：猜对 %.0f%%。' % (100 * rep['which_acc']),
    ]
    gm = g.get('gemini/test')
    if gm:
        lines.append('从没训练过的 Gemini（%d 段）：%.0f%% 判成“很像 AI”，%.0f%% 被当成“更像人写的”。' % (
            gm['n'], 100 * gm['likely_ai'], 100 * gm['likely_human']))
    og = g.get('old_chatgpt/test')
    if og:
        hit = round(og['likely_ai'] * og['n'])
        lines.append('2023 年初的旧版 ChatGPT（这几类问题从没训练过）：%d 段里%s判成“很像 AI”，'
                     '%.0f%% 反被当成“更像人写的”——模型换代后它会失灵。' % (
                         og['n'], '一段也没' if hit == 0 else '只有 %d 段' % hit, 100 * og['likely_human']))
    lines.append('AI 样本：三家各约 %d 篇、%d 道题；平实的百科、科普、新闻类 AI 文字基本认不出。' % (
        round(sum(per.values()) / 3.0), prompts))
    return lines


if __name__ == '__main__':
    main()
