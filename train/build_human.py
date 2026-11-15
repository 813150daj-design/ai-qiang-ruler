"""Build the human-text set from the raw downloads (see fetch_human.sh). Human texts are NEVER put in the repo:
the output goes to the work folder given by --work (default ../../AI腔尺子_工作), only statistics leave it.

Every item: id, source, license, use ('train' = may be used to fit the model; 'eval' = only measured), genre, text.
Long texts are cut into 150-400 character pieces at sentence ends, like the AI texts (most are 150-350).

  python build_human.py            -> <work>/human/human.jsonl and a count table
"""
import argparse
import csv
import glob
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ref_features import cjk_count  # noqa: E402

SOURCES = {
    # name: (licence, use, genre)
    'clue': ('MIT (CLUECorpusSmall, from brightmart/nlp_chinese_corpus)', 'train', '问答/评论/新闻 混合'),
    'hc3_open_qa': ('MIT (WebTextQA & BaikeQA via HC3-Chinese)', 'train', '问答'),
    'hc3_finance': ('CC BY 4.0 (FinanceZhidao via HC3-Chinese)', 'train', '问答·理财'),
    'hc3_psychology': ('CC0 (via HC3-Chinese)', 'train', '问答·心理'),
    'cmrc': ('CC BY-SA 4.0 (CMRC 2018, Chinese Wikipedia)', 'train', '百科说明'),
    'cedcc': ('未声明 (CEDCC, EMNLP 2023) - 只测不训', 'eval', '中学生作文 (2021)'),
    'weibo': ('未声明 (weibo_senti_100k) - 只测不训', 'eval', '微博 (约 2012)'),
    'hc3_baike': ('未声明 (Baidu Baike via HC3) - 只测不训', 'eval', '百科问答'),
    'hc3_nlpcc_dbqa': ('未声明 (NLPCC-DBQA via HC3) - 只测不训', 'eval', '问答'),
    'hc3_medicine': ('CC BY-NC 4.0 (via HC3) - 只测不训', 'eval', '问答·医疗'),
    'hc3_law': ('未声明 (LegalQA via HC3) - 只测不训', 'eval', '问答·法律'),
    'pd_famous': ('公有领域 (维基文库) - 课本名篇，只测不训', 'eval', '民国散文/小说 (课本名篇)'),
    'pd_other': ('公有领域 (维基文库: 鲁迅、朱自清)', 'train', '民国散文/小说'),
}
# school-textbook pieces people are likely to paste in: kept out of training so the test on them is fair
PD_FAMOUS = {'背影', '荷塘月色', '匆匆', '春 (朱自清)', '落花生', '從百草園到三味書屋', '藤野先生', '阿長與《山海經》',
             '孔乙己', '故鄉', '社戲', '祝福', '一件小事', '秋夜 (魯迅)', '雪 (鲁迅)', '風波', '藥', '狂人日記'}

_SENT = re.compile(r'(?<=[。！？!?；;…])|\n+')


def chunks(text, rng, lo=150, hi=400):
    """cut text into pieces of lo..hi characters at sentence ends (a piece may run past hi if one sentence is long)"""
    parts = [p for p in _SENT.split(text) if p and p.strip()]
    out, cur, target = [], '', rng.randint(lo, hi)
    for p in parts:
        cur += p
        if len(cur) >= target:
            out.append(cur.strip())
            cur, target = '', rng.randint(lo, hi)
    if cur.strip() and (len(cur) >= lo or not out):
        out.append(cur.strip())
    return out


_WB_RT = re.compile(r'//@.*$')
_WB_REPLY = re.compile(r'^回复@[^:：]{1,30}[:：]')


def clean_weibo(t):
    t = t.replace('﻿', '')
    t = _WB_RT.sub('', t)
    t = _WB_REPLY.sub('', t)
    return t.strip()


def ok(t, min_cjk=50):
    return cjk_count(t) >= min_cjk and cjk_count(t) >= 0.5 * len(re.sub(r'\s', '', t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.join(HERE, '..', '..', 'AI腔尺子_工作'))
    ap.add_argument('--seed', type=int, default=1114)
    a = ap.parse_args()
    raw = os.path.join(a.work, 'human_raw')
    rng = random.Random(a.seed)
    items = []

    def add(source, doc, text, max_pieces=None, whole=False):
        pieces = [text] if whole else chunks(text, rng)
        pieces = [p for p in pieces if ok(p)]
        if max_pieces:
            pieces = pieces[:max_pieces]
        for k, p in enumerate(pieces):
            lic, use, genre = SOURCES[source]
            items.append(dict(id='%s:%s:%d' % (source, doc, k), doc='%s:%s' % (source, doc), source=source,
                              license=lic, use=use, genre=genre, text=p))

    # CLUECorpusSmall slices: one document per line (the file is shuffled: Q&A, reviews, news, wiki mixed)
    for f in sorted(glob.glob(os.path.join(raw, 'clue', 'off_*.txt'))):
        b = io.open(f, 'rb').read().decode('utf-8', 'ignore').split('\n')[1:-1]   # first/last lines are cut
        for i, line in enumerate(b):
            line = line.strip()
            if 60 <= len(line) <= 3000:
                add('clue', '%s_%d' % (os.path.basename(f)[4:6], i), line, max_pieces=2)
    # HC3-Chinese human answers
    for split in ('open_qa', 'finance', 'psychology', 'baike', 'nlpcc_dbqa', 'medicine', 'law'):
        p = os.path.join(raw, 'hc3', split + '.jsonl')
        for i, line in enumerate(io.open(p, encoding='utf-8')):
            d = json.loads(line)
            for j, ans in enumerate(d.get('human_answers') or []):
                if ans and ans.strip():
                    add('hc3_' + split, '%d_%d' % (i, j), ans.strip(), max_pieces=2)
    # CMRC 2018 contexts (Chinese Wikipedia), de-duplicated
    seen = set()
    for f in sorted(glob.glob(os.path.join(raw, 'cmrc', 'rows_*.json'))):
        try:
            rows = json.load(io.open(f, encoding='utf-8')).get('rows', [])
        except ValueError:
            continue
        for r in rows:
            c = (r.get('row') or {}).get('context', '')
            if c and c not in seen:
                seen.add(c)
                add('cmrc', str(len(seen)), c, max_pieces=2)
    # CEDCC student essays: whole essays (as a reader would paste them) and pieces
    for d in json.load(io.open(os.path.join(raw, 'cedcc', 'CEDCC.json'), encoding='utf-8')):
        text = '\n'.join(p.strip() for p in d['text'] if p.strip())
        add('cedcc', 'whole_' + str(d['id']), text, whole=True)
        add('cedcc', 'part_' + str(d['id']), text)
    # Weibo posts (>= 50 CJK characters after removing retweet chains)
    with io.open(os.path.join(raw, 'weibo', 'weibo_senti_100k.csv'), encoding='utf-8') as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            t = clean_weibo(row.get('review') or '')
            add('weibo', str(i), t, whole=True)

    # public-domain essays and stories from Wikisource (fetch_pd_essays.py): whole pieces up to 1500 chars, and pieces
    for f in sorted(glob.glob(os.path.join(raw, 'pd', '*.txt'))):
        t = io.open(f, encoding='utf-8').read().strip()
        name = os.path.basename(f)[:-4]
        src = 'pd_famous' if name.replace('_', ' ') in PD_FAMOUS else 'pd_other'
        if len(t) <= 1500:
            add(src, 'whole_' + name, t, whole=True)
        add(src, 'part_' + name, t)

    out_dir = os.path.join(a.work, 'human')
    os.makedirs(out_dir, exist_ok=True)
    with io.open(os.path.join(out_dir, 'human.jsonl'), 'w', encoding='utf-8') as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    by = {}
    for x in items:
        by.setdefault(x['source'], []).append(cjk_count(x['text']))
    for s, L in sorted(by.items()):
        L.sort()
        print('%-16s %6d pieces  median %3d CJK  use=%s' % (s, len(L), L[len(L) // 2], SOURCES[s][1]))
    print('total', len(items))


if __name__ == '__main__':
    main()
