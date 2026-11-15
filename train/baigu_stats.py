"""How often each of the 10 「AI 八股」 checks fires, per group of texts (AI models and human sources).
Light: regexes only. Writes ../reports/baigu_stats.json (numbers only, no text).

  python baigu_stats.py [--work <work folder>]
"""
import argparse
import collections
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ref_features as R  # noqa: E402
from train import load_ai, CLASSES  # noqa: E402

NAMES = [n for n, _ in R.CHECKS]


def stats(texts):
    n = len(texts)
    fired = collections.Counter()
    idx = []
    for t in texts:
        m = R.markers(t)
        idx.append(sum(1 for v in m.values() if v))
        for k, v in m.items():
            if v:
                fired[k] += 1
    idx.sort()
    return dict(n=n, mean_index=round(sum(idx) / n, 2), median_index=idx[n // 2],
                share_index_ge3=round(sum(1 for v in idx if v >= 3) / n, 3),
                fired={k: round(fired[k] / n, 3) for k in NAMES})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.join(ROOT, '..', 'AI腔尺子_工作'))
    a = ap.parse_args()
    groups = collections.OrderedDict()
    ai = load_ai()
    for c in (1, 2, 3):
        groups[CLASSES[c]] = [x['text'] for x in ai if x['y'] == c]
    by = collections.defaultdict(list)
    for line in io.open(os.path.join(a.work, 'human', 'human.jsonl'), encoding='utf-8'):
        x = json.loads(line)
        key = x['source']
        if key == 'cedcc':
            key = 'cedcc_whole' if ':whole_' in x['doc'] else 'cedcc_part'
        by[key].append(x['text'])
    rng = random.Random(7)
    for k in sorted(by):
        xs = by[k]
        rng.shuffle(xs)
        groups[k] = xs[:2000]
    out = collections.OrderedDict((k, stats(v)) for k, v in groups.items())
    print('%-16s %5s  %5s  %5s  ' % ('group', 'n', 'mean', '>=3') + ' '.join('%-6s' % k[:4] for k in NAMES))
    for k, s in out.items():
        print('%-16s %5d  %5.2f  %4.0f%%  ' % (k, s['n'], s['mean_index'], 100 * s['share_index_ge3']) +
              ' '.join('%5.0f%%' % (100 * s['fired'][n]) for n in NAMES))
    os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
    io.open(os.path.join(ROOT, 'reports', 'baigu_stats.json'), 'w', encoding='utf-8').write(
        json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
