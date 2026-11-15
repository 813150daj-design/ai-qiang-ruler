"""Golden values for the real model: tests/golden_real.json (read by tests/test_real_model.py).
Texts: some of our AI samples, the sample text, a few passages written for this file, and edge cases.
No human dataset text goes in here (those never enter the repo).

  python make_golden.py
"""
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ref_features as R  # noqa: E402

OWN = [
    '今天下班的时候雨下得很大，我在地铁口站了十几分钟，看着卖伞的大叔一会儿就卖空了一筐。后来雨小了点，'
    '我就顶着包跑回家，鞋子全湿透了。晚饭随便煮了包面，加了个蛋，吃完窝在沙发上刷了会儿手机就困了。',
    '这个问题其实没有标准答案。我自己的经验是先把最难的那件事放在上午做，下午人容易犯困，适合处理杂事。'
    '另外手机别放在桌上，放远一点，真的有用。坚持两周以后，你会发现拖延没那么可怕。',
    '首先，我们要明白，学习不仅是获取知识的过程，更是塑造人格的旅程。其次，在这个信息爆炸的时代，'
    '我们需要学会筛选。总之，只有坚持终身学习，才能在不断变化的世界中找到属于自己的位置。',
    '外婆家的院子里有一棵老枣树，每年秋天都结满了枣子。小时候我总是爬上去摘，外婆就站在树下喊：慢点慢点，'
    '别摔着！现在外婆不在了，那棵树也被砍了，只有我还记得枣子的甜味。',
]
EDGE = ['', '   ', 'Hello world, this is English only.', '短短一句。', '😀😀 今天开心！！[哈哈] @小明 #周末# http://a.cn',
        '“引号”「引号」……——2024年3月，OK。\r\n第二行，第二行，第二行，第二行。']


def main():
    model = json.load(io.open(os.path.join(ROOT, 'aiqiang', 'model.json'), encoding='utf-8'))
    texts = []
    for m in ('deepseek', 'gpt', 'claude'):
        fs = sorted(glob.glob(os.path.join(ROOT, 'data', 'ai', m, 'p*_s*.txt')))
        for f in fs[::12][:6]:
            texts.append(('%s/%s' % (m, os.path.basename(f)), io.open(f, encoding='utf-8').read()))
    texts.append(('sample', io.open(os.path.join(ROOT, 'aiqiang', 'sample.txt'), encoding='utf-8').read()))
    # long texts (scored piece by piece): our own long AI answers, and one of them doubled with odd spacing
    longs = [f for d in ('deepseek', 'gpt', 'claude')
             for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'ai', d, 'q*_s*.txt')))
             if len(io.open(f, encoding='utf-8').read()) > 900][:4]
    for f in longs:
        texts.append(('long/' + os.path.basename(f), io.open(f, encoding='utf-8').read()))
    if longs:
        t = io.open(longs[0], encoding='utf-8').read()
        texts.append(('long/doubled', t + '\n\n  \r\n' + t.replace('。', '。 ')))
    texts += [('own_%d' % i, t) for i, t in enumerate(OWN)]
    texts += [('edge_%d' % i, t) for i, t in enumerate(EDGE)]
    cases = []
    for k, (name, t) in enumerate(texts):
        case = dict(name=name, text=t, normalized=R.normalize(t), distorted=R.distort(R.normalize(t)),
                    chars=R.cjk_count(t), markers=R.markers(t))
        if k < 3:
            case['features'] = [R.featurize(t, fs) for fs in model['feature_sets']]
        case.update(R.predict(model, t))
        case['pieces'] = [list(s) for s in R.split_pieces(t, model.get('long_text'))]
        case['doc'] = R.predict_doc(model, t)
        cases.append(case)
    out = dict(format='aiqiang-golden', version=1, model_created=model.get('created'), cases=cases)
    p = os.path.join(ROOT, 'tests', 'golden_real.json')
    io.open(p, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print('golden_real.json: %d cases' % len(cases))


if __name__ == '__main__':
    main()
