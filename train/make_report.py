"""Write ../reports/eval_report.md (Chinese) from ../reports/eval_results.json and ../reports/baigu_stats.json.
Numbers only; no human text is quoted.

  python make_report.py
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NAMES = {
    'clue': 'CLUECorpusSmall（社区问答、点评、新闻）',
    'hc3_open_qa': 'HC3 社区问答',
    'hc3_finance': 'HC3 理财问答',
    'hc3_psychology': 'HC3 心理问答',
    'cmrc': '维基百科段落（CMRC 2018）',
    'pd_other': '民国散文小说（鲁迅、朱自清）',
    'pd_famous': '民国课本名篇（《背影》《故乡》等）',
    'cedcc': '中学生作文（2021 年）',
    'weibo': '微博（约 2012 年）',
    'hc3_baike': 'HC3 百科问答',
    'hc3_nlpcc_dbqa': 'HC3 问答（NLPCC-DBQA）',
    'hc3_medicine': 'HC3 医疗问答',
    'hc3_law': 'HC3 法律问答',
    'old_chatgpt': '旧版 ChatGPT 回答（2023 年初，从没见过）',
}
HALF = {'oof': '交叉验证', 'heldout': '留出的文档', 'test': '只测的一半', 'calib': '定门槛的一半'}
TRAINED = {'clue', 'hc3_open_qa', 'hc3_finance', 'hc3_psychology', 'cmrc', 'pd_other'}


def pct(v, d=1):
    return ('%.' + str(d) + 'f%%') % (100 * v)


def main():
    r = json.load(io.open(os.path.join(ROOT, 'reports', 'eval_results.json'), encoding='utf-8'))
    bp = os.path.join(ROOT, 'reports', 'baigu_stats.json')
    bg = json.load(io.open(bp, encoding='utf-8')) if os.path.exists(bp) else None
    th = r['thresholds']
    L = []
    w = L.append
    w('# AI腔尺子 测试报告')
    w('')
    w('生成时间：%s。模型设置：%s，λ=%s，min_df=%s；训练池 %d 段（其中 AI %d 段）。' % (
        r['created'], r['training']['kind'], r['training']['lam'], r['training']['min_df'],
        r['training']['pool'], r['training']['ai']))
    w('')
    w('判定门槛：AI 可能性 ≥ %.4f 判「很像 AI 写的」；≤ %.4f 判「更像人写的」；中间是「不确定」。' % (
        th['ai_high'], th['human_low']))
    w('')
    w('## 一、人类文字（都写于 2022 年以前）')
    w('')
    w('「很像 AI」那一栏就是误判率。没参加训练的类别（标 ✱）最能说明它在陌生文字上的表现。')
    w('')
    w('| 类别 | 怎么测的 | 段数 | 判成“很像 AI” | 不确定 | 更像人写的 |')
    w('|---|---|---:|---:|---:|---:|')
    rows = []
    for k, v in r['groups'].items():
        if v['kind'] != 'human' or k.endswith('/calib'):
            continue
        g, half = k.split('/')
        rows.append((g in TRAINED, g, half, v))
    rows.sort(key=lambda t: (t[0], t[1], t[2]))
    tot = n_ai = 0
    for trained, g, half, v in rows:
        mark = '' if trained else ' ✱'
        w('| %s%s | %s | %d | %s | %s | %s |' % (NAMES.get(g, g), mark, HALF.get(half, half), v['n'],
                                                pct(v['likely_ai'], 2), pct(v['uncertain']),
                                                pct(v['likely_human'])))
        tot += v['n']
        n_ai += v['likely_ai'] * v['n']
    w('| **合计** | | %d | **%s** | | |' % (tot, pct(n_ai / tot, 2)))
    w('')
    w('## 二、AI 文字（每段都由没见过这道题的模型打分）')
    w('')
    w('| 来源 | 段数 | 判成“很像 AI” | 不确定 | 被当成“更像人写的” | 猜对是哪家 |')
    w('|---|---:|---:|---:|---:|---:|')
    for name in ('DeepSeek', 'GPT', 'Claude'):
        v = r['groups'][name]
        w('| %s | %d | %s | %s | %s | %s |' % (name, v['n'], pct(v['likely_ai']), pct(v['uncertain']),
                                              pct(v['likely_human']), pct(v['which_acc'])))
    a = r['ai_all']
    w('| **合计** | %d | **%s** | %s | %s | **%s** |' % (a['n'], pct(a['likely_ai']), pct(a['uncertain']),
                                                       pct(a['likely_human']), pct(r['which_acc'])))
    ld = r.get('ai_long_docs')
    if ld:
        w('| 其中 500 字以上的长文（按整篇算：分段打分再平均） | %d | %s | %s | %s | — |' % (
            ld['n'], pct(ld['likely_ai']), pct(ld['uncertain']), pct(ld['likely_human'])))
    for key, label in (('old_chatgpt_train/oof', '旧版 ChatGPT（2023 年初；训练里有别的问题的回答，按问题交叉验证）'),
                       ('old_chatgpt/test', '旧版 ChatGPT（2023 年初；百科、医疗、问答、法律，从没训练过）'),
                       ('gemini/test', 'Gemini 3.1 Pro（从没训练过的 AI）')):
        og = r['groups'].get(key)
        if og:
            w('| %s | %d | %s | %s | %s | — |' % (label, og['n'], pct(og['likely_ai']), pct(og['uncertain']),
                                                 pct(og['likely_human'])))
    w('')
    if r.get('ai_by_genre'):
        w('按题目体裁分：')
        w('')
        w('| 体裁 | 段数 | 判成“很像 AI” | 不确定 | 被当成“更像人写的” |')
        w('|---|---:|---:|---:|---:|')
        for gname, v in r['ai_by_genre'].items():
            label = {'[v1]': '第一批（日常题目）', '[v2]': '第二批（问答、点评、百科、新闻）',
                     '[v3]': '第三批（百科科普、新闻通知、点评、书信应用文、口语）'}.get(gname, gname)
            w('| %s | %d | %s | %s | %s |' % (label, v['n'], pct(v['likely_ai']), pct(v['uncertain']),
                                           pct(v['likely_human'])))
        w('')
    w('「猜是哪家」的混淆表（行 = 真实，列 = 猜测）：')
    w('')
    w('| | 猜 DeepSeek | 猜 GPT | 猜 Claude |')
    w('|---|---:|---:|---:|')
    for name, row in zip(('DeepSeek', 'GPT', 'Claude'), r['which_confusion']):
        w('| %s | %d | %d | %d |' % (name, row[0], row[1], row[2]))
    w('')
    w('## 三、按长度（人类：只测的一半和留出的文档；AI：交叉验证）')
    w('')
    w('| 汉字数 | 人类段数 | 人类误判成“很像 AI” | AI 段数 | AI 判成“很像 AI” |')
    w('|---|---:|---:|---:|---:|')
    for k, v in r['by_length'].items():
        h, x = v.get('human'), v.get('ai')
        w('| %s | %s | %s | %s | %s |' % (k.replace('-100000', ' 以上'), h['n'] if h else 0,
                                        pct(h['likely_ai'], 2) if h else '—', x['n'] if x else 0,
                                        pct(x['likely_ai']) if x else '—'))
    w('')
    w('## 四、分得开吗（AUC，1 = 完全分开，0.5 = 瞎猜）')
    w('')
    w('每一类人类文字和全部 AI 文字之间的 AUC：')
    w('')
    for k, v in sorted(r['auc'].items()):
        if not k.endswith('/calib'):
            g, half = k.split('/')
            w('- %s（%s）：%.3f' % (NAMES.get(g, g), HALF.get(half, half), v))
    w('')
    if bg:
        w('## 五、「AI 八股」十条检查各类文字踩中的比例')
        w('')
        w('八股分 = 踩中几条。AI 和作文片段约 250 字，作文整篇约 600 字；字越多踩中的机会越多，'
          '所以只宜和长度相近的文字比。中学生作文的整篇分数和 DeepSeek 的片段不相上下：这十条量的是文风，不是作者。')
        w('')
        w('| 类别 | 段数 | 平均八股分 | 踩中 3 条以上 |')
        w('|---|---:|---:|---:|')
        for k, v in bg.items():
            label = {'cedcc_whole': '中学生作文（整篇）', 'cedcc_part': '中学生作文（片段）'}.get(k, NAMES.get(k, k))
            w('| %s | %d | %.2f | %s |' % (label, v['n'], v['mean_index'], pct(v['share_index_ge3'], 0)))
        w('')
    w('## 测试方法')
    w('')
    w('- AI 文字按题目分 6 折交叉验证：打分的模型没见过这道题的任何一篇。')
    w('- 能训练的人类数据按文档分 6 折，另留出一部分文档完全不参与训练。')
    mf = r['training'].get('max_fpr', 0.01)
    w('- 只测不训的人类数据按文档对半分：一半用来定「很像 AI」的门槛（定门槛时每一类误判都不超过 %s），'
      '另一半只用来报告。' % pct(mf, 1))
    w('- AI 可能性用 Platt 方法校准：AI 与人各占一半，人这一半里每一类权重相同。')
    w('- 结果只代表这些题目、这些长度、这几个模型版本。')
    io.open(os.path.join(ROOT, 'reports', 'eval_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('reports/eval_report.md written')


if __name__ == '__main__':
    main()
