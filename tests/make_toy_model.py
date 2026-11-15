"""Build deterministic test fixtures using ONLY the training reference maths.

Run: python tests/make_toy_model.py
This is a numerical test model, not an evaluated detector. It is never installed
as aiqiang/model.json. The reference is loaded without writing into train/.
"""

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'


def load_reference():
    spec = importlib.util.spec_from_file_location('aiqiang_training_reference', ROOT / 'train' / 'ref_features.py')
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def fixture_texts():
    texts = []
    for name in ('claude', 'deepseek', 'gpt'):
        for path in sorted((ROOT / 'data' / 'ai' / name).glob('*.txt'))[:4]:
            texts.append((str(path.relative_to(ROOT)).replace('\\', '/'), path.read_text(encoding='utf-8-sig')))
    texts.append(('sample', (ROOT / 'aiqiang' / 'sample.txt').read_text(encoding='utf-8')))
    texts.extend([
        ('empty', ''),
        ('spaces', ' \t\n\r\n\u3000 '),
        ('english', 'Hello WORLD! This is a test, 12345.67 and 99.'),
        ('emoji', '😀🌱✨原来，猫咪也会打哈欠。'),
        ('social', '看 https://example.org/a?x=1 @张三-abc [哈哈] #天气# www.example.com 好。'),
        ('fullwidth', 'ＡＢＣ１２３，原来！不是迷路，而是散步；不只是今天，更是以后。'),
        ('quotes', '“生活”「原来」『如此』〝你好〞〟以及‘单引号’和\'ASCII\"。'),
        ('ellipses', '…… ... 。。 ⋯⋯ ··· … —— -- –– ― ‒'),
        ('numbers', '2026年11月15日，123,456.78；99.9%和１２３４以及0。'),
        ('long6000', '春风走过小城的街道我们今天又去看花日子平静' * 300),
        ('long_crlf', ' \r\n' + ('原来，今天路边的花开了。' * 30 + '\r\n') * 3 + ' \r\n'),
        ('long_blank_lines', '\n\n' + ('这是一段普通的生活记录。' * 30 + '\n \t\n') * 3 + '\t '),
        ('long_no_sentence_end', '我们今天又去街边看花' * 60),
        ('long_short_remainder', ('春' * 149 + '。') * 4 + '尾' * 40),
        ('long_sentence_marks', ('甲' * 149 + '！？!?；;…') * 4),
        ('long_low_cjk_pieces', 'a' * 299 + '。' + '中' * 299 + '。' + 'b' * 299 + '。' + '文' * 299 + '。'),
        ('long_all_low_cjk', ('a' * 299 + '。') * 3),
        ('crlf', '  首先，\r\n它让脚步变轻，\r\n它让心跳放慢，\r\n它让烦恼远去。\r\n温柔的人生！  \r\n'),
        ('parallel4', '它来了，它笑了，它坐下，它走了。'),
        ('parallel5', ' 我看花， 我看云， 我看雨， 我看海， 我看山。'),
        ('parallel_two_runs', '他走，他跑，他停，你来，你笑，你哭。'),
        ('parallel_empty_parts', '我来，，我看、；我想,我走。'),
        ('parallel_break', '他来，他走。它飞，它跑，它跳！'),
        ('newline_merge', ' 他\n来，他去，他回。人生\n温柔。 \n'),
        ('carriage_only', '它来\r，它走\r，它停\r。\r生活。\r'),
        ('uplift_trim', '\n上一句。 \t 人间值得。  \r\n'),
        ('uplift_not_last', '生活真美好。今天吃了面。'),
        ('outline', '# 标题\n- 好问题\n* 以下是\n1. 希望对你有帮助\n**加粗**'),
        ('markdown_rules', '前文\n---\n中间\n***\n后文\n___\n结尾'),
        ('markdown_spaced_rules', '\t- - - \n正文\n  *\t* * *\t\n结尾\n_ _ _'),
        ('markdown_nfkc_rules', '前文\n－－－\n＃ 标题\n＊＊＊\n＿＿＿\n后文'),
        ('markdown_mixed_rules', '---\n# 标题\n* * *\n- 正文\n___\n**结尾**\n正文---结尾'),
        ('all_markers', '首先，不是远方而是此刻。不仅是宇宙，更是灵魂。原来——'),
        ('negation_comma', '不是相遇，是重逢！不是离开，而是回家。与其说选择不如说勇气。'),
        ('epiphany', '才发现才明白才懂得才知道忽然就觉得突然发现蓦然明白这才意识到终于懂了。'),
        ('cjk_extension', '㐀㐁㐂䶿一鿿龘，𠀀并不在参考的计数范围。'),
        ('punctuation_only', '。！？!?，、；,\r\n——**'),
        ('rare_symbols', 'éé ΩΩ ①②Ⅲ ㍿ \u200b猫\u200d咪'),
        ('short_human', '哈哈，饭糊了，锅还没洗。'),
        ('threshold50', '中' * 50),
        ('low_cjk', '中文' + 'english and spaces 123 ' * 20),
    ])
    # Exactly 6000 Python characters, independently of phrase length.
    texts = [(name, (text + '文' * 6000)[:6000] if name == 'long6000' else text) for name, text in texts]
    return texts


def build():
    ref = load_reference()
    texts = fixture_texts()
    randomizer = random.Random(20261114)
    feature_sets = []
    for name, source, ns, limit in [('char', 'plain', [1, 2, 3], 450), ('skeleton', 'distorted', [2, 3, 4], 350)]:
        fs = dict(name=name, source=source, n=ns)
        if source == 'distorted':
            fs.update(keep=ref.KEEP, punct=''.join(sorted(ref.PUNCT)), mask=ref.MASK)
        document_frequency = Counter()
        for _, text in texts:
            document_frequency.update(ref.ngram_counts(ref.source_string(text, fs), ns).keys())
        grams = sorted(document_frequency, key=lambda gram: (-document_frequency[gram], gram))[:limit]
        vocab = {}
        for gram in grams:
            idf = 1.0 + ref.math.log((1 + len(texts)) / (1 + document_frequency[gram]))
            weights = [round(randomizer.uniform(-0.7, 0.7), 8) for _ in range(4)]
            vocab[gram] = [idf] + weights
        fs['vocab'] = vocab
        feature_sets.append(fs)
    model = dict(
        format='aiqiang-model', version=1, created='2026-11-14',
        classes=['人', 'DeepSeek', 'GPT', 'Claude'], feature_sets=feature_sets,
        bias=[0.05, -0.1, 0.07, -0.03], ai_calibration={'a': 0.87, 'b': -0.45},
        thresholds={'min_chars': 50, 'ai_high': 0.9, 'human_low': 0.2, 'which_min': 0.5},
        eval={'summary_lines': ['这是仅供程序测试的玩具模型，没有检测效度。', '请勿用玩具模型判断作者身份。']},
    )
    cases = []
    for name, text in texts:
        prediction = ref.predict(model, text)
        case = dict(
            name=name, text=text, normalized=ref.normalize(text), distorted=ref.distort(ref.normalize(text)),
            chars=ref.cjk_count(text), markers=ref.markers(text),
            features=[ref.featurize(text, fs) for fs in feature_sets],
            pieces=ref.split_pieces(text, model.get('long_text')), doc=ref.predict_doc(model, text),
        )
        case.update(prediction)
        cases.append(case)
    golden = dict(format='aiqiang-golden', version=1, cases=cases)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for filename, value in [('toy_model.json', model), ('toy_golden.json', golden)]:
        content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        (FIXTURES / filename).write_bytes(content.encode('utf-8'))
    print('Created {} golden cases, {} grams.'.format(len(cases), sum(len(fs['vocab']) for fs in feature_sets)))


if __name__ == '__main__':
    build()
