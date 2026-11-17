import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aiqiang.analyzer import analyze, load_model
from aiqiang.features import normalize, source_string
model = load_model()
golden = json.loads((ROOT/'tests/golden_real.json').read_text(encoding='utf-8'))['cases']
texts = [(c['name'], c['text']) for c in golden]
extras = [
    '', '太短了😀', '生活！！', '我走来，我坐下，我开门，我离开。',
    '😀不是一段故事，而是一种体验。\n我们喜欢风，我们喜欢雨，我们喜欢生活。',
    '# 标题\n**首先**，这不仅是生活，更是成长。\n- 希望以上对你有帮助。',
    'ＡＢＣ１２３“引号”……—— @hello https://a.example/1 [微笑] #话题#',
    ('今天我出门买菜。邻居问我吃了没有。我说还没，回去就煮面。😀\n'*90),
    'x\n'*600, '我'*500+'\n生活。', '我'*501+'\n'+'好'*149,
    '一'*310+'。\n'+'hello '*100+'。\n'+'二'*320+'。',
]
texts += [('edge-'+str(i),t) for i,t in enumerate(extras)]
rows=[dict(name=n,text=t,normalized=normalize(t),sources=[source_string(t,f) for f in model['feature_sets']],result=analyze(t,model)) for n,t in texts]
(ROOT/'tests/web_reference.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
print(f'Generated {len(rows)} Python reference cases')
