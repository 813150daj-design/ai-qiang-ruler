"""Build a self-contained Toy directory without third-party dependencies."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aiqiang.markers import CHECKS, HINTS

web = ROOT / 'web'
web.mkdir(exist_ok=True)
shutil.copyfile(ROOT / 'aiqiang/model.json', web / 'model.json')
checks = [dict(name=name, pattern=regex.pattern if regex else None, hint=hint)
          for (name, regex), hint in zip(CHECKS, HINTS)]
(web / 'checks.json').write_text(json.dumps(checks, ensure_ascii=False), encoding='utf-8')
shutil.copyfile(ROOT / 'LICENSE', web / 'LICENSE')
source = (ROOT / 'README.md').read_text(encoding='utf-8')
(web / 'SOURCE-README.md').write_text(source, encoding='utf-8')

# Name every runtime asset by its content. Process dependencies first.
out = ROOT / 'toy-dist'
out.mkdir(exist_ok=True)
mapping = {}
for name in ['model.json', 'checks.json', 'engine.mjs', 'worker.mjs', 'app.mjs', 'style.css', 'index.html']:
    data = (web / name).read_bytes()
    if name.endswith(('.html', '.mjs', '.css')):
        text = data.decode('utf-8')
        for old, new in mapping.items():
            text = text.replace('./' + old, './' + new)
        data = text.encode('utf-8')
    suffix = '.js' if name.endswith('.mjs') else Path(name).suffix
    target = name if name == 'index.html' else f'{Path(name).stem}-{hashlib.sha256(data).hexdigest()[:10]}{suffix}'
    mapping[name] = target
    (out / target).write_bytes(data)
for name in ['credits.html', 'LICENSE', 'SOURCE-README.md']:
    shutil.copyfile(web / name, out / name)
# Only remove obsolete build assets from this dedicated output directory.
expected = set(mapping.values()) | {'credits.html', 'LICENSE', 'SOURCE-README.md'}
for path in out.iterdir():
    if path.is_file() and path.name not in expected:
        path.unlink()
print(json.dumps({'path': str(out), 'files': len(expected), 'bytes': sum(p.stat().st_size for p in out.iterdir() if p.is_file())}))
