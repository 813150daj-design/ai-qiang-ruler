"""Gemini writes a TEST set (never used for training): an AI the ruler has never seen.
Prompts: all of prompts.json and r0-r29 of prompts_v3.json, one text each, a fresh call (no history) per text.
The key comes from the environment (GEMINI_API_KEY, or a file named by GEMINI_KEY_FILE), is read at request time
and is never printed or saved.

  python gen_gemini.py --list                 list the models this key can use
  python gen_gemini.py --model <name>         -> ../data/ai_eval/gemini/<pid>.txt and meta.jsonl
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'data', 'ai_eval', 'gemini')
API = 'https://generativelanguage.googleapis.com/v1beta'


def key():
    return os.environ.get('GEMINI_API_KEY') or io.open(os.environ['GEMINI_KEY_FILE'], encoding='utf-8').read().strip()


def call(url, body=None, timeout=180):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json', 'x-goog-api-key': key()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8')


def generate(model, prompt):
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 1.0, 'maxOutputTokens': 4096}}
    url = '%s/models/%s:streamGenerateContent?alt=sse' % (API, model)
    for attempt in range(4):
        try:
            raw = call(url, body, timeout=240)
            parts = []
            for line in raw.splitlines():
                if line.startswith('data:'):
                    d = json.loads(line[5:].strip())
                    for c in d.get('candidates', []):
                        for p in (c.get('content') or {}).get('parts', []):
                            if p.get('text') and not p.get('thought'):
                                parts.append(p['text'])
            text = ''.join(parts).strip()
            if text:
                return text
            print('  empty answer, retry', flush=True)
        except Exception as e:  # noqa: BLE001
            print('  retry', attempt + 1, type(e).__name__, str(e)[:120], flush=True)
        time.sleep(5 + 5 * attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--model', default='gemini-4-argon')
    a = ap.parse_args()
    if a.list:
        d = json.loads(call(API + '/models?pageSize=200'))
        for m in d.get('models', []):
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                print(m['name'])
        return
    ps = json.load(io.open(os.path.join(HERE, 'prompts.json'), encoding='utf-8'))['prompts']
    ps += [p for p in json.load(io.open(os.path.join(HERE, 'prompts_v3.json'), encoding='utf-8'))['prompts']
           if int(p['id'][1:]) < 30]
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, 'meta.jsonl'), 'a', encoding='utf-8') as log:
        for p in ps:
            path = os.path.join(OUT, '%s.txt' % p['id'])
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            t0 = time.time()
            text = generate(a.model, p['text'])
            if not text:
                print(p['id'], 'FAILED', flush=True)
                continue
            io.open(path, 'w', encoding='utf-8').write(text + '\n')
            log.write(json.dumps(dict(model='Gemini', prompt=p['id'], served_model=a.model, chars=len(text),
                                      seconds=round(time.time() - t0, 1), time=time.strftime('%Y-%m-%d %H:%M:%S')),
                                 ensure_ascii=False) + '\n')
            log.flush()
            print(p['id'], len(text), 'chars', '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
