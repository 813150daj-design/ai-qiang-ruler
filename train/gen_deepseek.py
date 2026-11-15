"""DeepSeek writes the training texts: every prompt in prompts.json, samples s1-s3 with the settings below,
a fresh context (one API call, no history) for each text. Already written files are skipped, so it can resume.
The key comes from the environment (DEEPSEEK_API_KEY, or a file named by DEEPSEEK_KEY_FILE), is read at request
time and is never printed or saved.

  python gen_deepseek.py                                 -> ../data/ai/deepseek/<pid>_s<k>.txt and meta.jsonl
  python gen_deepseek.py --prompts prompts_v2.json --rotate
"""
import concurrent.futures as cf
import io
import json
import os
import socket
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'data', 'ai', 'deepseek')
CFG = None   # set in main(): prompts.json (3 samples per prompt) or prompts_v2.json --rotate (1 per prompt)
KEY_FILE = os.environ.get('DEEPSEEK_KEY_FILE', '')   # a text file holding the API key (never printed or saved)
URL = 'https://api.deepseek.com/chat/completions'

SETTINGS = {
    1: dict(model='deepseek-v4-pro', thinking={'type': 'disabled'}, temperature=1.0, max_tokens=1500),
    2: dict(model='deepseek-v4-pro', thinking={'type': 'enabled'}, reasoning_effort='high', max_tokens=12000),
    3: dict(model='deepseek-flash', thinking={'type': 'disabled'}, temperature=1.0, max_tokens=1500),
}


def ask(prompt, k):
    key = os.environ.get('DEEPSEEK_API_KEY') or io.open(KEY_FILE, encoding='utf-8').read().strip()
    body = dict(SETTINGS[k])
    body['messages'] = [{'role': 'user', 'content': prompt}]
    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(URL, data=data, headers={'Content-Type': 'application/json',
                                                          'Authorization': 'Bearer ' + key})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read().decode('utf-8'))
            text = (d['choices'][0]['message'].get('content') or '').strip()
            if text:
                return text, d.get('model', body['model'])
            print('empty answer, retry', flush=True)
        except Exception as e:  # noqa: BLE001
            print('retry', attempt + 1, type(e).__name__, flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError('DeepSeek failed 6 times')


def job(p, k):
    path = os.path.join(OUT, '%s_s%d.txt' % (p['id'], k))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return None
    text, served = ask(p['text'], k)
    io.open(path, 'w', encoding='utf-8').write(text + '\n')
    meta = dict(model='DeepSeek', prompt=p['id'], sample=k, served_model=served,
                settings={a: b for a, b in SETTINGS[k].items() if a != 'max_tokens'},
                chars=len(text), time=time.strftime('%Y-%m-%d %H:%M:%S'))
    return meta


def main():
    global CFG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', default='prompts.json')
    ap.add_argument('--rotate', action='store_true', help='one text per prompt, setting (i mod 3) + 1')
    ap.add_argument('ids', nargs='*')
    a = ap.parse_args()
    CFG = json.load(io.open(os.path.join(HERE, a.prompts), encoding='utf-8'))
    os.makedirs(OUT, exist_ok=True)
    if a.rotate:
        jobs = [(p, i % 3 + 1) for i, p in enumerate(CFG['prompts'])]
    else:
        jobs = [(p, k) for p in CFG['prompts'] for k in (1, 2, 3)]
    if a.ids:
        jobs = [j for j in jobs if j[0]['id'] in a.ids]
    with cf.ThreadPoolExecutor(4) as ex, io.open(os.path.join(OUT, 'meta.jsonl'), 'a', encoding='utf-8') as log:
        futs = {ex.submit(job, p, k): (p['id'], k) for p, k in jobs}
        for f in cf.as_completed(futs):
            pid, k = futs[f]
            try:
                m = f.result()
            except Exception as e:  # noqa: BLE001
                print('FAILED', pid, k, e, flush=True)
                continue
            if m:
                log.write(json.dumps(m, ensure_ascii=False) + '\n')
                log.flush()
                print(pid, 's%d' % k, m['chars'], 'chars', m['served_model'], flush=True)
    print('done')


if __name__ == '__main__':
    socket.setdefaulttimeout(300)
    main()
