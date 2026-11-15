"""Claude's training texts: each one was written by a fresh Claude Code subagent (general-purpose agent, no access to
any other conversation), given exactly the prompt text plus one line telling it to save the answer with its Write
tool to a neutrally named scratch file (p<i>_<k>.txt / q<i>_<k>.txt, so the file path says nothing about AI
detection). Sample k: 1 and 2 = Claude Opus 5.5, 3 = Claude Sonnet 5.5. The v2 prompts (q*) also said
"answer directly, don't look anything up or call other tools".
This script copies the scratch files into ../data/ai/claude/<pid>_s<k>.txt and writes meta.jsonl.

  python collect_claude.py <scratch folder>
"""
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'data', 'ai', 'claude')
MODELS = {1: 'claude-opus-5-5', 2: 'claude-opus-5-5', 3: 'claude-sonnet-5-5'}


def main(src):
    os.makedirs(OUT, exist_ok=True)
    meta_path = os.path.join(OUT, 'meta.jsonl')
    have = set()
    if os.path.exists(meta_path):
        for line in io.open(meta_path, encoding='utf-8'):
            m = json.loads(line)
            have.add((m['prompt'], str(m['sample'])))
    n = 0
    with io.open(meta_path, 'a', encoding='utf-8') as log:
        for f in sorted(os.listdir(src)):
            m = re.match(r'([pqr]\d+)_(\d)\.txt$', f)
            if not m or os.path.getsize(os.path.join(src, f)) == 0:
                continue
            pid, k = m.group(1), int(m.group(2))
            t = io.open(os.path.join(src, f), encoding='utf-8-sig').read().strip()
            dst = os.path.join(OUT, '%s_s%d.txt' % (pid, k))
            io.open(dst, 'w', encoding='utf-8').write(t + '\n')
            if (pid, str(k)) not in have:
                log.write(json.dumps(dict(model='Claude', prompt=pid, sample=k, served_model=MODELS[k],
                                          how='fresh Claude Code subagent (general-purpose), one per text',
                                          chars=len(t), time=time.strftime(
                                              '%Y-%m-%d %H:%M:%S', time.gmtime(os.path.getmtime(os.path.join(src, f))))),
                                     ensure_ascii=False) + '\n')
                n += 1
    print('copied; %d new meta rows' % n)


if __name__ == '__main__':
    main(sys.argv[1])
