"""Public-domain essays and stories (authors who died before 1950) from Chinese Wikisource, for the evaluation:
famous old texts that people will surely paste in (《背影》《从百草园到三味书屋》...). Texts go to the work folder only.

  python fetch_pd_essays.py [--work <work folder>]   -> <work>/human_raw/pd/<title>.txt
"""
import argparse
import html
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

API = 'https://zh.wikisource.org/w/api.php'
INDEX_PAGES = ['野艸', '吶喊']          # collections whose chapter pages are all wanted
SINGLE_PAGES = [
    # 朱自清 (d. 1948)
    '荷塘月色', '背影', '匆匆', '春 (朱自清)', '一封信', '儿女', '女人', '旅行雜記', '海行雜記', '白種人——上帝的驕子',
    '阿河', '懷魏握青君', '說夢', '飄零',
    # 许地山 (d. 1941)
    '落花生',
    # 鲁迅 (d. 1936): 朝花夕拾, 彷徨
    '朝花夕拾/小引', '二十四孝圖', '五猖會', '從百草園到三味書屋', '范愛農', '藤野先生', '阿長與《山海經》', '无常',
    '父亲的病', '狗·猫·鼠', '琐记', '傷逝 (魯迅)', '在酒樓上', '孤獨者', '幸福的家庭', '弟兄', '示衆', '祝福', '肥皂',
    '長明燈', '離婚', '高老夫子',
]
HEADERS = {'User-Agent': 'aiqiang-ruler-research/0.1 (offline text-style tool; one-off fetch)'}


def call(params):
    params = dict(params, format='json', formatversion='2')
    url = API + '?' + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:  # noqa: BLE001
            print('  retry', attempt + 1, type(e).__name__, flush=True)
            time.sleep(3 + 3 * attempt)
    return {}


def links(page):
    d = call(dict(action='parse', page=page, prop='links'))
    return [x['title'] for x in (d.get('parse') or {}).get('links', []) if x.get('exists', True) and x.get('ns') == 0]


def text_of(page):
    d = call(dict(action='parse', page=page, prop='text', variant='zh-cn', disablelimitreport=1))
    h = (d.get('parse') or {}).get('text', '')
    paras = re.findall(r'<p>(.*?)</p>', h, flags=re.S)
    out = []
    for p in paras:
        p = re.sub(r'<sup.*?</sup>', '', p, flags=re.S)
        p = re.sub(r'<[^>]+>', '', p)
        p = html.unescape(p).strip()
        if len(p) >= 10:
            out.append(p)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                   'AI腔尺子_工作'))
    a = ap.parse_args()
    out = os.path.join(a.work, 'human_raw', 'pd')
    os.makedirs(out, exist_ok=True)
    pages = list(SINGLE_PAGES)
    for idx in INDEX_PAGES:
        sub = links(idx)
        print(idx, len(sub), 'chapters', sub[:30], flush=True)
        pages += sub
    n = 0
    for p in pages:
        fn = os.path.join(out, re.sub(r'[\\/:*?"<>| ]', '_', p) + '.txt')
        if os.path.exists(fn):
            continue
        t = text_of(p)
        if len(t) >= 200:
            io.open(fn, 'w', encoding='utf-8').write(t + '\n')
            n += 1
            print(' ', p, len(t), flush=True)
        time.sleep(0.5)
    print('saved', n)


if __name__ == '__main__':
    main()
