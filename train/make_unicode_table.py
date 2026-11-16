"""Freeze the two Unicode facts the features depend on, as Python 3.8 (Unicode 12.1) sees them:
  assigned  code points whose general category is not Cn (unassigned)
  alnum     code points where str.isalnum() is True (the same set regex \\w uses, minus '_')
The model was trained on Python 3.8. Newer Pythons ship newer Unicode data, where characters added after 12.1
(e.g. U+4DBF, U+9FFF) become letters, so distort() masked them and the golden tests failed on 3.9+ (reported
11-16). Both ref_features.py and the program read this table instead of the running Python's Unicode data.

  python make_unicode_table.py      (must run on Python 3.8: Unicode 12.1.0)
-> ../aiqiang/unicode_12_1.json     {"unicode": "12.1.0", "assigned": [[a, b], ...], "alnum": [[a, b], ...]}
"""
import io
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'aiqiang', 'unicode_12_1.json')


def ranges(test):
    out, start = [], None
    for cp in range(0x110000):
        ok = test(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append([start, cp - 1])
            start = None
    if start is not None:
        out.append([start, 0x10FFFF])
    return out


def main():
    if unicodedata.unidata_version != '12.1.0':
        sys.exit('run this on Python 3.8 (Unicode 12.1.0), not %s' % unicodedata.unidata_version)
    assigned = ranges(lambda cp: unicodedata.category(chr(cp)) != 'Cn')
    alnum = ranges(lambda cp: chr(cp).isalnum())
    data = dict(unicode='12.1.0', made_by='train/make_unicode_table.py on Python %s' % sys.version.split()[0],
                assigned=assigned, alnum=alnum)
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write(json.dumps(data, separators=(',', ':')) + '\n')
    print('assigned %d ranges, alnum %d ranges, %.1f KB' % (len(assigned), len(alnum), os.path.getsize(OUT) / 1e3))


if __name__ == '__main__':
    main()
