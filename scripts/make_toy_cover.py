from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'toy-assets'
out.mkdir(exist_ok=True)
font_path = 'C:/Windows/Fonts/msyh.ttc'
bold_path = 'C:/Windows/Fonts/msyhbd.ttc'
def font(size, bold=False):
    return ImageFont.truetype(bold_path if bold else font_path, size)
im = Image.new('RGB', (1200, 900), '#edf2f7')
d = ImageDraw.Draw(im)
d.rounded_rectangle((65, 65, 1135, 835), radius=24, fill='white')
d.text((120, 112), 'AI 腔尺子', font=font(92, True), fill='#255b91')
d.text((126, 252), '贴一段中文，量量它的 AI 腔。', font=font(30), fill='#586e82')
d.rounded_rectangle((120, 346, 1080, 526), radius=10, fill='#e5eef6')
for x in range(140, 1061, 16):
    long = (x - 140) % 80 == 0
    d.line((x, 347, x, 405 if long else 376), fill='#255b91', width=3 if long else 2)
d.text((151, 438), 'AI 可能性    /    三家文风倾向    /    十条八股', font=font(29), fill='#255b91')
d.text((126, 582), '只量文风，不能证明是谁写的。', font=font(34, True), fill='#213448')
d.text((126, 655), '原作者：Claude大小姐在运营', font=font(24), fill='#586e82')
d.text((126, 700), '原视频：BV1xNsV6CE7d  ·  浏览器 / Toy 移植版', font=font(22), fill='#586e82')
im.save(out/'cover.png', optimize=True)
icon = Image.new('RGB', (512, 512), '#255b91')
di = ImageDraw.Draw(icon)
for x in range(32, 490, 32):
    di.line((x, 0, x, 58 if x % 64 == 0 else 34), fill='#c5d8ea', width=3)
di.text((115, 116), '尺', font=font(280, True), fill='white')
icon.save(out/'icon.png', optimize=True)
print('Created 1200x900 cover and 512x512 icon')
