#!/usr/bin/env python3
"""生成站点图标：favicon.ico 与 images/apple-touch-icon.png。

设计：白文印（洋红底 + 纸白字），取站名 SGLDBHXS 的首字母 S，
      用 Georgia Bold 衬线，呼应页面 Noto Serif SC 的衬线气质。
只用 css/styles.css 已有色板：--rice #f4f2ec / --ink #2a2927 / --accent #db2777。

依赖：Pillow（本机 Python 运行环境自带；无则 `pip install pillow`）。
用法：在仓库根执行 `python scripts/make-favicon.py`。

复核方式：脚本末尾会把 16px 真实像素按"距底色的覆盖率"打成字符画。
不要把这一步换成二值化——细笔画在降采样后残留的中间色会被判成底色，
看上去"字形消失"，实际是糊了。
"""
from PIL import Image, ImageDraw, ImageFont
import math
import pathlib

RICE = (0xF4, 0xF2, 0xEC, 255)
INK = (0x2A, 0x29, 0x27, 255)
ACCENT = (0xDB, 0x27, 0x77, 255)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREVIEW_DIR = ROOT.parent / 'designsource' / 'favicon-candidates'
SS = 8  # 超采样倍数：先大尺寸绘制再降采样，小尺寸边缘才干净

SERIF_BOLD = r'C:\Windows\Fonts\georgiab.ttf'  # Georgia Bold
SONG = r'C:\Windows\Fonts\simsun.ttc'          # 宋体


def fit_size(font_path, glyph, canvas, ratio):
    """反解字号，让字形包围盒高度占画布的 ratio。"""
    probe = 1000
    font = ImageFont.truetype(font_path, probe)
    box = font.getbbox(glyph)
    return max(1, int(probe * (canvas * ratio) / (box[3] - box[1])))


def seal(glyph, font_path, size, ratio, fg=RICE, bg=ACCENT, bold=0.0):
    """白文印：圆角色块 + 居中字。

    bold 是描边半径（占画布比例）。CJK 细笔画在 16px 会细到亚像素，
    需要靠描边把笔画加粗到 2px 左右才立得住。
    """
    side = size * SS
    im = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle([0, 0, side - 1, side - 1], radius=int(side * 0.18), fill=bg)

    font = ImageFont.truetype(font_path, fit_size(font_path, glyph, side, ratio))
    box = draw.textbbox((0, 0), glyph, font=font)
    x = (side - (box[2] - box[0])) / 2 - box[0]
    y = (side - (box[3] - box[1])) / 2 - box[1]

    radius = side * bold
    offsets = [(0, 0)]
    if radius >= 1:
        offsets += [(radius * math.cos(math.pi * i / 4), radius * math.sin(math.pi * i / 4))
                    for i in range(8)]
    for dx, dy in offsets:
        draw.text((x + dx, y + dy), glyph, font=font, fill=fg)
    return im.resize((size, size), Image.LANCZOS)


def ripple(size, bg=RICE, ring=INK, core=ACCENT):
    """候选 C：墨色涟漪（两圈墨环 + 中心洋红点），呼应页面雨滴背景。"""
    side = size * SS
    im = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle([0, 0, side - 1, side - 1], radius=int(side * 0.18), fill=bg)
    center = side / 2
    for frac, width in ((0.40, 0.075), (0.23, 0.065)):
        r = side * frac
        draw.ellipse([center - r, center - r, center + r, center + r],
                     outline=ring, width=max(1, int(side * width)))
    r = side * 0.075
    draw.ellipse([center - r, center - r, center + r, center + r], fill=core)
    return im.resize((size, size), Image.LANCZOS)


def preview(im, label, on_pink):
    grid = im.convert('RGBA').resize((16, 16), Image.LANCZOS)
    print(f'-- {label} 真实 16px（# 满 / + 中 / . 淡 / 空 透明）--')
    for y in range(16):
        row = ''
        for x in range(16):
            r, g, b, a = grid.getpixel((x, y))
            if a < 60:
                row += ' '
                continue
            if on_pink:
                cov = 1 - (r - g) / 178
            else:
                cov = 1 - (0.299 * r + 0.587 * g + 0.114 * b - 42) / (244 - 42)
            cov = max(0.0, min(1.0, cov))
            row += ' ' if cov < 0.18 else ('.' if cov < 0.45 else ('+' if cov < 0.72 else '#'))
        print('   ' + ''.join(ch * 2 for ch in row))


def main():
    # 备选方案留一份预览放进 designsource/（该目录不入库，只作设计参考）
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    seal('S', SERIF_BOLD, 512, 0.68).resize((128, 128), Image.LANCZOS).save(
        PREVIEW_DIR / 'A-seal-S.png')
    seal('山', SONG, 512, 0.70, bold=0.022).resize((128, 128), Image.LANCZOS).save(
        PREVIEW_DIR / 'B-seal-shan.png')
    ripple(512).resize((128, 128), Image.LANCZOS).save(PREVIEW_DIR / 'C-ripple.png')

    # 正式产物：采用方案 A（站名首字母，出处在 index.html 的 h1 与 Colophon）
    seal('S', SERIF_BOLD, 256, 0.68).save(
        ROOT / 'favicon.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    seal('S', SERIF_BOLD, 180, 0.68).convert('RGB').save(
        ROOT / 'images' / 'apple-touch-icon.png', 'PNG', optimize=True)

    preview(seal('S', SERIF_BOLD, 512, 0.68), 'favicon.ico（16px 档）', True)
    print('已写出 favicon.ico（16/32/48/64）与 images/apple-touch-icon.png（180x180）')


if __name__ == '__main__':
    main()
