#!/usr/bin/env python3
"""生成站点图标：favicon.ico 与 images/apple-touch-icon.png。

图案母版是 images/favicon.svg（太阳：金色圆盘 + 8 条深色射线）。
本机没有 SVG 光栅化工具，母版无法直接栅格化，所以这里按同一套几何用 Pillow
再实现一遍——好处是 SVG 与位图回退保证是同一个图案，不会分裂成两套。

几何（viewBox 48）：圆盘 cx=24 cy=24 r=10，射线 8 条 span 半径 14→20，
描边 #2A2A2A / 线宽 3.5 / 圆头端点。SVG 的 stroke 居中于路径，故圆盘描边外扩半线宽；
根节点的 stroke 会被 <circle> 继承，所以圆盘是带描边的，与母版一致。

配色：站点色板 --rice #f4f2ec 与 --ink #2a2927，外加一处明确例外——
圆盘金色 #E9C46A（见 CONSTRAINTS.md）。

依赖：Pillow（`pip install pillow`）。用法：仓库根执行 python scripts/make-favicon.py
"""
from PIL import Image, ImageDraw
import math
import pathlib

GOLD = (0xE9, 0xC4, 0x6A, 255)
STROKE = (0x2A, 0x2A, 0x2A, 255)
RICE = (0xF4, 0xF2, 0xEC, 255)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREVIEW_DIR = ROOT.parent / 'designsource' / 'favicon-candidates'
SS = 8                      # 超采样倍数：先大尺寸绘制再降采样，小尺寸边缘才干净
VIEWBOX = 48.0
STROKE_W = 3.5

RAYS = [((24, 4), (24, 10)), ((24, 38), (24, 44)),
        ((4, 24), (10, 24)), ((38, 24), (44, 24)),
        ((10, 10), (14, 14)), ((34, 34), (38, 38)),
        ((10, 38), (14, 34)), ((34, 14), (38, 10))]


def sun(size, background=None, stroke=STROKE, disc=GOLD):
    """按 viewBox 48 的几何画太阳；background 非空时铺满整块底（iOS 桌面图标用）。"""
    side = size * SS
    k = side / VIEWBOX
    im = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    if background is not None:
        draw.rectangle([0, 0, side - 1, side - 1], fill=background)

    width = STROKE_W * k

    # 圆盘（含继承来的描边）：先画外扩半线宽的描边圆，再叠填充圆
    cx = cy = 24 * k
    radius = 10 * k
    outer = radius + width / 2
    draw.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=stroke)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=disc)

    # 射线：线体 + 两端圆头（Pillow 没有 line cap）
    for (x1, y1), (x2, y2) in RAYS:
        p1, p2 = (x1 * k, y1 * k), (x2 * k, y2 * k)
        draw.line([p1, p2], fill=stroke, width=max(1, int(round(width))))
        for px, py in (p1, p2):
            draw.ellipse([px - width / 2, py - width / 2,
                          px + width / 2, py + width / 2], fill=stroke)
    return im.resize((size, size), Image.LANCZOS)


def preview(im, label):
    """16px 真实像素复核：按像素类型分符号，比亮度二值化更能看出形状有没有糊。"""
    grid = im.convert('RGBA').resize((16, 16), Image.LANCZOS)
    print(f'-- {label} 真实 16px（# 描边 / o 圆盘 / + 混色 / 空白）--')
    for y in range(16):
        row = ''
        for x in range(16):
            r, g, b, a = grid.getpixel((x, y))
            if a < 70:
                row += ' '
                continue
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 110
            near_gold = abs(r - GOLD[0]) + abs(g - GOLD[1]) + abs(b - GOLD[2]) < 150
            row += '#' if dark else ('o' if near_gold else '+')
        print('   ' + ''.join(ch * 2 for ch in row))


def main():
    ico = sun(256)
    ico.save(ROOT / 'favicon.ico', format='ICO',
             sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    # iOS 会把 apple-touch-icon 的透明区域按黑色处理，深色射线画在黑底上等于消失，
    # 所以这一格垫站点纸色底。
    sun(180, background=RICE).convert('RGB').save(
        ROOT / 'images' / 'apple-touch-icon.png', 'PNG', optimize=True)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sun(512).resize((128, 128), Image.LANCZOS).save(PREVIEW_DIR / 'D-sun-as-written.png')
    sun(512, background=RICE).resize((128, 128), Image.LANCZOS).save(
        PREVIEW_DIR / 'G-sun-on-rice.png')

    preview(sun(512), 'favicon.ico（16px 档）')
    print('已写出 favicon.ico（16/32/48/64）与 images/apple-touch-icon.png（180x180）')


if __name__ == '__main__':
    main()
