#!/usr/bin/env python3
"""Render the Open Graph card (web/og.png, 1200x630).

Deterministic Pillow drawing in the site's visual language: ink background,
layers x tokens heatmap with logit-lens flip dots on the right, wordmark +
tagline on the left. Regenerate after brand/tagline changes:

    .venv/bin/python scripts/make_og.py
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
INK = (34, 52, 74)
INK_CELL = (44, 63, 88)          # empty cell on ink
PAPER = (238, 241, 244)
MUTED = (147, 163, 181)
AMBER = (227, 155, 45)
RED = (176, 46, 60)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "og.png")


def rnd(i, j):                    # same pseudo-random the mock views use
    s = math.sin(i * 127.1 + j * 311.7) * 43758.5453
    return s - math.floor(s)


def heat(w):
    def mix(a, b, t):
        return tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(3))
    if w <= 0:
        return INK_CELL
    if w < .55:
        return mix((237, 240, 243), AMBER, w / .55)
    return mix(AMBER, RED, (w - .55) / .45)


def font(size, bold=False):
    menlo = "/System/Library/Fonts/Menlo.ttc"
    if os.path.exists(menlo):
        return ImageFont.truetype(menlo, size, index=1 if bold else 0)
    return ImageFont.truetype(                     # linux fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono"
        + ("-Bold" if bold else "") + ".ttf", size)


img = Image.new("RGB", (W, H), INK)
d = ImageDraw.Draw(img)

# heatmap field bleeding off the right edge; columns fade in toward the text
CELL, GAP = 30, 4
X0, COLS, ROWS = 640, 18, 19
for c in range(COLS):
    fade = min(1.0, (c + 1) / 5)              # left columns ease toward ink
    for r in range(ROWS):
        v = rnd(r, c)
        v = 0.0 if v < .30 else (v - .30) / .70
        col = heat(v)
        col = tuple(round(INK[k] + (col[k] - INK[k]) * fade) for k in range(3))
        x = X0 + c * (CELL + GAP)
        y = -10 + r * (CELL + GAP)
        d.rounded_rectangle([x, y, x + CELL, y + CELL], 4, fill=col)
        if fade > .8 and rnd(r + 31, c + 17) > .86:   # lens flip dots
            settle = rnd(r + 7, c + 3) > .5
            cx, cy = x + CELL - 8, y + 8
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4],
                      fill=RED if settle else PAPER,
                      outline=PAPER if settle else INK, width=2)

# left panel: logo, wordmark, tagline, domain
LX = 64
d.rounded_rectangle([LX, 72, LX + 84, 156], 18, fill=INK)  # logo tile
lc = [[AMBER, RED, (216, 170, 78)],
      [(200, 106, 53), AMBER, RED],
      [RED, (216, 170, 78), AMBER]]
for r in range(3):
    for c in range(3):
        x = LX + 8 + c * 24
        y = 80 + r * 24
        d.rounded_rectangle([x, y, x + 20, y + 20], 4, fill=lc[r][c])
d.ellipse([LX + 56, 104, LX + 76, 124], outline=PAPER, width=4)

d.text((LX + 104, 88), "model-graph", font=font(52, bold=True), fill=PAPER)

d.text((LX, 226), "see your", font=font(72, bold=True), fill=PAPER)
d.text((LX, 312), "model think.", font=font(72, bold=True), fill=AMBER)

sub = ["live LLM internals, streamed per token",
       "norms · attention · logit lens · MoE",
       "OpenAI-compatible · open source · MIT"]
for i, line in enumerate(sub):
    d.text((LX, 436 + i * 38), line, font=font(24), fill=MUTED)

d.text((LX, 566), "model-graph.com", font=font(26, bold=True), fill=AMBER)

img.save(OUT, optimize=True)
print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")
