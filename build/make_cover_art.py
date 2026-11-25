#!/usr/bin/env python3
"""
Prepare the cover plates for "The Patient Quant".

Both plates are 6 x 9 in at 300 DPI (1800 x 2700 px), saved as JPEG so that
LaTeX embeds them as DCTDecode rather than as raw RGB.  The front uses the
whole artwork; the back uses a different crop so it is not a repeat.

The scrim is the alpha ramp that keeps the cover type legible over the
artwork.  It is baked into the plate here rather than drawn in the cover
document, so that the finished cover carries one raster image and vector
type, and no hundreds of vector rectangles.

The front's ramp is shaped around the cover layout: the title block sits at
the top of the page and the credits at the foot, so both are held down to
near-ink; the band through the middle, where the two wealth paths run, is
left almost clear, because that is the subject and it has to read.
"""
import pathlib

from PIL import Image, ImageEnhance

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
SRC = FIG / "cover_art.jpg"
DPI = 300
W, H = 6 * DPI, 9 * DPI

FIG.mkdir(parents=True, exist_ok=True)

im = Image.open(SRC).convert("RGB")
# A mild lift before the scrim, so that the plate keeps its luminosity
# instead of being flattened to black by the legibility ramp.
im = ImageEnhance.Brightness(im).enhance(1.24)
im = ImageEnhance.Contrast(im).enhance(1.08)
w, h = im.size

# Alpha at the given height in inches from the foot of the page.
#
# The front darkens the top band, where the title block is, and the foot,
# where the credits are, and leaves the middle of the page clear.
FRONT_STOPS = [(0.00, 0.95), (0.85, 0.94), (1.40, 0.90), (1.70, 0.72),
               (1.95, 0.42), (2.30, 0.24), (3.20, 0.17), (4.80, 0.16),
               (5.60, 0.20), (5.95, 0.40), (6.35, 0.70), (6.75, 0.82),
               (7.20, 0.85), (8.10, 0.85), (9.00, 0.85)]

# The back crop is the crisis at the left of the chart, so it wants a
# flatter ramp: the type sits in the upper half and the paths in the lower
# third, and neither may fight the blurb.
BACK_STOPS = [(0.00, 0.93), (1.30, 0.90), (2.60, 0.86), (3.90, 0.80),
              (5.20, 0.74), (6.50, 0.70), (7.60, 0.72), (9.00, 0.74)]

INK = (0x08, 0x0B, 0x14)


def ramp(stops, y):
    """Alpha at height y (inches from the foot) for a list of (y, alpha) stops."""
    if y <= stops[0][0]:
        return stops[0][1]
    if y >= stops[-1][0]:
        return stops[-1][1]
    for (y0, a0), (y1, a1) in zip(stops[:-1], stops[1:]):
        if y0 <= y <= y1:
            return a0 + (a1 - a0) * (y - y0) / (y1 - y0)
    return stops[-1][1]


def scrim(plate, stops):
    """Bake the legibility ramp into the plate."""
    import numpy as np
    a = np.asarray(plate).astype("float32")
    h = a.shape[0]
    ys = (1.0 - np.arange(h) / h) * 9.0        # inches from the foot
    alphas = np.array([ramp(stops, float(y)) for y in ys], dtype="float32")
    alphas = alphas[:, None, None]
    ink = np.array(INK, dtype="float32")[None, None, :]
    out = a * (1.0 - alphas) + ink * alphas
    return Image.fromarray(out.clip(0, 255).astype("uint8"))


front = scrim(im.resize((W, H), Image.BICUBIC), FRONT_STOPS)
front.save(FIG / "cover_front_art.jpg", "JPEG", quality=88, optimize=True,
           progressive=True, subsampling=1)

# A 2:3 crop, so the plate is not distorted: the left end of the chart,
# where the two paths part company at the crisis.
back = im.crop((0, int(h * 0.115), int(w * 0.54), int(h * 0.82)))
back = scrim(back.resize((W, H), Image.BICUBIC), BACK_STOPS)
back.save(FIG / "cover_back_art.jpg", "JPEG", quality=88, optimize=True,
          progressive=True, subsampling=1)

for p in (FIG / "cover_front_art.jpg", FIG / "cover_back_art.jpg"):
    print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")
