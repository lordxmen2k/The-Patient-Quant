#!/usr/bin/env python3
"""
The source artwork for the cover of "The Patient Quant".

This is the one raster artefact in the book that is not a figure: the
original artwork that the cover plates are cropped and scrimmed from.  It is
generated here rather than licensed, so that the cover is original work and
carries no attribution requirement.

The composition is flat by design.  There is no book, no spine, no binding,
no page edges and no perspective: the artwork is a single flat field, edge to
edge.

The subject is the book's own subject, and the artwork is drawn from the
book's own measured record rather than from an illustration of it:

  * The hero is the pair of wealth paths over the system record --- the
    index, bought and held, and the system --- on one axis, with the index's
    worst drawdown washed in as the hole it fell into and the system's path
    staying out of it.  That is the book's central result, drawn from
    build/series/record.csv and build/series/panel_returns.csv.

  * Behind it runs the investable universe: the sixty-one funds placed on
    their own measured risk and return, the ones the system actually holds
    picked out in the accent.  The cloud is soft, so it reads as the field
    the record was carved out of rather than as a second chart.

  * The house grid runs through both, at the same 0.36-inch pitch as the
    page furniture.

Output: figures/cover_art.jpg at 2400 x 3600 px (400 DPI at the 6 x 9 trim).
"""
import gc
import pathlib

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
SERIES = ROOT / "build" / "series"
FIG.mkdir(parents=True, exist_ok=True)

W, H = 2400, 3600
DPI = 400
SS = 2                      # supersampling factor for the line work

INK = (0x08, 0x0B, 0x14)
DEEP = (0x04, 0x05, 0x0B)
PANEL = (0x12, 0x18, 0x28)
GOLD = (0xE8, 0xB8, 0x5C)   # the accent: the system, the numbers that carry
COOL = (0x6F, 0xA3, 0xD6)   # the comparator, held at arm's length
PATH_COOL = (0x4C, 0x78, 0xB0)   # the comparator's line: a shade deeper
TXT = (0xE6, 0xE9, 0xF0)
WARM = (0xC9, 0x70, 0x5A)   # the drawdown: the thing that went wrong
GRID = (0x14, 0x1A, 0x28)

# ---- the chart band, in inches from the foot of the page -------------------
CHART_BOTTOM = 1.85
CHART_TOP = 5.85
CHART_LEFT = 0.70
CHART_RIGHT = 5.30

# ---- the cloud band, in inches from the foot --------------------------------
CLOUD_BOTTOM = 6.20
CLOUD_TOP = 8.55


# =====================================================================
#  The data: the book's own measured record
# =====================================================================
def load_paths():
    """The index path and the system path, over the system record.

    Both are normalised to 1.0 at the first month of the record so that the
    two curves are comparable on one axis.
    """
    rec = pd.read_csv(SERIES / "record.csv", parse_dates=["date"])
    panel = pd.read_csv(SERIES / "panel_returns.csv", parse_dates=["date"])
    first, last = rec["date"].min(), rec["date"].max()

    window = panel[(panel["date"] >= first) & (panel["date"] <= last)]
    window = window.sort_values("date").reset_index(drop=True)
    index_path = (1.0 + window["SPY"].fillna(0.0)).cumprod()
    index_path = index_path / index_path.iloc[0]

    system_path = rec["exempt_value"].to_numpy(dtype="float64")
    system_path = system_path / system_path[0]

    n = min(len(index_path), len(system_path))
    return (index_path.to_numpy()[:n], system_path[:n],
            rec["date"].to_numpy()[:n])


def load_universe(first, last):
    """The sixty-one funds, placed on their own measured risk and return.

    A fund is placed at its annualised volatility and its annualised return
    over the record window, using only the months it actually existed.  The
    funds the system holds are flagged, because the book's second result is
    how few of them are ever held at once.
    """
    panel = pd.read_csv(SERIES / "panel_returns.csv", parse_dates=["date"])
    window = panel[(panel["date"] >= first) & (panel["date"] <= last)]

    weights = pd.read_csv(SERIES / "weights.csv", parse_dates=["date"])
    held = {c for c in weights.columns
            if c != "date" and float(weights[c].abs().max()) > 1e-9}

    points = []
    for ticker in panel.columns:
        if ticker == "date":
            continue
        series = window[ticker].to_numpy(dtype="float64")
        series = series[~np.isnan(series)]
        if len(series) < 24:
            continue
        total = float(np.prod(1.0 + series))
        years = len(series) / 12.0
        cagr = total ** (1.0 / years) - 1.0 if total > 0 else -1.0
        vol = float(np.std(series, ddof=1) * np.sqrt(12.0))
        points.append((ticker, vol, cagr, ticker in held))
    return points


# =====================================================================
#  The ground
# =====================================================================
def field() -> Image.Image:
    """A deep vertical gradient with a faint panel wash in the middle.

    The ground is drawn at final resolution; only the line work is
    supersampled, so the memory cost stays small.
    """
    ys = np.linspace(0.0, 1.0, H, dtype="float32")[:, None, None]
    head = np.array(DEEP, dtype="float32")[None, None, :]
    foot = np.array(INK, dtype="float32")[None, None, :]
    ramp = head * (1.0 - ys) + foot * ys
    wash = np.exp(-((ys - 0.46) ** 2) / 0.060)
    band = (ramp * (1.0 - 0.30 * wash)
            + np.array(PANEL, dtype="float32")[None, None, :] * 0.30 * wash)
    pixels = np.repeat(band, W, axis=1).clip(0, 255).astype("uint8")
    image = Image.fromarray(pixels)
    del ramp, wash, band, pixels, ys
    gc.collect()
    return image


def grid(draw: ImageDraw.ImageDraw) -> None:
    """The house grid, at the same 0.36-inch pitch as the page furniture."""
    pitch = int(0.36 * DPI)
    for x in range(0, W + pitch, pitch):
        draw.line([(x, 0), (x, H)], fill=GRID, width=2)
    for y in range(0, H + pitch, pitch):
        draw.line([(0, y), (W, y)], fill=GRID, width=2)


# =====================================================================
#  The cloud: the investable universe
# =====================================================================
def cloud(points):
    """A soft layer of the sixty-one funds, on their own risk and return.

    The layer is blurred before it is composited, so that it reads as the
    field behind the record rather than as a chart competing with it.  The
    funds the system holds are drawn again, sharp, on top.
    """
    vols = np.array([p[1] for p in points])
    cagrs = np.array([p[2] for p in points])
    v0, v1 = np.percentile(vols, 2.0), np.percentile(vols, 98.0)
    c0, c1 = np.percentile(cagrs, 3.0), np.percentile(cagrs, 97.0)

    def px(vol):
        return CHART_LEFT * DPI + (vol - v0) / (v1 - v0) * (CHART_RIGHT - CHART_LEFT) * DPI

    def py(cagr):
        return H - (CLOUD_BOTTOM * DPI
                    + (cagr - c0) / (c1 - c0) * (CLOUD_TOP - CLOUD_BOTTOM) * DPI)

    soft = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    sd = ImageDraw.Draw(soft)
    sharp = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    hd = ImageDraw.Draw(sharp)

    for ticker, vol, cagr, is_held in points:
        x, y = px(vol) * SS, py(cagr) * SS
        r = 11 * SS
        if is_held:
            hd.ellipse([x - r, y - r, x + r, y + r], fill=GOLD + (210,))
            hd.ellipse([x - r * 2.2, y - r * 2.2, x + r * 2.2, y + r * 2.2],
                       outline=GOLD + (70,), width=2 * SS)
        else:
            sd.ellipse([x - r, y - r, x + r, y + r], fill=COOL + (80,))
    soft = soft.filter(ImageFilter.GaussianBlur(radius=5 * SS))
    sharp = sharp.filter(ImageFilter.GaussianBlur(radius=1 * SS))
    return soft, sharp


# =====================================================================
#  The hero: the two wealth paths
# =====================================================================
def hero(index_path, system_path):
    """The record, drawn.

    The index is the comparator, so it is drawn in the cool colour; the
    system is the book's subject, so it is drawn in the accent and slightly
    heavier.  The axis is logarithmic, because a wealth path over twenty-four
    years is a compound rate and nothing else: on a log axis each path is a
    straight line whose slope is its own annualised return, and the distance
    between the two lines is the return the protection gives up.

    The index's worst drawdown is washed in as the hole it fell into, and
    the system's path is drawn to stay out of it.
    """
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    log_index = np.log(index_path)
    log_system = np.log(system_path)
    lo = min(log_index.min(), log_system.min())
    hi = max(log_index.max(), log_system.max())
    span = hi - lo

    def px(i):
        return CHART_LEFT * DPI + i / (len(index_path) - 1) * (CHART_RIGHT - CHART_LEFT) * DPI

    def py(v):
        return H - (CHART_BOTTOM * DPI + (v - lo) / span * (CHART_TOP - CHART_BOTTOM) * DPI)

    ip = [(px(i) * SS, py(v) * SS) for i, v in enumerate(log_index)]
    sp = [(px(i) * SS, py(v) * SS) for i, v in enumerate(log_system)]

    # The index's worst drawdown, washed in: the region between the running
    # peak and the trough that followed it.
    run = np.maximum.accumulate(index_path)
    dd = index_path / run - 1.0
    trough_at = int(np.argmin(dd))
    peak_at = int(np.argmax(index_path[:trough_at + 1])) if trough_at > 0 else 0
    hole = ([(px(i) * SS, py(log_index[i]) * SS) for i in range(peak_at, trough_at + 1)]
            + [(px(i) * SS, py(np.log(run[i])) * SS) for i in
               range(trough_at, peak_at - 1, -1)])
    d.polygon(hole, fill=WARM + (72,))

    # The month of the trough, marked on both paths: what the index and the
    # system were worth at the same moment.
    x = px(trough_at) * SS
    d.line([(x, CHART_BOTTOM * DPI * SS), (x, CHART_TOP * DPI * SS)],
           fill=WARM + (70,), width=2 * SS)
    for path, colour in ((log_index, PATH_COOL), (log_system, GOLD)):
        y = py(path[trough_at]) * SS
        d.ellipse([x - 7 * SS, y - 7 * SS, x + 7 * SS, y + 7 * SS],
                  fill=colour + (255,))

    d.line(ip, fill=PATH_COOL + (235,), width=6 * SS, joint="curve")
    d.line(sp, fill=GOLD + (245,), width=9 * SS, joint="curve")

    # The end points: where each path finished.
    d.ellipse([ip[-1][0] - 8 * SS, ip[-1][1] - 8 * SS,
               ip[-1][0] + 8 * SS, ip[-1][1] + 8 * SS], fill=PATH_COOL + (255,))
    d.ellipse([sp[-1][0] - 11 * SS, sp[-1][1] - 11 * SS,
               sp[-1][0] + 11 * SS, sp[-1][1] + 11 * SS], fill=GOLD + (255,))

    return layer


# =====================================================================
def main() -> int:
    index_path, system_path, dates = load_paths()
    points = load_universe(pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]))
    print(f"record {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}, "
          f"{len(index_path)} months; {len(points)} funds in the cloud")

    base = field()
    base_draw = ImageDraw.Draw(base)
    grid(base_draw)

    soft, sharp = cloud(points)
    base = Image.alpha_composite(base.convert("RGBA"),
                                 soft.resize((W, H), Image.LANCZOS))
    base = Image.alpha_composite(base, sharp.resize((W, H), Image.LANCZOS))
    del soft, sharp
    gc.collect()

    base = Image.alpha_composite(base, hero(index_path, system_path)
                                  .resize((W, H), Image.LANCZOS))
    base = base.convert("RGB")

    out = FIG / "cover_art.jpg"
    base.save(out, "JPEG", quality=93, optimize=True, progressive=True,
              subsampling=1)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
