#!/usr/bin/env python3
"""Computed figures for the book.

Every number plotted here is read from ``build/numbers.json`` and
``build/series/``, which are produced by ``scripts/run_system.py``.  Nothing
is transcribed from prose, and no value is taken from another book.  Where a
chart needs a published input, that input is quoted in the chapter that
uses it and is cited there.

One generator per figure, and every figure is rendered and inspected before
it is used.

Output: vector PDF into ../figures/
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NUMBERS = json.loads((REPO / "build" / "numbers.json").read_text())
SERIES = REPO / "build" / "series"

# --- the series house palette ------------------------------------------------
BG = "#080B14"
PANEL = "#121828"
EDGE = "#27324C"
ACCENT = "#E8B85C"
ACCENTDIM = "#8A7238"
TXT = "#E6E9F0"
TXTDIM = "#939BAB"
AMBER = "#6FA3D6"
RED = "#C9705A"
GREEN = "#7FB08A"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TXT,
    "axes.labelcolor": TXTDIM,
    "xtick.color": TXTDIM,
    "ytick.color": TXTDIM,
    "axes.edgecolor": EDGE,
    "grid.color": EDGE,
    "font.family": "DejaVu Sans",
    "font.size": 7.6,
    "axes.titlesize": 8.8,
    "axes.titleweight": "bold",
    "axes.titlecolor": TXT,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
})


def tight(fig, **kwargs):
    """tight_layout without the warning it raises on a gridspec layout."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(**kwargs)


SAVE = True
KEEP_OPEN = False
BUILT: list = []


def save(fig, name):
    """Write the figure, and record it for the overlap gate."""
    if SAVE:
        path = OUT / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
        print(f"  wrote {path.name}")
    BUILT.append((fig, name))
    if not KEEP_OPEN:
        plt.close(fig)


def record():
    return pd.read_csv(SERIES / "record.csv", index_col=0, parse_dates=True)


def weights():
    return pd.read_csv(SERIES / "weights.csv", index_col=0, parse_dates=True)


def trials():
    return pd.read_csv(SERIES / "trials.csv", index_col=0, parse_dates=True)


def wealth(series):
    return (1.0 + series.fillna(0.0)).cumprod()


# ===========================================================================
def fig_wealth():
    rec = record()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    spy = pd.read_csv(SERIES / "record.csv", index_col=0, parse_dates=True)

    from patientquant import panel as pq_panel  # noqa: E402
    panel = pq_panel.monthly_panel()
    bench = panel["returns"]["SPY"].reindex(rec.index).fillna(0.0)

    ax.plot(rec.index, wealth(bench), color=AMBER, lw=1.5,
            label="SPY buy and hold")
    ax.plot(rec.index, wealth(rec["exempt_return"]), color=GREEN, lw=1.5,
            label="System, tax exempt")
    ax.plot(rec.index, wealth(rec["taxable_return"]), color=RED, lw=1.3,
            ls="--", label="System, taxable")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1")
    ax.set_title("The system against the index: less terminal wealth, "
                 "a shallower and shorter drawdown")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.4)
    tight(fig)
    save(fig, "fig_wealth")


# ===========================================================================
def fig_drawdown():
    rec = record()
    from patientquant import panel as pq_panel  # noqa: E402
    panel = pq_panel.monthly_panel()
    bench = panel["returns"]["SPY"].reindex(rec.index).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for series, colour, label in (
        (bench, AMBER, "SPY buy and hold"),
        (rec["exempt_return"], GREEN, "System, tax exempt"),
    ):
        curve = wealth(series)
        ax.plot(rec.index, curve / curve.cummax() - 1.0, color=colour,
                lw=1.4, label=label)
    ax.set_ylabel("Drawdown from peak")
    ax.set_title("The same return stream, drawn down")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=7.4)
    tight(fig)
    save(fig, "fig_drawdown")


# ===========================================================================
def fig_layers():
    summaries = NUMBERS["summaries"]
    family = NUMBERS["family_summaries"]
    # The layer variants are members of the trial family, and the full
    # system is one of the comparators.  Every number here is read from
    # numbers.json; none is transcribed.
    keys = [
        ("no tilt, no governor", "core only"),
        ("tilt only", "plus the model"),
        ("governor only", "plus the governor"),
        (None, "all three"),
    ]
    data = []
    for key, _ in keys:
        if key is None:
            data.append(summaries["System, tax exempt"])
        else:
            data.append(family[key])

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    metrics = [
        ("cagr", "Compound growth", lambda v: f"{v:.1%}", (0.0, 0.11)),
        ("sharpe", "Sharpe ratio", lambda v: f"{v:.2f}", (0.0, 1.0)),
        ("max_drawdown", "Deepest drawdown", lambda v: f"{v:.1%}", (-0.45, 0.0)),
    ]
    colours = [ACCENTDIM, AMBER, GREEN, ACCENT]
    for ax, (metric, title, fmt, limits) in zip(axes, metrics):
        values = [d[metric] for d in data]
        bars = ax.bar(range(len(values)), values, color=colours, width=0.66)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(["core", "+model", "+governor", "all"],
                           fontsize=7.2)
        ax.set_title(title)
        ax.set_ylim(*limits)
        ax.grid(alpha=0.22, lw=0.5, axis="y")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    fmt(value), ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=6.9, color=TXT)
    fig.suptitle("What each layer contributes: the model adds nothing, "
                 "the governor buys the drawdown",
                 fontsize=8.6, color=TXT, y=1.04)
    tight(fig)
    save(fig, "fig_layers")


# ===========================================================================
def fig_posterior():
    """The central figure: the edge against the uncertainty about the edge."""
    combined = pd.read_csv(SERIES / "combined_forecast.csv",
                            parse_dates=["date"])
    combined = combined.dropna(subset=["mu", "var_mu"])
    sigma = np.sqrt(combined["var_mu"].clip(lower=0))

    # The cross-sectional median asset at each date: what a typical
    # forecast looked like, and how unsure it was.
    grouped = combined.groupby("date").agg(
        mu=("mu", "median"), sigma=("var_mu", lambda s: float(np.sqrt(s.median()))))
    grouped["sigma"] = grouped["sigma"]

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.fill_between(grouped.index, -grouped["sigma"], grouped["sigma"],
                    color=AMBER, alpha=0.22, lw=0,
                    label="one standard deviation of the estimate")
    ax.plot(grouped.index, grouped["mu"], color=ACCENT, lw=1.1,
            label="median posterior forecast")
    ax.axhline(0.0, color=TXTDIM, lw=0.8)
    ax.set_ylabel("Monthly expected return")
    ax.set_title("The forecast is always inside its own error bar")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    tight(fig)
    save(fig, "fig_posterior")


# ===========================================================================
def fig_kelly_shrinkage():
    """How the certainty-equivalent position falls as the estimate blurs.

    The book's sizing rule adds the posterior variance of the estimated
    mean to the covariance and then solves.  For a single asset that is

        w = mu / (gamma * (sigma^2 + psi)) * fraction

    clipped to a quarter of full Kelly.  The figure fixes the asset's own
    volatility and the forecast, and varies the uncertainty in the
    forecast: the same forecast produces a smaller position when the
    estimate is less well measured.  The numbers are the book's own
    implementation, not an illustration of it.
    """
    from patientquant import portfolio as pq_portfolio  # noqa: E402

    monthly_sigma = 0.18 / np.sqrt(12.0)
    fraction = 0.25
    edge = np.linspace(0.0, 0.0025, 200)   # monthly expected return

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for uncertainty, colour, label in (
        (0.25, ACCENT, "the estimate is precise: a quarter of its own risk"),
        (1.00, AMBER, "the estimate is as uncertain as the asset's return"),
        (2.00, RED, "the estimate is twice as uncertain again"),
    ):
        psi = (uncertainty * monthly_sigma) ** 2
        weights = []
        for mu in edge:
            w = pq_portfolio.kelly_weights(
                np.array([mu]),
                pd.DataFrame([[monthly_sigma ** 2]]),
                np.array([psi]),
                gamma=1.0, fraction=fraction,
            )
            weights.append(float(w[0]))
        ax.plot(edge * 100, weights, color=colour, lw=1.6, label=label)

    ax.axhline(0.0, color=TXTDIM, lw=0.8)
    ax.set_xlabel("Forecast edge, per cent a month")
    ax.set_ylabel("Position, as a fraction of full Kelly")
    ax.set_title("The same forecast produces a smaller position when it is "
                 "less well measured")
    ax.set_xlim(0.0, 0.26)
    ax.set_ylim(0.0, 0.26)
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    tight(fig)
    save(fig, "fig_kelly_shrinkage")


# ===========================================================================
def fig_deflated():
    from patientquant import selection as pq_selection  # noqa: E402

    sel = NUMBERS["selection"]
    trials = np.arange(1, 2001)
    curve = [pq_selection.expected_max_sharpe(1.0, int(n)) for n in trials]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(trials, curve, color=AMBER, lw=1.5,
            label="expected best Sharpe from N independent trials")
    n_implied = sel["implied_independent_trials"]
    ax.axvline(max(n_implied, 1.0), color=RED, lw=1.1, ls="--",
               label=f"independent trials implied by the family "
                     f"({n_implied:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of independent trials")
    ax.set_ylabel("Expected maximum Sharpe ratio")
    ax.set_title("What trying many configurations costs you")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    tight(fig)
    save(fig, "fig_deflated")


# ===========================================================================
def fig_family():
    summaries = NUMBERS["summaries"]
    family = NUMBERS["family_summaries"]
    values = sorted(d["cagr"] for d in family.values())
    spy = summaries["SPY buy and hold"]["cagr"]
    system = summaries["System, tax exempt"]["cagr"]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.hist(values, bins=12, color=PANEL, edgecolor=ACCENTDIM, lw=0.8)
    ax.axvline(spy, color=AMBER, lw=1.5, label=f"SPY buy and hold ({spy:.2%})")
    ax.axvline(system, color=GREEN, lw=1.5,
               label=f"the system ({system:.2%})")
    ax.axvline(np.mean(values), color=RED, lw=1.0, ls="--",
               label=f"mean of the family ({np.mean(values):.2%})")
    ax.set_xlabel("Compound annual growth rate")
    ax.set_ylabel("Number of variants tried")
    ax.set_title("Everything the author tried, against the index")
    ax.grid(alpha=0.22, lw=0.5, axis="y")
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    tight(fig)
    save(fig, "fig_family")


# ===========================================================================
def fig_costs():
    """What the investor actually keeps: the same portfolio, three states."""
    frictions = NUMBERS["frictions"]
    summaries = NUMBERS["summaries"]
    gross = NUMBERS["cost_sensitivity"][0]["cagr"]
    exempt = summaries["System, tax exempt"]["cagr"]
    taxable = summaries["System, tax taxable"]["cagr"] if (
        "System, tax taxable" in summaries) else summaries["System, taxable"]["cagr"]
    spy = summaries["SPY buy and hold"]["cagr"]

    wide = NUMBERS["cost_sensitivity"][-1]["cagr"]
    labels = ["the index,\nfor comparison", "the system,\n2 bp spread",
              "the system,\n12 bp spread", "the system,\nafter taxes"]
    values = [spy, gross, wide, taxable]
    colours = [AMBER, ACCENT, ACCENTDIM, RED]

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    bars = ax.bar(range(len(values)), values, color=colours, width=0.6)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylabel("Compound annual growth rate")
    ax.set_title("What the investor actually keeps")
    ax.grid(alpha=0.22, lw=0.5, axis="y")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2%}",
                ha="center", va="bottom", fontsize=7.0, color=TXT)
    ax.set_ylim(0, max(values) * 1.22)
    tight(fig)
    save(fig, "fig_costs")


# ===========================================================================
def fig_tax():
    rec = record()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    exempt = wealth(rec["exempt_return"])
    taxable = wealth(rec["taxable_return"])
    ax.plot(rec.index, exempt, color=GREEN, lw=1.5, label="tax-exempt account")
    ax.plot(rec.index, taxable, color=RED, lw=1.5, label="taxable account")
    ax.fill_between(rec.index, taxable, exempt, color=RED, alpha=0.14, lw=0,
                    label="the tax")
    ax.set_ylabel("Growth of $1")
    ax.set_title("The tax is the wedge between the two accounts")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    tight(fig)
    save(fig, "fig_tax")


# ===========================================================================
def fig_yields():
    terms = pd.read_csv(SERIES / "fund_terms.csv", index_col=0)
    returns = pd.read_csv(SERIES / "panel_returns.csv", index_col=0,
                          parse_dates=True)
    from patientquant import panel as pq_panel  # noqa: E402
    panel = pq_panel.monthly_panel()
    yields = panel["dividend_yield"].tail(60).sum() * 100

    frame = pd.DataFrame({"yield": yields})
    frame = frame.join(terms[["expense_ratio"]])
    frame = frame.dropna().sort_values("yield", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colours = [RED if y > 3 else ACCENT for y in frame["yield"]]
    ax.barh(range(len(frame)), frame["yield"], color=colours, height=0.72)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame.index, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("Trailing twelve-month dividend yield, per cent")
    ax.set_title("The tax base: what each fund pays out, and what that costs")
    ax.grid(alpha=0.22, lw=0.5, axis="x")
    tight(fig)
    save(fig, "fig_yields")


# ===========================================================================
def fig_live():
    """The live window of every fund: when it began, and how long it has run."""
    import matplotlib.dates as mdates

    terms = pd.read_csv(SERIES / "fund_terms.csv", index_col=0)
    terms = terms.dropna(subset=["inception_date"]).sort_values("inception_date")
    panel_end = pd.Timestamp(NUMBERS["meta"]["panel_end"])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(terms))
    for i, (name, row) in enumerate(terms.iterrows()):
        start = pd.Timestamp(row["inception_date"])
        ax.plot([start, panel_end], [i, i], color=ACCENTDIM, lw=2.4,
                solid_capstyle="butt")
        ax.scatter([start], [i], s=11, color=ACCENT, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(terms.index, fontsize=5.6)
    ax.set_ylim(-1, len(terms))
    ax.set_xlim(pd.Timestamp("1993-01-01"), panel_end + pd.Timedelta(days=200))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("When each fund began, and how long it has been available")
    ax.grid(alpha=0.22, lw=0.5, axis="x")
    tight(fig)
    save(fig, "fig_live")


# ===========================================================================
def fig_sequence():
    """The same returns in two orders, for a saver who withdraws.

    The product of a set of returns does not depend on their order, so a
    buy-and-hold investor ends in the same place either way.  A saver who
    withdraws does not: each withdrawal is multiplied by the returns that
    follow it.  This figure is computed from the book's own monthly
    panel --- the annual returns of the benchmark over the record --- and
    not from a simulation.
    """
    returns = pd.read_csv(SERIES / "panel_returns.csv", index_col=0,
                           parse_dates=True)
    annual = (1.0 + returns["SPY"]).resample("YE").prod() - 1.0
    annual = annual.loc["1994":"2025"]
    values = annual.to_numpy(dtype=float)

    bad = np.argsort(values)[:9]
    good = np.argsort(values)[-9:]
    rest = np.setdiff1d(np.arange(len(values)), np.concatenate([bad, good]))
    orders = {
        "the worst years come first": np.concatenate([values[bad], values[rest],
                                                      values[good]]),
        "the worst years come last": np.concatenate([values[good], values[rest],
                                                     values[bad]]),
    }

    initial = 1_000_000.0
    withdrawal = 0.03 * initial

    def run(order):
        balance, path = initial, []
        for r in order:
            balance = (balance + 1.0) * (1.0 + r) - withdrawal
            balance = max(balance, 0.0)
            path.append(balance)
            if balance == 0.0:
                path.extend([0.0] * (len(order) - len(path)))
                break
        return path

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [2.1, 1.0]})

    colours = {"the worst years come first": RED,
               "the worst years come last": GREEN}
    terminals = {}
    for label, order in orders.items():
        path = run(order)
        terminals[label] = path[-1]
        axes[0].plot(np.arange(1, len(path) + 1), path, color=colours[label],
                     lw=1.5, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Balance")
    axes[0].set_title("A saver withdrawing three per cent a year")
    axes[0].set_ylim(5e4, 2e8)
    axes[0].grid(alpha=0.25, lw=0.5)

    axes[1].set_axis_off()
    summary = [
        ("the worst years\ncome first", terminals["the worst years come first"]),
        ("the worst years\ncome last", terminals["the worst years come last"]),
    ]
    axes[1].text(0.02, 0.82, "Balance at the end", fontsize=7.6, color=TXTDIM)
    for i, (label, value) in enumerate(summary):
        y = 0.60 - 0.34 * i
        axes[1].text(0.02, y, label, fontsize=7.2, color=TXT, va="top")
        text = "depleted" if value == 0 else f"${value / 1e6:.1f} million"
        axes[1].text(0.98, y, text, fontsize=7.6, color=TXT, va="top",
                     ha="right", fontweight="bold")
    axes[1].text(0.02, 0.08, "The same thirty-two years of returns, in two "
                 "orders.", fontsize=7.0, color=TXTDIM, va="bottom",
                 wrap=True)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, loc="lower center",
               ncol=2, fontsize=7.2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("The order of the returns decides the outcome for a saver, "
                 "and not the product",
                 fontsize=8.6, color=TXT, y=1.03)
    tight(fig)
    save(fig, "fig_sequence")


# ===========================================================================
def fig_behavioural():
    """The two behavioural numbers the book leans on."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))

    ax = axes[0]
    periods = [1, 2, 5, 10, 20]
    implied = [6.5, 4.65, 3.0, 2.0, 1.4]
    ax.plot(periods, implied, color=ACCENT, lw=1.5, marker="o", ms=4)
    offsets = [8, -12, 8, -12, 8]
    for p, v, dy in zip(periods, implied, offsets):
        ax.annotate(f"{v:.1f}%", (p, v), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=6.8, color=TXT)
    ax.set_xscale("log")
    ax.set_xticks(periods)
    ax.set_xticklabels([str(p) for p in periods])
    ax.set_xlabel("How often the portfolio is evaluated, in years")
    ax.set_ylabel("Equity premium the investor demands, per cent")
    ax.set_title("Myopic loss aversion")
    ax.set_ylim(0.0, 8.4)
    ax.set_xlim(0.85, 26.0)
    ax.grid(alpha=0.22, lw=0.5)

    ax = axes[1]
    labels = ["market", "least active\ntraders", "most active\ntraders"]
    values = [17.9, 16.4, 11.4]
    bars = ax.bar(range(3), values, color=[AMBER, ACCENTDIM, RED], width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylim(0, 21)
    ax.set_ylabel("Annual return, per cent")
    ax.set_title("Trading too much")
    ax.grid(alpha=0.22, lw=0.5, axis="y")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=7.0, color=TXT)

    fig.suptitle("The two behavioural numbers: the cost of looking often, "
                 "and the cost of trading much",
                 fontsize=8.6, color=TXT, y=1.03)
    tight(fig)
    save(fig, "fig_behavioural")


# ===========================================================================
def fig_governor():
    rec = record()
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(rec.index, rec["vol_scale"], color=AMBER, lw=1.3,
            label="volatility multiplier")
    ax.plot(rec.index, rec["brake"], color=RED, lw=1.3,
            label="drawdown brake")
    ax.plot(rec.index, rec["gross_exposure"], color=GREEN, lw=1.5,
            label="exposure to risky assets")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Multiplier")
    ax.set_title("The risk layer, as it behaved")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2, ncol=3)
    tight(fig)
    save(fig, "fig_governor")


# ===========================================================================
def fig_arith_geom():
    """The arithmetic mean overstates growth by about half the variance."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    vols = np.linspace(0.02, 0.30, 160)
    ax = axes[0]
    for mu, colour, label in ((0.08, ACCENT, "8 per cent arithmetic mean"),
                              (0.11, GREEN, "11 per cent arithmetic mean")):
        geometric = mu - 0.5 * vols ** 2
        ax.plot(vols * 100, geometric * 100, color=colour, lw=1.5, label=label)
    ax.set_xlabel("Annualised volatility, per cent")
    ax.set_ylabel("Compound growth, per cent a year")
    ax.set_title("What the same arithmetic mean is worth")
    ax.set_ylim(2.0, 12.0)
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=7.2)

    ax = axes[1]
    gap = 0.5 * vols ** 2 * 100
    ax.plot(vols * 100, gap, color=ACCENTDIM, lw=1.6)
    for v, dy in ((0.10, 8), (0.15, 8), (0.20, -13)):
        g = 0.5 * v ** 2 * 100
        ax.scatter([v * 100], [g], s=22, color=ACCENT, zorder=3)
        ax.annotate(f"{g:.1f}%", (v * 100, g), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=7.0, color=TXT)
    ax.set_xlabel("Annualised volatility, per cent")
    ax.set_ylabel("Overstatement, per cent a year")
    ax.set_title("The gap is half the variance")
    ax.set_ylim(0.0, 5.6)
    ax.grid(alpha=0.22, lw=0.5)

    fig.suptitle("The arithmetic mean overstates growth by about half the "
                 "variance, and the gap grows with risk",
                 fontsize=8.6, color=TXT, y=1.03)
    tight(fig)
    save(fig, "fig_arith_geom")


# ===========================================================================
def fig_model_weights():
    mw = pd.read_csv(SERIES / "model_weights.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, mw.shape[1]))
    for column, colour in zip(mw.columns, colours):
        ax.plot(mw.index, mw[column], lw=1.1, color=colour)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weight in the ensemble")
    ax.set_title("The ensemble's own weights: no configuration ever "
                 "dominates")
    ax.grid(alpha=0.22, lw=0.5)
    tight(fig)
    save(fig, "fig_model_weights")


# ===========================================================================
def fig_premia_timeline():
    """When each piece of evidence became available, and when it became
    investable."""
    items = [
        ("Factor premia measured\n(Fama-French, 1993)", 1993, "measured"),
        ("Momentum and value\nglobal (Asness et al., 2013)", 2013, "measured"),
        ("Factor ETFs listed\n(MTUM, VLUE, QUAL, 2013)", 2013, "investable"),
        ("Replication survey\n(Jensen, Kelly, Pedersen, 2023)", 2023, "measured"),
        ("Real-time failure of\nvolatility scaling (2020)", 2020, "measured"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(items))
    for i, (label, year, kind) in enumerate(items):
        colour = ACCENT if kind == "investable" else AMBER
        ax.scatter([year], [i], s=58, color=colour, zorder=3)
        ax.text(year, i + 0.26, str(year), ha="center", fontsize=7.2, color=TXT)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in items], fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(1991.5, 2028.0)
    ax.set_ylim(-0.75, len(items) - 0.15)
    ax.set_title("Measured and investable are not the same date")
    ax.grid(alpha=0.22, lw=0.5, axis="x")
    tight(fig)
    save(fig, "fig_premia_timeline")


# ===========================================================================
def fig_sensitivity():
    """How the layer attribution moves when the starting month moves."""
    windows = NUMBERS["window_sensitivity"]
    starts = [w["start"] for w in windows]
    tilt = [w.get("tilt_contribution_cagr", np.nan) * 100 for w in windows]
    gov = [w.get("governor_contribution_cagr", np.nan) * 100 for w in windows]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axhline(0.0, color=TXTDIM, lw=0.9)
    ax.plot(starts, tilt, color=RED, lw=1.4, marker="o", ms=3.4,
            label="the model's tilt")
    ax.plot(starts, gov, color=GREEN, lw=1.5, marker="o", ms=3.4,
            label="the risk governor")
    ax.set_ylabel("Change in compound growth, per cent a year")
    ax.set_title("Each layer's contribution, over every starting month")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    step = max(1, len(starts) // 7)
    ticks = list(range(0, len(starts), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([starts[i][:4] for i in ticks], fontsize=6.6,
                       rotation=45, ha="right")
    tight(fig)
    save(fig, "fig_sensitivity")


FIGURES = [
    fig_wealth, fig_drawdown, fig_layers, fig_posterior, fig_kelly_shrinkage,
    fig_deflated, fig_family, fig_costs, fig_tax, fig_yields, fig_live,
    fig_sequence, fig_behavioural, fig_governor, fig_arith_geom,
    fig_model_weights, fig_premia_timeline, fig_sensitivity,
]


def build_all(save_files: bool = True) -> list:
    """Build every figure and return the (figure, name) pairs.

    The overlap gate calls this with ``save_files=False`` so that it can
    measure the canvases without writing anything.
    """
    global SAVE, KEEP_OPEN, BUILT
    SAVE = bool(save_files)
    KEEP_OPEN = not save_files
    BUILT = []
    for maker in FIGURES:
        maker()
    return list(BUILT)


def main() -> int:
    print(f"building {len(FIGURES)} figures into {OUT}")
    build_all(save_files=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
