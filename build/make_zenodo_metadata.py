#!/usr/bin/env python3
"""Write the Zenodo metadata for "The Patient Quant".

The deposit record quotes the book's headline numbers, so it is generated
from ``build/numbers.json`` rather than typed.  The format follows the
series house format: a title, a two-paragraph description with the key
sentences in bold, twenty keywords split into ten generic and ten
specific, and the remaining metadata fields.

    python3 build/make_zenodo_metadata.py

Writes ZENODO_METADATA.md in the root of the repository.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
NUMBERS = ROOT / "build" / "numbers.json"
OUT = ROOT / "ZENODO_METADATA.md"

TITLE = ("The Patient Quant: Building a Machine-Learned Portfolio for One "
         "Investor, and Measuring What It Is Actually Worth")
CREATOR = "Gerald Enrique Nelson Mc Kenzie"
PUBLISHER = "Tico Internet LLC"
EDITION = "2026 First Edition"
LICENSE = "Apache License 2.0"
DOI = "10.5281/zenodo.24527200"

GENERIC_KEYWORDS = [
    "portfolio management",
    "machine learning",
    "Bayesian estimation",
    "risk management",
    "investment strategy",
    "backtesting",
    "quantitative finance",
    "asset allocation",
    "performance evaluation",
    "financial mathematics",
]

SPECIFIC_KEYWORDS = [
    "Kelly criterion",
    "deflated Sharpe ratio",
    "probability of backtest overfitting",
    "Hansen superior predictive ability",
    "White reality check",
    "Ledoit-Wolf shrinkage",
    "capital gains tax",
    "wash sale rule",
    "volatility targeting",
    "exchange-traded funds",
]

SUBJECT_HEADINGS = (
    "Portfolio management — Personal finance; "
    "Portfolio performance — Evaluation; "
    "Machine learning — Applications; "
    "Bayesian analysis; "
    "Kelly criterion; "
    "Risk management; "
    "Capital gains tax; "
    "Mutual funds — United States; "
    "Exchange-traded funds; "
    "Saving and investing; "
    "Risk."
)


def pct(value: float, places: int = 2) -> str:
    return f"{value * 100:.{places}f} per cent"


def main() -> int:
    data = json.loads(NUMBERS.read_text(encoding="utf-8"))
    summaries = data["summaries"]
    family = data["family_summaries"]
    selection = data["selection"]
    meta = data["meta"]
    window = data["window_summary"]

    system = summaries["System, tax exempt"]
    taxable = summaries["System, taxable"]
    index = summaries["SPY buy and hold"]
    core = family["no tilt, no governor"]
    tilt = family["tilt only"]
    governor = family["governor only"]

    tilt_gap = (tilt["cagr"] - core["cagr"]) * 100
    governor_gap = (governor["cagr"] - core["cagr"]) * 100
    drawdown_gain = (core["max_drawdown"] - governor["max_drawdown"]) * 100

    paragraph_one = (
        "**This book builds one complete investment system for a single "
        f"investor and runs it over {meta['system_months']} months of history, "
        "net of trading costs and of the capital gains tax, and reports what "
        "it is actually worth.** The universe is sixty-one equity and bond "
        "exchange-traded funds; the strategic allocation is a stated policy "
        "that the model is never allowed to set; the machine's job is to "
        "estimate the posterior over the edge and to size the position from "
        f"it. The system earns {pct(system['cagr'])} a year with a Sharpe "
        f"ratio of {system['sharpe']:.3f} and a worst drawdown of "
        f"{pct(system['max_drawdown'])}, against "
        f"{pct(index['cagr'])} and {pct(index['max_drawdown'])} for the "
        f"index bought and held, and {pct(taxable['cagr'])} in the same "
        "portfolio held in a taxable account."
    )

    paragraph_two = (
        "**The result is a null result, and the book reports it as one: the "
        "layer that forecasts returns and tilts the portfolio inside its "
        f"sleeves contributes about {tilt_gap:+.2f} percentage points a year, "
        "and the layer that slows the portfolio down when volatility rises "
        "and brakes it when the drawdown deepens contributes about "
        f"{governor_gap:+.2f} percentage points a year and about "
        f"{drawdown_gain:.2f} points of drawdown.** The protection is "
        "positive in every one of the "
        f"{window['n_windows']} starting windows the book tests. The test "
        "that chose the system is reported too, and it is unflattering: the "
        f"probability of backtest overfitting is {selection['pbo']['pbo']:.3f}, "
        "which is worse than a coin flip, and the average correlation "
        "between the members of the trial family is "
        f"{selection['average_correlation_between_trials']:.5f}, so the "
        "twenty-five members are worth about "
        f"{selection['implied_independent_trials']:.2f} independent trials "
        "rather than twenty-five. **Every number is produced by one "
        "generator and re-derived at build time, the Python package, the "
        "data cache, the source ledger and the verification gates are "
        "published with the book, and the arithmetic can therefore be "
        "checked, an assumption changed and the argument re-run.**"
    )

    keywords = GENERIC_KEYWORDS + SPECIFIC_KEYWORDS
    assert len(keywords) == 20, "the house format is twenty keywords"

    text = f"""Title: {TITLE}

Creator: {CREATOR}
Publisher: {PUBLISHER}
Edition: {EDITION}
License: {LICENSE}
DOI: {DOI}

{paragraph_one}

{paragraph_two}

Subject headings: {SUBJECT_HEADINGS}

Keywords: {", ".join(keywords[:10])}; {", ".join(keywords[10:])}

Related identifiers:
- Software: https://github.com/lordxmen2k/The-Patient-Quant

Language: English

Date of publication: 2026

Version: v1.0
"""

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
