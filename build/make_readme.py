#!/usr/bin/env python3
"""Write the repository README for "The Patient Quant".

The README quotes the book's headline numbers, so it is generated from
``build/numbers.json`` rather than typed by hand.  The same rule that keeps
a number out of the manuscript unless the generator produced it keeps it
out of the repository's front page.

    python3 build/make_readme.py

Writes README.md in the root of the repository.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
NUMBERS = ROOT / "build" / "numbers.json"
OUT = ROOT / "README.md"

TITLE = ("The Patient Quant: Building a Machine-Learned Portfolio for One "
         "Investor, and Measuring What It Is Actually Worth")
DOI = "10.5281/zenodo.24527200"


def pct(value: float, places: int = 2) -> str:
    return f"{value * 100:.{places}f} per cent"


def main() -> int:
    data = json.loads(NUMBERS.read_text(encoding="utf-8"))
    summaries = data["summaries"]
    family = data["family_summaries"]
    selection = data["selection"]
    meta = data["meta"]
    terms = data["terms"]
    window = data["window_summary"]
    frictions = data["frictions"]

    system = summaries["System, tax exempt"]
    taxable = summaries["System, taxable"]
    index = summaries["SPY buy and hold"]
    core = family["no tilt, no governor"]
    tilt = family["tilt only"]
    governor = family["governor only"]

    tilt_gap = (tilt["cagr"] - core["cagr"]) * 100
    governor_gap = (governor["cagr"] - core["cagr"]) * 100
    drawdown_gain = (core["max_drawdown"] - governor["max_drawdown"]) * 100

    pbo = selection["pbo"]["pbo"]
    if pbo > 0.5:
        pbo_reading = ("which is worse than a coin flip: the procedure of "
                       "selecting among the family's members is actively "
                       "unreliable on this evidence")
    else:
        pbo_reading = ("which is a coin flip: the procedure of selecting "
                       "among the family's members is no better than chance, "
                       "and no worse either")

    text = f"""# {TITLE}

Gerald Enrique Nelson Mc Kenzie — Tico Internet LLC
2026 First Edition · Apache License 2.0 · DOI {DOI}

A PhD-level reference on what a machine-learned portfolio is actually worth
to a single investor managing their own money — and on the uncomfortable
result that the prediction layer earns nothing, that the protection layer
earns almost everything, and that the procedure which chose the system
cannot be shown to be better than chance.

## Contents

```
source/          the LaTeX manuscript
  preamble.tex   identity, design tokens, page furniture, table column types
  main.tex       the document spine: covers, front matter, body, back matter
  frontmatter.tex, body.tex, backmatter.tex
  chapters/      ch01.tex .. ch38.tex
  cover_front.tex, cover_back.tex
  generated_numbers.tex       the headline macros, written by the generator
  generated_appendix_*.tex    the appendix tables
patientquant/    the Python package the book is built on
scripts/         fetch_data.py, run_system.py, build_ledger.py
build/           the figure, cover, number and verification scripts
data/cache/      the cached price history, with its checksums
research/        the source ledger and the registry records behind it
figures/         the cover plates and the book's eighteen figures
output/          the finished PDF
tests/           the unit tests for the package
```

## Building

```sh
./build.sh
```

Requires XeLaTeX, `makeindex`, Python 3 with numpy, pandas and matplotlib, and
the fonts in `fonts/`. The build runs the system, regenerates every macro,
every figure and every appendix table from the cached data, typesets the
book, and then runs the verification gates against the finished PDF.

```sh
SKIP_RUN=1 ./build.sh    # reuse build/numbers.json instead of re-running
COVERS=1 ./build.sh      # rebuild the cover plates
```

The covers are built separately, because they are separate documents:

```sh
build/make_covers.sh
```

The package's own tests:

```sh
python3 -m unittest discover -s tests
```

## The argument

This book builds one complete investment system for a single investor and runs
it over {meta['system_months']} months of history, net of trading costs and of the
capital gains tax, and reports what it is actually worth. The universe is
{terms['n_funds']} equity and bond exchange-traded funds. The strategic allocation is a
stated policy the model is never allowed to set; the machine's job is to
estimate the posterior over the edge and to size the position from it.

The system earns {pct(system['cagr'])} a year with a Sharpe ratio of
{system['sharpe']:.3f} and a worst drawdown of {pct(system['max_drawdown'])}, against
{pct(index['cagr'])} and {pct(index['max_drawdown'])} for the index bought and held, and
{pct(taxable['cagr'])} for the same portfolio in a taxable account. It turns
out to be worth *less* than the index and *more* than the index in the only
respect that matters to the investor who has to live with it.

**The result is a null result, and the book reports it as one.** The layer
that forecasts returns and tilts the portfolio inside its sleeves contributes
about {tilt_gap:+.2f} percentage points a year. The layer that slows the portfolio down
when volatility rises and brakes it when the drawdown deepens contributes about
{governor_gap:+.2f} percentage points a year and about {drawdown_gain:.2f} points of drawdown, and it does
so in every one of the {window['n_windows']} starting windows the book tests. The prediction
layer is not what makes this work, and the book says so.

The selection tests are reported with the same honesty. The Deflated Sharpe
Ratio is {selection['dsr']:.4f}; the Hansen superior predictive ability test and the
White reality check both return a p-value of one; the Newey--West alpha
against the index is {selection['alpha_vs_benchmark']['tstat']:.2f} with a p-value of
{selection['alpha_vs_benchmark']['pvalue']:.3f}. The probability of backtest overfitting is
{pbo:.3f}, {pbo_reading}. The average correlation between the members of the
trial family is {selection['average_correlation_between_trials']:.5f}, so the
{meta['n_trial_family']} members are worth about {selection['implied_independent_trials']:.2f} independent
trials rather than {meta['n_trial_family']}. The trial family was declared before the
tests were run, and the family includes the ablations that turn the machine
off, so a favourable reading was always available to be checked.

**Every number is produced by one generator and re-derived at build time.**
The Python package, the data cache, the source ledger and the verification
gates are published with the book, so the arithmetic can be checked, an
assumption changed, and the argument re-run.

## The numbers

Every headline number in the book is written by `build/make_numbers_tex.py`
from `build/numbers.json`, which `scripts/run_system.py` writes from the
cached price history. Nothing is transcribed. The macros are checked against
the canonical data and then against the rendered PDF:

```sh
python3 build/check_numbers.py
```

## Verification

`build/verify.py` runs twenty gates against the finished PDF: that it
builds and is a book, with no errors in the typesetting log; that every
figure named in the source exists; that every figure is used exactly once
and no label is defined twice; that no label in any figure overlaps a
curve, a marker, a legend or another label; that every number macro used
in the text is defined and is a legal control word; that every macro
agrees with the canonical data; that every macro's value is on the page;
that every source in the ledger is cited; that the ledger resolves against
its registry; that the page and word counts in the colophon are right; that
no template residue survives; that the copyright string is exact; that no
secret is in the repository; that the house fonts are embedded; that the
PDF is structurally sound; that the data cache matches its checksums; that
the panel rebuilds from the cache; that there is no look-ahead in the
training set; and that every heading argument in the source is closed.

## Data

The price history comes from one vendor's public chart endpoint, is cached
under `data/cache/` with a SHA-256 for every series, and is rebuilt into
monthly returns from the price and dividend components rather than from the
adjusted close. Months before a fund existed are masked rather than
zero-filled, because a zero return makes a new fund look as though it had
neither volatility nor beta.

## Licence

Apache License 2.0. See `LICENSE`.
"""

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
