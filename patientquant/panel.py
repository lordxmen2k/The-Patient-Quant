"""The monthly panel: prices, total returns, dividends, and fund terms.

Everything downstream reads from this module and nothing else.  The panel is
built from the cached Yahoo responses and nothing is imputed, filled from a
vendor's library, or spliced from another source.

Three conventions matter and are stated here rather than buried.

**Month ends.**  Prices are sampled at the last observation each fund has in a
calendar month.  Funds listed on different calendars (a foreign equity ETF
follows its home market) do not all have a price on the same day; the panel
therefore samples each fund on its own month end and *then* aligns.

**Missing months are zero returns, not gaps.**  If a fund did not trade during
a month, its last known price is carried forward and its return for that month
is zero.  Forward-filling the *price* rather than the return is the
conservative choice: it cannot manufacture a return.

**Adjusted close is a return series, not a price series.**  The level of
Yahoo's adjusted close is an artefact of every dividend and split that
happened afterwards; only the ratios are meaningful.  The book uses ratios
only.  The expense ratio of each fund is therefore *already inside* these
returns and must not be charged a second time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as pq_data
from . import fundamentals as pq_fundamentals
from .universe import TICKERS

REPO_ROOT = Path(__file__).resolve().parent.parent
FUNDAMENTALS_DIR = REPO_ROOT / "data" / "cache" / "fundamentals"


def load_daily_prices(tickers=None) -> dict[str, pd.DataFrame]:
    """Adjusted close *and* raw close for each fund, straight from the raw cache.

    Both are needed and they are not interchangeable.  The adjusted close
    carries dividends forward, so its ratios are total returns; but its
    *level* is deflated by every dividend paid afterwards, so it cannot be
    used as a transaction price or as the denominator of a dividend yield.
    The raw close is the price an investor actually trades at, and it is what
    the tax layer needs.
    """
    tickers = tickers or TICKERS
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = pq_data.CACHE_DIR / f"{ticker}.json"
        if not path.exists():
            continue
        frame = pq_data.to_frame(json.loads(path.read_text()))
        out[ticker] = frame[["adjclose", "close"]].dropna(subset=["adjclose"])
    return out


def load_daily_volume(tickers=None) -> dict[str, pd.Series]:
    """Share volume for each fund, from the same cached responses."""
    tickers = tickers or TICKERS
    out: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = pq_data.CACHE_DIR / f"{ticker}.json"
        if not path.exists():
            continue
        frame = pq_data.to_frame(json.loads(path.read_text()))
        out[ticker] = frame["volume"].dropna()
    return out


def load_dividend_cash(tickers=None) -> dict[str, pd.Series]:
    """Dividend cash actually paid, indexed by ex-date."""
    tickers = tickers or TICKERS
    out: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = pq_data.CACHE_DIR / f"{ticker}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        events = ((payload.get("chart") or {}).get("result") or [{}])[0]
        dividends = (events.get("events") or {}).get("dividends") or {}
        if not dividends:
            out[ticker] = pd.Series(dtype=float)
            continue
        index = pd.to_datetime(
            [int(v["date"]) for v in dividends.values()], unit="s", utc=True
        ).tz_convert(None).normalize()
        amounts = pd.Series(
            [float(v["amount"]) for v in dividends.values()], index=index
        ).sort_index()
        out[ticker] = amounts
    return out


def monthly_panel(tickers=None) -> dict[str, pd.DataFrame]:
    """Monthly prices, total returns and dividend yields on a common calendar.

    Returns a dictionary with the keys ``prices`` (month-end adjusted close),
    ``returns`` (monthly total return), ``dividends`` (cash paid in the
    month) and ``dividend_yield`` (cash as a fraction of the prior month's
    price).
    """
    tickers = tickers or TICKERS
    prices_daily = load_daily_prices(tickers)
    dividends_daily = load_dividend_cash(tickers)

    volumes_daily = load_daily_volume(tickers)

    monthly_adj: dict[str, pd.Series] = {}
    monthly_raw: dict[str, pd.Series] = {}
    monthly_dividends: dict[str, pd.Series] = {}
    monthly_volume: dict[str, pd.Series] = {}
    for ticker, frame in prices_daily.items():
        monthly_adj[ticker] = frame["adjclose"].resample("ME").last()
        monthly_raw[ticker] = frame["close"].resample("ME").last()
        div = dividends_daily.get(ticker, pd.Series(dtype=float))
        monthly_dividends[ticker] = (
            div.resample("ME").sum() if len(div) else pd.Series(dtype=float)
        )
        volume = volumes_daily.get(ticker, pd.Series(dtype=float))
        monthly_volume[ticker] = volume.resample("ME").sum() if len(volume) else pd.Series(dtype=float)

    adj_frame = pd.DataFrame(monthly_adj).sort_index()
    raw_frame = pd.DataFrame(monthly_raw).sort_index()
    dividend_frame = pd.DataFrame(monthly_dividends).sort_index()
    dividend_frame = dividend_frame.reindex_like(adj_frame).fillna(0.0)
    volume_frame = pd.DataFrame(monthly_volume).sort_index()
    volume_frame = volume_frame.reindex_like(adj_frame).fillna(0.0)

    # Carry each fund's last observed price into months where it did not
    # trade, so that a silent month earns zero rather than becoming a gap.
    filled_adj = adj_frame.ffill()
    filled_raw = raw_frame.ffill()
    returns = filled_adj.pct_change()

    # Dividends are cash paid on the raw price, not on the deflated
    # adjusted price.  Using the adjusted close here would overstate every
    # historical yield by decades of subsequent distributions.
    prior_raw = filled_raw.shift(1)
    dividend_yield = dividend_frame / prior_raw.replace(0.0, np.nan)

    # The price return alone, for splitting a portfolio's gross return into
    # its price and dividend components.
    price_returns = filled_raw.pct_change()

    # Total return built from components rather than read off the adjusted
    # close.  The adjusted close is a perfectly good total-return series, but
    # its dividend factor is applied on the vendor's own convention, which
    # does not line up exactly with the month in which the cash was paid.
    # Building the return from the cash actually paid keeps the gross
    # simulation and the taxable simulation on one set of numbers.
    total_returns = (1.0 + price_returns.fillna(0.0)) * (
        1.0 + dividend_yield.fillna(0.0)
    ) - 1.0

    # Before a fund lists, it has no return at all.  Carrying a zero back
    # from the start of the panel would give every future listing a
    # history of zero returns, which is not a neutral convention: it makes
    # a new fund's realised volatility look like zero and its beta look
    # like zero, and both of those are inputs to the model.  The series is
    # therefore masked to missing before inception.  Months *after*
    # inception in which the fund did not trade remain zero returns, which
    # is the convention stated at the top of this module.
    live = filled_raw.notna()
    total_returns = total_returns.where(live)
    price_returns = price_returns.where(live)

    return {
        "prices": adj_frame,
        "raw_prices": raw_frame,
        "returns": total_returns,
        "adjusted_returns": returns,
        "price_returns": price_returns,
        "dividends": dividend_frame,
        "dividend_yield": dividend_yield,
        "volume": volume_frame,
    }


def fund_terms(tickers=None) -> pd.DataFrame:
    """Expense ratio, inception date, net assets, turnover and yield."""
    tickers = tickers or TICKERS
    rows = []
    for ticker in tickers:
        path = FUNDAMENTALS_DIR / f"{ticker}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows.append(pq_fundamentals._parse(ticker, payload).as_row())
    frame = pd.DataFrame(rows).set_index("ticker")
    frame["inception_date"] = pd.to_datetime(frame["inception_date"], errors="coerce")
    return frame.sort_index()


def universe_start(terms: pd.DataFrame, panel: dict[str, pd.DataFrame]) -> pd.Series:
    """First month each fund has a usable return, for reporting the live window."""
    returns = panel["returns"]
    firsts = {}
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        if len(series):
            firsts[ticker] = series.index[0]
    return pd.Series(firsts, name="first_return_month").sort_values()


def availability(terms: pd.DataFrame, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join fund terms with the first month of live return history."""
    returns = panel["returns"]
    firsts = pd.Series(
        {t: returns[t].dropna().index[0] for t in returns.columns
         if len(returns[t].dropna())},
        name="first_return_month",
    )
    joined = terms.join(firsts)
    joined["live_years"] = (
        (returns.index[-1] - joined["first_return_month"]).dt.days / 365.25
    )
    return joined
