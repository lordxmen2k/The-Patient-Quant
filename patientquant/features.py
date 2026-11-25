"""Feature construction: what the model is allowed to see.

Every feature here is computed from information available at the close of the
month on which the portfolio is formed.  Three rules are enforced:

1.  No feature may use a price from after the formation date.  Momentum is
    measured over a window that *ends* at the formation date and *excludes*
    the most recent month, because the most recent month is the one whose
    price the market has already reacted to.
2.  No feature may use data that is revised.  Nothing here comes from a
    macroeconomic series that gets revised, and nothing comes from a fund's
    current assets under management.  Liquidity is measured from traded
    volume, which is contemporaneous and final.
3.  No feature may be scaled by a statistic computed over the whole sample.
    Standardisation is done per date, using only the cross-section that
    exists on that date.

The feature set is deliberately price-based.  The book draws its data from a
single source, and a system that could see book value, earnings or
profitability would have a different -- and larger -- claim to make.  What
this feature set can support is an honest claim about the *precision* of a
forward-return estimate, not about the presence of a fundamental edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mom_12_1",
    "mom_6_1",
    "mom_36",
    "rev_1",
    "vol_12",
    "vol_ratio",
    "trend_12",
    "drawdown_12",
    "beta_36",
    "corr_36",
    "illiq_12",
    "turnover_proxy",
]

FEATURE_LABELS = {
    "mom_12_1": "12-month momentum (skip most recent month)",
    "mom_6_1": "6-month momentum (skip most recent month)",
    "mom_36": "36-month momentum",
    "rev_1": "one-month reversal",
    "vol_12": "12-month realised volatility",
    "vol_ratio": "6-to-12 month volatility ratio",
    "trend_12": "share of positive months over 12 months",
    "drawdown_12": "distance from the 12-month high",
    "beta_36": "36-month beta to the S&P 500",
    "corr_36": "36-month correlation to the S&P 500",
    "illiq_12": "12-month Amihud illiquidity",
    "turnover_proxy": "share of the cross-section traded in the past month",
}

MIN_MONTHS = 13  # a feature window of 12 months plus the skip month


def _rolling_sum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window, min_periods=window).sum()


def build_features(panel: dict[str, pd.DataFrame], market: str = "SPY"
                   ) -> dict[str, pd.DataFrame]:
    """Return one (dates x tickers) frame per feature, unscaled.

    Every feature is strictly backward-looking: the value on row ``t`` uses
    only data up to and including month ``t``.
    """
    prices = panel["raw_prices"]
    returns = panel["returns"]
    volume = panel.get("volume")

    out: dict[str, pd.DataFrame] = {}

    # --- Momentum -------------------------------------------------------
    # Cumulative return over [t-12, t-1] and [t-6, t-1]: the trailing window
    # that ends one month before the formation date.
    cum12 = prices / prices.shift(12) - 1.0
    cum6 = prices / prices.shift(6) - 1.0
    cum36 = prices / prices.shift(36) - 1.0
    out["mom_12_1"] = cum12.shift(1) - 1.0
    out["mom_6_1"] = cum6.shift(1) - 1.0
    out["mom_36"] = cum36 - 1.0

    # --- Reversal -------------------------------------------------------
    out["rev_1"] = -(returns.shift(1))

    # --- Volatility -----------------------------------------------------
    monthly_vol = returns.rolling(6, min_periods=6).std() * np.sqrt(12)
    out["vol_12"] = returns.rolling(12, min_periods=12).std() * np.sqrt(12)
    out["vol_ratio"] = monthly_vol / out["vol_12"] - 1.0

    # --- Trend ----------------------------------------------------------
    out["trend_12"] = (returns > 0).rolling(12, min_periods=12).mean() - 0.5

    # --- Drawdown state -------------------------------------------------
    rolling_high = prices.rolling(12, min_periods=12).max()
    out["drawdown_12"] = prices / rolling_high - 1.0

    # --- Market exposure ------------------------------------------------
    market_returns = returns[market]
    beta_window, corr_window = 36, 36
    beta = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    corr = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ticker in prices.columns:
        asset = returns[ticker]
        cov = asset.rolling(beta_window, min_periods=beta_window).cov(market_returns)
        var = market_returns.rolling(beta_window, min_periods=beta_window).var()
        beta[ticker] = cov / var.replace(0.0, np.nan)
        corr[ticker] = asset.rolling(corr_window, min_periods=corr_window).corr(market_returns)
    out["beta_36"] = beta
    out["corr_36"] = corr

    # --- Liquidity ------------------------------------------------------
    # Amihud (2002): the average absolute return per unit of dollar volume.
    # High values mean thin markets: the price moves a lot for little money.
    if volume is not None:
        dollar_volume = volume * prices
        illiq = (returns.abs() / dollar_volume.replace(0.0, np.nan)).rolling(
            12, min_periods=12
        ).mean()
        out["illiq_12"] = illiq * 1e6
        # Share of the month's dollar volume accounted for by this fund: a
        # crude turnover measure that needs no fund-level data.
        out["turnover_proxy"] = dollar_volume.div(
            dollar_volume.sum(axis=1), axis=0
        )
    else:
        empty = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        out["illiq_12"] = empty
        out["turnover_proxy"] = empty

    # A fund that has not listed yet has no feature values.  Some of the
    # constructions above would otherwise manufacture one --- the trend
    # count treats a missing month as a losing month, for instance --- so
    # every feature is masked by the presence of a price.
    live = prices.notna()
    for name, frame in out.items():
        out[name] = frame.where(live)

    return out


def cross_sectional_scale(features: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Standardise each feature within each date.

    Standardisation uses only the cross-section that exists on that date, so
    no future information leaks into the scale.
    """
    scaled = {}
    for name, frame in features.items():
        means = frame.mean(axis=1)
        stds = frame.std(axis=1, ddof=0).where(lambda s: s > 1e-12)
        scaled[name] = frame.sub(means, axis=0).div(stds, axis=0)
    return scaled


def build_target(panel: dict[str, pd.DataFrame], horizon: int = 3) -> pd.DataFrame:
    """Forward total return over ``horizon`` months, formed at each date.

    The target on row ``t`` is the total return from ``t`` to ``t + horizon``.
    Because the target is unknown until ``t + horizon``, the training set at
    any date must exclude the final ``horizon`` rows.
    """
    returns = panel["returns"]
    forward = (1.0 + returns).rolling(horizon, min_periods=horizon).apply(
        np.prod, raw=True
    ) - 1.0
    # The rolling product above looks forward; shift it back so that row t
    # holds the return realised between t and t + horizon.
    forward = forward.shift(-horizon)
    return forward
