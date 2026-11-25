"""The risk layer: two rules that can be written down and checked.

A long-horizon portfolio that protects its principal needs rules that act
*before* the loss rather than after it.  Two are used.

**The volatility governor.**  The total exposure to risky assets is scaled
so that the portfolio's estimated volatility approaches a target.  The
estimate uses a covariance matrix of sleeve returns and a trailing window of
monthly observations; the scaling is *damped* by half, and the scale is
applied gradually rather than instantly.

The damping is not timidity.  Moreira and Muir (2017) showed that
volatility-managed portfolios can beat their unmanaged counterparts
out of sample, and Cederburg, O'Doherty, Wang and Yan (2020) showed that
realistic out-of-sample versions earn *lower* certainty-equivalent returns
and Sharpe ratios, because the scaling regression is structurally unstable.
The book's answer is to damp the rule, and to report the undamped variant
beside it rather than quietly choosing the flattering one.

**The drawdown brake.**  When the portfolio's own trailing drawdown passes
a threshold, exposure to risky assets is cut.  This is not a stop-loss: it
does not sell on a price, it sells on the portfolio's own state, and it acts
once per month.  Its purpose is the one that matters for a private investor
holding money that has a future: the loss that causes an investor to quit is
the loss that arrives before the gain, and a rule that reduces exposure
during a drawdown is a rule that makes it easier to stay invested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(returns: pd.Series, window: int = 12) -> float:
    """Annualised volatility of a trailing window of monthly returns."""
    clean = returns.dropna()
    if len(clean) < 3:
        return float("nan")
    return float(clean.tail(window).std(ddof=1) * np.sqrt(12))


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Annualised volatility of a weighted portfolio."""
    tickers = [t for t in weights.index if t in covariance.columns]
    if not tickers:
        return float("nan")
    w = weights[tickers].to_numpy(dtype=float)
    cov = covariance.loc[tickers, tickers].to_numpy(dtype=float)
    variance = float(w @ cov @ w)
    return float(np.sqrt(max(variance, 0.0)))


def volatility_scale(current_vol: float, target_vol: float,
                     damping: float = 0.5) -> float:
    """The multiplier applied to risky exposure.

    ``damping`` of one reproduces the undamped rule; one half means the
    portfolio moves halfway towards its target each month.
    """
    if not np.isfinite(current_vol) or current_vol <= 0:
        return 1.0
    raw = target_vol / current_vol
    return float(np.clip(1.0 + damping * (raw - 1.0), 0.0, 2.0))


def drawdown_brake(drawdown: float, first_level: float = 0.15,
                   second_level: float = 0.25) -> float:
    """Multiplier applied to risky exposure as a function of drawdown.

    No brake below the first threshold, half exposure above it, a quarter
    above the second.
    """
    if not np.isfinite(drawdown):
        return 1.0
    if drawdown <= -second_level:
        return 0.25
    if drawdown <= -first_level:
        return 0.50
    return 1.0


def trailing_drawdown(equity: pd.Series) -> float:
    """Drawdown of a wealth series at its most recent point."""
    clean = equity.dropna()
    if len(clean) < 2:
        return 0.0
    peak = clean.cummax()
    return float((clean.iloc[-1] / peak.iloc[-1]) - 1.0)


def apply_governor(risky_weights: pd.Series, covariance: pd.DataFrame,
                   target_vol: float, damping: float = 0.5,
                   drawdown: float = 0.0,
                   first_level: float = 0.15,
                   second_level: float = 0.25) -> tuple[pd.Series, float, float]:
    """Scale risky weights by the governor and the brake.

    Returns ``(scaled_weights, volatility_multiplier, brake_multiplier)``.
    """
    vol = portfolio_volatility(risky_weights, covariance)
    scale = volatility_scale(vol, target_vol, damping)
    brake = drawdown_brake(drawdown, first_level, second_level)
    combined = scale * brake
    scaled = risky_weights * combined
    return scaled, scale, brake
