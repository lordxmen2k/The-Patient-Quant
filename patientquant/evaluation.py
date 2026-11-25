"""Measuring a record without flattering it.

The functions here are the book's contract with the reader.  A strategy's
record is reported as a *distribution* with an explicit uncertainty, not as a
single Sharpe ratio.

Three corrections matter enough to be built in rather than mentioned in a
footnote.

**Geometric versus arithmetic growth.**  A portfolio's wealth grows at the
geometric mean of its returns, not the arithmetic mean, and the gap is
approximately half the variance.  The book reports the geometric mean and
says so wherever a number appears.

**The Sharpe ratio is an estimate.**  Annualising a monthly Sharpe ratio by
multiplying by the square root of twelve is correct only when the returns
are serially uncorrelated.  Lo (2002) derives the correct multiplier, which
depends on the autocorrelations of the return series.  A strategy with
positive autocorrelation -- a trend follower, for instance -- has its Sharpe
ratio overstated by the naive annualisation, sometimes by more than half.
The multiplier is computed here from the sample autocorrelations.

**Inference needs the right standard error.**  Monthly returns are not
independent, so the t-statistic on an alpha uses a Newey-West standard error
with a lag length chosen by rule rather than by taste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------
def geometric_mean(returns: pd.Series, periods: int = 12) -> float:
    """Annualised geometric mean return."""
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    total = float(np.prod(1.0 + clean.to_numpy(dtype=float)))
    if total <= 0:
        return float("nan")
    return total ** (periods / len(clean)) - 1.0


def arithmetic_mean(returns: pd.Series, periods: int = 12) -> float:
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    return float(clean.mean() * periods)


def cagr_from_returns(returns: pd.Series, periods: int = 12) -> float:
    """Compound annual growth rate, the same thing written more plainly."""
    return geometric_mean(returns, periods)


def volatility(returns: pd.Series, periods: int = 12) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Deepest peak-to-trough fall of the wealth curve."""
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    wealth = (1.0 + clean).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()
    wealth = (1.0 + clean).cumprod()
    return wealth / wealth.cummax() - 1.0


def time_to_recovery(returns: pd.Series) -> int:
    """Months from the deepest drawdown to the day the old peak was regained."""
    clean = returns.dropna()
    if len(clean) == 0:
        return 0
    wealth = (1.0 + clean).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    trough_pos = int(np.argmin(drawdown.to_numpy()))
    peak_value = float(peak.iloc[trough_pos])
    after = wealth.iloc[trough_pos:]
    recovered = np.where(after.to_numpy() >= peak_value)[0]
    if len(recovered) == 0:
        return -1
    return int(recovered[0])


def worst_window(returns: pd.Series, months: int = 12) -> float:
    """Worst rolling ``months``-month total return."""
    clean = returns.dropna()
    if len(clean) < months:
        return float("nan")
    rolled = (1.0 + clean).rolling(months).apply(np.prod, raw=True) - 1.0
    return float(rolled.min())


def downside_deviation(returns: pd.Series, periods: int = 12) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    downside = clean[clean < 0]
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(downside.to_numpy()))) * np.sqrt(periods))


# ---------------------------------------------------------------------------
# Sharpe ratio and the Lo correction
# ---------------------------------------------------------------------------
def lo_scale_factor(q: int, autocorrelations) -> float:
    """Lo's (2002) multiplier for aggregating a one-period Sharpe ratio.

    For a stationary return series with autocorrelations ``rho_1 ... rho_{q-1}``
    the correct multiplier is

        eta(q) = sqrt(q) / sqrt(1 + (2/q) sum_{k=1}^{q-1} (q - k) rho_k)

    Under an AR(1) process with correlation ``rho`` this reduces to Lo's
    equation 22::

        eta(q) = sqrt(q) [1 + (2 rho / (1 - rho))
                          (1 - (1 - rho^q) / (q (1 - rho)))]^(-1/2)

    With no serial correlation it is exactly ``sqrt(q)``, which is why the
    familiar annualisation is usually right and sometimes very wrong.
    """
    total = 0.0
    for k, rho_k in enumerate(autocorrelations, start=1):
        if k >= q:
            break
        total += (q - k) * float(rho_k)
    return float(np.sqrt(q) / np.sqrt(max(1.0 + 2.0 * total / q, 1e-12)))


def autocorrelations(returns: pd.Series, max_lag: int = 11) -> list[float]:
    """Sample autocorrelations of the return series up to ``max_lag``."""
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) < 3:
        return []
    centred = clean - clean.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 0:
        return []
    out = []
    for lag in range(1, max_lag + 1):
        if len(clean) <= lag:
            break
        out.append(float(np.dot(centred[:-lag], centred[lag:]) / denominator))
    return out


def sharpe_ratio(returns: pd.Series, periods: int = 12,
                 correct_serial: bool = True) -> dict:
    """Sharpe ratio of a return series, with and without the Lo correction.

    Returns a dictionary with the naive annualised Sharpe, the Lo-corrected
    annualised Sharpe, the scale factors involved, and the first-order
    autocorrelation of the monthly returns.
    """
    clean = returns.dropna()
    if len(clean) < 3:
        return {"sharpe": float("nan"), "sharpe_naive": float("nan"),
                "scale_factor": float("nan"), "rho1": float("nan"),
                "observations": len(clean)}

    mean = float(clean.mean())
    sd = float(clean.std(ddof=1))
    monthly_sharpe = mean / sd if sd > 0 else float("nan")

    rhos = autocorrelations(clean, max_lag=periods - 1)
    if correct_serial and rhos:
        factor = lo_scale_factor(periods, rhos)
    else:
        factor = float(np.sqrt(periods))
    return {
        "sharpe": monthly_sharpe * factor,
        "sharpe_naive": monthly_sharpe * float(np.sqrt(periods)),
        "scale_factor": factor,
        "rho1": rhos[0] if rhos else float("nan"),
        "observations": len(clean),
    }


def sortino_ratio(returns: pd.Series, periods: int = 12) -> float:
    clean = returns.dropna()
    if len(clean) < 3:
        return float("nan")
    downside = downside_deviation(clean, periods)
    if not np.isfinite(downside) or downside <= 0:
        return float("nan")
    return float(clean.mean() * periods / downside)


def calmar_ratio(returns: pd.Series) -> float:
    growth = geometric_mean(returns)
    drawdown = max_drawdown(returns)
    if not np.isfinite(drawdown) or drawdown == 0:
        return float("nan")
    return float(growth / abs(drawdown))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def newey_west_mean_tstat(x: pd.Series, y: pd.Series, lags: int = 6) -> dict:
    """HAC t-statistic for the mean of ``x`` (an excess return) against ``y``.

    The lag length defaults to ``floor(4 (T/100)^(2/9))`` capped at the
    value given, which is the rule of thumb from Andrews, Stock and Sun
    (2019) for the optimal bandwidth.
    """
    frame = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(frame) < 10:
        return {"mean": float("nan"), "tstat": float("nan"), "pvalue": float("nan"),
                "hac_se": float("nan"), "n": len(frame)}

    xv = frame["x"].to_numpy(dtype=float)
    yv = frame["y"].to_numpy(dtype=float)
    n = len(xv)

    # Regress x on a constant and y; the intercept is the alpha.
    X = np.column_stack([np.ones(n), yv])
    beta, *_ = np.linalg.lstsq(X, xv, rcond=None)
    residuals = xv - X @ beta
    alpha = float(beta[0])

    bandwidth = int(min(lags, max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))))
    meat = np.zeros((2, 2))
    for i in range(n):
        u = residuals[i] * X[i]
        meat += np.outer(u, u)
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = np.zeros((2, 2))
        for i in range(lag, n):
            gamma += np.outer(residuals[i] * X[i], residuals[i - lag] * X[i - lag])
        meat += weight * (gamma + gamma.T)

    try:
        bread = np.linalg.inv(X.T @ X)
        variance = bread @ meat @ bread
        se = float(np.sqrt(max(variance[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        se = float("nan")

    tstat = alpha / se if se and np.isfinite(se) and se > 0 else float("nan")
    from .stats import norm_cdf
    pvalue = 2.0 * (1.0 - norm_cdf(abs(tstat))) if np.isfinite(tstat) else float("nan")
    return {"mean": alpha, "tstat": tstat, "pvalue": pvalue,
            "hac_se": se, "n": n, "bandwidth": bandwidth}


def summarise(returns: pd.Series, label: str = "") -> dict:
    """The full reporting card for a return series."""
    sharpe = sharpe_ratio(returns)
    growth = geometric_mean(returns)
    drawdown = max_drawdown(returns)
    return {
        "label": label,
        "observations": int(returns.dropna().shape[0]),
        "cagr": growth,
        "arithmetic_mean": arithmetic_mean(returns),
        "volatility": volatility(returns),
        "sharpe": sharpe["sharpe"],
        "sharpe_naive": sharpe["sharpe_naive"],
        "lo_scale_factor": sharpe["scale_factor"],
        "rho1": sharpe["rho1"],
        "sortino": sortino_ratio(returns),
        "max_drawdown": drawdown,
        "calmar": calmar_ratio(returns),
        "worst_12m": worst_window(returns),
        "time_to_recovery_months": time_to_recovery(returns),
        "total_return": float((1.0 + returns.dropna()).prod() - 1.0),
    }
