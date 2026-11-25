"""Portfolio construction: sizing an uncertain edge.

The rule used here is the Kelly criterion, applied to a *distribution* rather
than to a point estimate.

For a portfolio with weights ``w``, expected log growth under a normal
return distribution with mean ``mu`` and covariance ``Sigma`` is

    G(w) = w'mu - (gamma / 2) w'Sigma w

with ``gamma`` the investor's relative risk aversion.  Maximising gives the
familiar ``w* = (1/gamma) Sigma^-1 mu``.

The subtlety that matters is that ``mu`` is not known.  The model supplies a
posterior for it, with a covariance ``Psi`` of the estimated means.  Writing
``E[mu]`` for the posterior mean, expected growth is

    E[G(w)] = w'E[mu] - (gamma / 2) w'(Sigma + Psi) w

so the optimal weights are

    w* = (1/gamma) (Sigma + Psi)^-1 E[mu]

The estimation error enters the *denominator*, on the same footing as the
risk of the assets themselves.  A signal whose estimated mean has a standard
error as large as the edge itself therefore produces a small position, and a
signal with no estimated edge produces no position at all.  That is the
mechanism by which "protect the principal" and "grow the wealth" become the
same action rather than competing ones.

``gamma`` is set from the Kelly criterion's own consistency requirement: the
growth-optimal fraction already embeds the risk aversion implied by the
return distribution, and a full-Kelly investor using an estimated ``mu``
overbets.  The book therefore applies a fraction of Kelly, and reports the
sensitivity.

The covariance is shrunk (Ledoit & Wolf, 2004) because a sample covariance
from a few hundred observations in sixty dimensions is not an estimate of
anything useful, and the shrinkage intensity is chosen by the estimator
rather than by taste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Pairwise-complete sample covariance and the effective sample size.

    A fund that listed three months ago has no return for the thirty-three
    months before it.  Those months are missing, not zero: treating them as
    zero would give the fund an artificially small variance and an
    artificially small covariance with everything else, which is precisely
    the kind of error that makes a brand-new listing look like a low-risk
    asset.  This routine therefore computes each covariance from the
    observations the pair actually has, and returns the median pairwise
    count as the effective sample size.
    """
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="all")
    values = clean.to_numpy(dtype=float)
    n_rows, p = values.shape
    if p < 2:
        return pd.DataFrame(
            np.zeros((max(p, 1), max(p, 1))),
            index=clean.columns, columns=clean.columns,
        ), float(n_rows)

    mask = ~np.isnan(values)
    counts = mask.sum(axis=0).astype(float)
    sums = np.where(mask, values, 0.0).sum(axis=0)
    means = sums / np.maximum(counts, 1.0)
    deviations = np.where(mask, values - means, 0.0)
    paired = np.minimum(counts[:, None], counts[None, :])
    denominator = np.maximum(paired - 1.0, 1.0)
    covariance = (deviations.T @ deviations) / denominator
    np.fill_diagonal(covariance, np.maximum(np.diag(covariance), 0.0))

    if p > 1:
        upper = paired[np.triu_indices(p, 1)]
        n_eff = float(np.median(upper[upper > 0])) if np.any(upper > 0) else float(n_rows)
    else:
        n_eff = float(n_rows)
    return pd.DataFrame(covariance, index=clean.columns, columns=clean.columns), n_eff


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Shrink the sample covariance toward a scaled identity.

    Returns ``(covariance, shrinkage_intensity)``.  The intensity is
    estimated from the data: it grows when the sample covariance is
    estimated from few observations relative to its dimension, and shrinks
    towards zero when the sample is long.
    """
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="all")
    values = clean.to_numpy(dtype=float)
    n, p = values.shape
    if n < 3 or p < 2:
        cov = np.cov(values, rowvar=False) if n > 1 else np.eye(1)
        return pd.DataFrame(cov, index=clean.columns, columns=clean.columns), 0.0

    sample, n_eff = pairwise_covariance(clean)
    sample = sample.to_numpy(dtype=float)
    n = n_eff if n_eff >= 3 else n
    mu = np.trace(sample) / p

    # pi: the total squared deviation of the off-diagonal elements of the
    # sample covariance from zero.
    off = sample - np.diag(np.diag(sample))
    pi_hat = float((off ** 2).sum() / p)

    # rho: the squared error one would expect from using the sample
    # covariance itself, estimated under the null of independent draws.
    tr_sq = float((sample ** 2).sum())
    rho_hat = n * (tr_sq / p - mu ** 2) + pi_hat
    if rho_hat <= 0:
        intensity = 0.0
    else:
        intensity = float(np.clip(pi_hat / rho_hat, 0.0, 1.0))

    target = mu * np.eye(p)
    shrunk = (1.0 - intensity) * sample + intensity * target
    return pd.DataFrame(shrunk, index=clean.columns, columns=clean.columns), intensity


def kelly_weights(mu: np.ndarray, sigma: np.DataFrame, psi: np.ndarray,
                  gamma: float = 1.0, fraction: float = 0.25,
                  ridge: float = 1e-8) -> np.ndarray:
    """Optimal weights for an uncertain mean, scaled by a fraction of Kelly.

    ``mu``     posterior predictive mean of the forward return, per asset
    ``sigma``  shrunk covariance of asset returns
    ``psi``    diagonal of the posterior covariance of the estimated means
    ``gamma``  relative risk aversion
    ``fraction`` the fraction of full Kelly to use
    """
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    if n == 0:
        return np.zeros(0)

    total = sigma.to_numpy(dtype=float).copy()
    total[np.diag_indices(n)] += np.asarray(psi, dtype=float)
    total[np.diag_indices(n)] += ridge
    total[np.diag_indices(n)] += ridge * float(np.trace(total)) / n

    try:
        raw = np.linalg.solve(total, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(total, mu, rcond=None)[0]

    # The full-Kelly solution is unbounded in sign and magnitude; scaling by
    # the risk aversion and by a fraction of Kelly is what makes it usable.
    raw = raw / max(gamma, 1e-6) * fraction

    # A tiny amount of shrinkage on the raw solution keeps a single lucky
    # feature from producing a corner portfolio.
    limit = 0.25
    raw = np.clip(raw, -limit, limit)
    return raw


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n) if n else np.zeros(0)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical risk parity weights (Lopez de Prado, 2016).

    Included as a comparator, not as the system's rule: HRP is a robust
    unsupervised allocation, and it is a fair question whether the
    supervised layer has earned its complexity over it.

    The correlation matrix is computed pairwise, and any pair without enough
    overlapping observations is given a distance of infinity so that it is
    never selected for merging.  Without that guard a single missing
    correlation silently changes the allocation.
    """
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if clean.shape[1] < 2:
        if clean.shape[1] == 1:
            return pd.Series(1.0, index=clean.columns)
        return pd.Series(dtype=float)
    corr = clean.corr(min_periods=max(6, clean.shape[0] // 4))
    tickers = list(corr.columns)
    weights = pd.Series(1.0, index=tickers)

    while len(tickers) > 1:
        block = corr.loc[tickers, tickers].to_numpy(dtype=float)
        distances = 1.0 - block
        np.fill_diagonal(distances, np.inf)
        # A pair with no usable correlation is never merged.
        distances = np.where(np.isfinite(distances), distances, np.inf)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        if not np.isfinite(distances[i, j]):
            break
        c = 0.5 * (1.0 - distances[i, j])
        weights[tickers[i]] *= c
        weights[tickers[j]] *= 1.0 - c
        tickers.pop(j)

    total = float(weights.sum())
    if total <= 0:
        weights = pd.Series(1.0, index=clean.columns)
    else:
        weights = weights / total
    return weights.reindex(returns.columns).fillna(0.0)


def inverse_volatility_weights(covariance: pd.DataFrame) -> pd.Series:
    """Weights proportional to the inverse of each asset's volatility."""
    vol = np.sqrt(np.clip(covariance.to_numpy(dtype=float).diagonal(), 1e-12, None))
    weights = 1.0 / vol
    return pd.Series(weights / weights.sum(), index=covariance.columns)


def cap_weights(weights: np.ndarray, max_weight: float = 0.25) -> np.ndarray:
    """Cap single weights, redistributing the excess proportionally."""
    weights = np.asarray(weights, dtype=float).copy()
    if len(weights) == 0:
        return weights
    for _ in range(100):
        over = weights > max_weight + 1e-12
        if not over.any():
            break
        excess = float((weights[over] - max_weight).sum())
        weights[over] = max_weight
        room = max_weight - weights[~over]
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        weights[~over] += excess * room / room_sum
    return weights


def normalise(weights: np.ndarray, target: float = 1.0) -> np.ndarray:
    total = float(np.sum(weights))
    if abs(total) < 1e-12:
        return np.zeros_like(weights)
    return weights * (target / total)
