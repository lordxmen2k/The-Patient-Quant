"""Selection discipline: how many things were tried, and does that matter?

A backtest that reports the best of many configurations is not a measurement;
it is the maximum of several noisy draws, and the maximum of noisy draws is
biased upward.  This module implements the tests that price that bias.

**The Deflated Sharpe Ratio** (Bailey and Lopez de Prado, 2014).  Given a
Sharpe ratio, the number of independent trials that produced it, the sample
length and the skewness and kurtosis of the returns, the DSR is the
probability that the true Sharpe ratio exceeds the highest Sharpe ratio
that could have been obtained by chance from the same number of trials.

**The probability of backtest overfitting** (Bailey, Borwein, Lopez de
Prado and Zhu, 2017).  Combinatorially symmetric cross-validation splits the
sample in every possible way, picks the best configuration in each
in-sample half, and measures how often that winner falls below the median
out of sample.

**The reality check** (White, 2000) and the **superior predictive ability
test** (Hansen, 2005).  Both are bootstrap tests of whether the best of a
family of strategies beats a benchmark better than chance would allow.
Hansen studentises the statistic and imposes a null that respects the
nuisance parameters, and it is the more powerful of the two.

The formulas are transcribed from the original papers, not from summaries.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .stats import norm_cdf, norm_ppf

EULER_MASCHERONI = 0.5772156649015329
EULERS_NUMBER = math.e


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def expected_max_sharpe(variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe ratio under the null of no skill.

    Bailey and Lopez de Prado's approximation (their equation 1)::

        E[max SR] ~ E[SR] + sqrt(V[SR]) [(1 - gamma) Z^-1(1 - 1/N)
                                           + gamma Z^-1(1 - 1/(N e))]

    with ``gamma`` the Euler-Mascheroni constant and ``Z`` the standard
    normal CDF.  The mean term vanishes under the null, so the function
    returns the standard-deviation-scaled part.
    """
    if n_trials < 2:
        return 0.0
    term_one = norm_ppf(1.0 - 1.0 / n_trials)
    term_two = norm_ppf(1.0 - 1.0 / (n_trials * EULERS_NUMBER))
    return float(math.sqrt(max(variance, 0.0)) * (
        (1.0 - EULER_MASCHERONI) * term_one + EULER_MASCHERONI * term_two))


def expected_max_sharpe_montecarlo(variance: float, n_trials: int,
                                    n_simulations: int = 200_000,
                                    seed: int = 20260101) -> float:
    """Simulation check on :func:`expected_max_sharpe`.

    The paper verifies its closed form by simulation; so does the book.
    """
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((n_simulations, n_trials))
    return float(draws.max(axis=1).mean() * math.sqrt(max(variance, 0.0)))


def skewness(returns: np.ndarray) -> float:
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    centred = x - x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(np.mean(centred ** 3) / s ** 3)


def kurtosis(returns: np.ndarray) -> float:
    """Non-excess kurtosis, so that a normal distribution has kurtosis 3."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 3.0
    centred = x - x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 3.0
    return float(np.mean(centred ** 4) / s ** 4)


def deflated_sharpe(sharpe: float, sr0: float, observations: int,
                    returns: np.ndarray) -> float:
    """The Deflated Sharpe Ratio: the probability that SR beats SR0.

    ``DSR = Z[ (SR - SR0) sqrt(T - 1)
              / sqrt(1 - gamma3 SR + ((gamma4 - 1)/4) SR^2) ]``

    where ``SR`` is the estimated one-period Sharpe ratio, ``SR0`` the
    expected maximum under the null, ``T`` the sample length, and gamma3 and
    gamma4 the skewness and (non-excess) kurtosis of the returns.
    """
    if not np.isfinite(sharpe) or observations < 3:
        return float("nan")
    denom = 1.0 - skewness(returns) * sharpe + ((kurtosis(returns) - 1.0) / 4.0) * sharpe ** 2
    if denom <= 0:
        return float("nan")
    z = (sharpe - sr0) * math.sqrt(observations - 1) / math.sqrt(denom)
    return float(norm_cdf(z))


def average_correlation(matrix: pd.DataFrame) -> float:
    """The weighted average pairwise correlation of a set of trials."""
    values = matrix.to_numpy(dtype=float)
    n = values.shape[1]
    if n < 2:
        return 0.0
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr)
    total = corr.sum() - n  # remove the unit diagonal
    return float(total / (n * (n - 1)))


def implied_independent_trials(n_trials: int, avg_correlation: float) -> float:
    """The number of independent trials implied by a set of dependent ones.

    Bailey and Lopez de Prado's appendix A.3: ``N* = rho + (1 - rho) M``,
    which reduces to ``M`` when the trials are uncorrelated and to ``1`` when
    they are perfectly correlated.
    """
    rho = float(np.clip(avg_correlation, 0.0, 1.0))
    return float(rho + (1.0 - rho) * n_trials)


# ---------------------------------------------------------------------------
# Combinatorially symmetric cross-validation
# ---------------------------------------------------------------------------
def pbo_cscv(performance_matrix: pd.DataFrame, n_splits: int = 8,
             metric: str = "sharpe") -> dict:
    """Probability of backtest overfitting by CSCV.

    ``performance_matrix`` has one row per observation and one column per
    trial (a configuration).  The sample is split into ``n_splits`` equal
    blocks; every way of choosing half of the blocks as the in-sample set
    is used.  For each split the best in-sample configuration is chosen and
    its out-of-sample rank recorded; the logit of that relative rank is
    collected, and the PBO is the probability mass of the logit distribution
    below zero.
    """
    values = performance_matrix.to_numpy(dtype=float)
    n_obs, n_trials = values.shape
    if n_splits % 2 != 0 or n_splits < 4 or n_trials < 2:
        raise ValueError("need an even number of splits of at least 4 and two trials")
    block = n_obs // n_splits
    if block < 2:
        raise ValueError("not enough observations for the requested number of splits")

    rows = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx, oos_idx = [], []
        for s in range(n_splits):
            target = is_idx if s in combo else oos_idx
            target.extend(range(s * block, (s + 1) * block))
        is_block = values[np.ix_(is_idx, np.arange(n_trials))]
        oos_block = values[np.ix_(oos_idx, np.arange(n_trials))]

        if metric == "sharpe":
            is_score = is_block.mean(axis=0) / (is_block.std(axis=0, ddof=1) + 1e-12)
            oos_score = oos_block.mean(axis=0) / (oos_block.std(axis=0, ddof=1) + 1e-12)
        else:
            is_score = is_block.mean(axis=0)
            oos_score = oos_block.mean(axis=0)

        best = int(np.argmax(is_score))
        rank = int(np.sum(oos_score > oos_score[best]) + 1)
        omega = rank / (n_trials + 1)
        lam = math.log(omega / (1.0 - omega))
        rows.append(lam)

    logits = np.asarray(rows, dtype=float)
    pbo = float(np.mean(logits < 0.0))
    return {
        "pbo": pbo,
        "median_logit": float(np.median(logits)),
        "mean_logit": float(np.mean(logits)),
        "n_combinations": len(logits),
        "n_splits": n_splits,
        "n_trials": n_trials,
        "probability_above_zero": float(np.mean(logits >= 0.0)),
    }


# ---------------------------------------------------------------------------
# Bootstrap tests of superior predictive ability
# ---------------------------------------------------------------------------
def stationary_bootstrap_indices(n: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis and Romano's stationary bootstrap indices."""
    indices = np.empty(n, dtype=int)
    i = 0
    while i < n:
        length = int(rng.geometric(1.0 / mean_block))
        start = int(rng.integers(0, n))
        for j in range(length):
            if i >= n:
                break
            indices[i] = (start + j) % n
            i += 1
    return indices


def hansen_spa(differences: pd.DataFrame, n_bootstrap: int = 2000,
               seed: int = 20260101, block_length: float | None = None) -> dict:
    """Hansen's (2005) test for superior predictive ability.

    ``differences`` holds, for each strategy and each period, the strategy's
    return minus the benchmark's return.  The test statistic is the
    studentised maximum; the null distribution is imposed by recentring the
    bootstrap resamples about a data-dependent estimate of the mean.

    Returns the observed statistic and the bootstrap p-value.
    """
    values = differences.to_numpy(dtype=float)
    n, m = values.shape
    if n < 10 or m < 1:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": n, "m": m}

    rng = np.random.default_rng(seed)
    means = values.mean(axis=0)
    sds = values.std(axis=0, ddof=1)
    sds = np.where(sds > 1e-12, sds, 1e-12)

    observed = float(np.max(np.sqrt(n) * means / sds))
    if observed <= 0:
        return {"statistic": observed, "pvalue": 1.0, "n": n, "m": m}

    if block_length is None:
        block_length = max(2.0, n ** (1.0 / 3.0))

    mu_c = means.copy()
    threshold = -math.sqrt(2.0 * math.log(math.log(max(n, 3))))
    bad = (math.sqrt(n) * means / sds) <= threshold
    mu_c = np.where(bad, 0.0, means)

    exceed = 0
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        resample = values[idx] - mu_c
        boot_means = resample.mean(axis=0)
        boot_sds = np.where(resample.std(axis=0, ddof=1) > 1e-12,
                            resample.std(axis=0, ddof=1), 1e-12)
        stat = np.max(np.sqrt(n) * boot_means / boot_sds)
        if stat >= observed:
            exceed += 1

    return {
        "statistic": observed,
        "pvalue": float((exceed + 1) / (n_bootstrap + 1)),
        "n": n,
        "m": m,
        "n_bootstrap": n_bootstrap,
    }


def white_reality_check(differences: pd.DataFrame, n_bootstrap: int = 2000,
                        seed: int = 20260101) -> dict:
    """White's (2000) reality check for data snooping.

    Unlike the SPA test, the statistic is not studentised and the null
    distribution is imposed by recentring every strategy at zero.
    """
    values = differences.to_numpy(dtype=float)
    n, m = values.shape
    if n < 10 or m < 1:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": n, "m": m}

    rng = np.random.default_rng(seed)
    means = values.mean(axis=0)
    observed = float(np.sqrt(n) * np.max(means))
    if observed <= 0:
        return {"statistic": observed, "pvalue": 1.0, "n": n, "m": m}

    block_length = max(2.0, n ** (1.0 / 3.0))
    exceed = 0
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        boot_means = values[idx].mean(axis=0)
        stat = float(np.sqrt(n) * np.max(boot_means))
        if stat >= observed:
            exceed += 1

    return {
        "statistic": observed,
        "pvalue": float((exceed + 1) / (n_bootstrap + 1)),
        "n": n,
        "m": m,
        "n_bootstrap": n_bootstrap,
    }
