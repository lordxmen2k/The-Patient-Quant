"""The prediction layer: a Bayesian linear model with a posterior over the edge.

The point of this module is not to forecast returns well.  It is to produce,
for each asset and each date, an estimate of a forward return *together with a
calibrated estimate of how wrong that estimate might be*.  The portfolio
layer then does the sizing.

The model is a panel regression with a normal-inverse-gamma conjugate prior::

    y_i = x_i' beta + e_i,      e_i ~ N(0, sigma^2)
    beta | sigma^2 ~ N(m0, sigma^2 Lambda^-1)
    sigma^2 ~ Inv-Gamma(alpha0, beta0)

Because the prior is conjugate, the posterior is available in closed form, so
the book can print it.  From the posterior we take two numbers per asset:

``mu``
    the posterior predictive mean of the forward return;

``var_mu``
    the posterior *variance of the mean*, not of the observation.  This is
    the error in the estimate of the edge, and it is the number that
    ultimately shrinks the position.  Reporting only ``mu`` would be the
    single most common error in applied machine learning for investing.

The model is estimated by walk-forward: at every formation date the training
set contains only rows whose target has already been realised.

Several configurations are run in parallel and combined by online Bayesian
model averaging.  The number of configurations tried is recorded, because it
is the number that the deflated Sharpe ratio has to be told about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES
from .stats import norm_cdf


@dataclass
class Config:
    """One member of the model set."""

    name: str
    features: tuple[str, ...] = tuple(FEATURE_NAMES)
    horizon: int = 3
    prior_strength: float = 1.0   # prior precision on each coefficient
    prior_mean: float = 0.0
    alpha0: float = 2.5           # inverse-gamma shape
    beta0: float = 0.5            # inverse-gamma scale

    def label(self) -> str:
        return f"{self.name}|h{self.horizon}|lam{self.prior_strength}"


@dataclass
class Prediction:
    """One formation date's worth of model output."""

    date: pd.Timestamp
    mu: pd.Series
    var_mu: pd.Series
    var_resid: float
    n_train: int


@dataclass
class PosteriorRegression:
    """Posterior quantities of the normal-inverse-gamma model."""

    beta_hat: np.ndarray
    precision: np.ndarray     # (X'X + Lambda)^-1
    sigma2_hat: float
    alpha_n: float
    beta_n: float

    def predictive(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, var of mean, var of observation) for rows of ``X``."""
        mean = X @ self.beta_hat
        quad = np.einsum("ij,jk,ik->i", X, self.precision, X)
        var_mean = self.sigma2_hat * quad
        var_obs = self.sigma2_hat * (1.0 + quad)
        return mean, var_mean, var_obs


def fit_posterior(X: np.ndarray, y: np.ndarray, config: Config) -> PosteriorRegression:
    """Fit the Bayesian panel regression by conjugate updating."""
    n, k = X.shape
    prior_precision = np.eye(k) * config.prior_strength
    prior_mean = np.full(k, config.prior_mean)

    a = X.T @ X + prior_precision
    rhs = X.T @ y + prior_precision @ prior_mean
    precision = np.linalg.inv(a)
    beta_hat = precision @ rhs

    residual = y - X @ beta_hat
    quad = prior_mean @ prior_precision @ prior_mean + y @ y - beta_hat @ (X.T @ X + prior_precision) @ beta_hat
    alpha_n = config.alpha0 + 0.5 * n
    beta_n = config.beta0 + 0.5 * max(quad, 1e-12)
    sigma2_hat = beta_n / (alpha_n - 1.0)
    return PosteriorRegression(beta_hat, precision, sigma2_hat, alpha_n, beta_n)


def walk_forward(features: dict[str, pd.DataFrame], target: pd.DataFrame, config: Config,
                 min_months: int = 48, min_rows: int = 240) -> pd.DataFrame:
    """Walk-forward predictions for one configuration.

    The training set at formation date ``t`` contains only rows whose target
    has already been realised, that is, rows dated ``s`` with
    ``s + horizon <= t``.

    Returns a long frame with the predictive mean and the posterior variance
    of the mean for every asset at every formation date.
    """
    cols = list(config.features)
    calendar = target.index

    # Long-format design matrix: one row per (formation date, asset), with
    # the features as columns.  Rows are sorted by date so that the training
    # set at any date is a prefix and can be taken with a slice.
    stacked = pd.concat(
        [features[name].stack(future_stack=True).rename(name) for name in cols],
        axis=1,
    )
    stacked = stacked.dropna()
    stacked = stacked.reindex(sorted(stacked.index, key=lambda ix: (ix[0], ix[1])))
    empty = pd.DataFrame(columns=["date", "ticker", "mu", "var_mu", "sigma2", "n_train"])
    if len(stacked) < min_rows:
        return empty

    y_long = target.stack(future_stack=True).reindex(stacked.index)
    stacked = stacked[y_long.notna()]
    y_long = y_long.reindex(stacked.index)
    if len(stacked) < min_rows:
        return empty

    row_dates = stacked.index.get_level_values(0)
    positions = calendar.get_indexer(row_dates)

    X_all = np.clip(stacked.to_numpy(dtype=float), -5.0, 5.0)
    y_all = np.clip(y_long.to_numpy(dtype=float), -0.95, 1.0)
    tickers_all = stacked.index.get_level_values(1)

    records: list[pd.DataFrame] = []
    for i, date in enumerate(calendar):
        if i < min_months:
            continue
        # Training rows are those at calendar position at most i - horizon.
        cut_position = i - config.horizon
        if cut_position < 0:
            continue
        cut = int(np.searchsorted(positions, cut_position, side="right"))
        if cut < min_rows:
            continue

        post = fit_posterior(X_all[:cut], y_all[:cut], config)

        # The cross-section that exists on this date.
        here = np.where(positions == i)[0]
        if len(here) == 0:
            continue
        mu, var_mu, _ = post.predictive(X_all[here])

        records.append(pd.DataFrame({
            "date": date,
            "ticker": tickers_all[here],
            "mu": mu,
            "var_mu": var_mu,
            "sigma2": post.sigma2_hat,
            "n_train": cut,
        }))

    if not records:
        return empty
    return pd.concat(records, ignore_index=True)


@dataclass
class Ensemble:
    """Online Bayesian model average over configurations."""

    members: list[Config]
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)  # date x config
    combined: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_trials(self) -> int:
        return len(self.members)


def ensemble_combine(predictions: dict[str, pd.DataFrame],
                     targets: dict[str, pd.DataFrame],
                     horizons: dict[str, int],
                     dates: pd.DatetimeIndex,
                     calendar: pd.DatetimeIndex,
                     half_life: float = 60.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the running model average and the combined forecast.

    ``predictions`` holds each member's forecast; ``targets`` holds each
    member's forward return; ``horizons`` says how many months ahead each
    member is looking.

    The critical rule is that a member's score may only use outcomes that
    have already happened.  A member with a three-month horizon produces a
    forecast at date ``s`` whose outcome is realised at ``s + 3``; that
    outcome may be used to score the member from date ``s + 3`` onward and
    not before.  Getting this wrong would let the ensemble weight itself on
    the future.

    The score itself is a proper scoring rule: for each asset it is minus
    one half of the squared standardised forecast error, averaged over the
    cross-section.  A member that is well calibrated scores about -1/2 on
    average whether or not it has any skill, and a member that is
    miscalibrated scores worse, so the rule rewards calibration rather than
    confidence.

    The score is accumulated with an exponential discount whose half-life is
    ``half_life`` months.  Two things are wrong with a plain running sum.
    First, a sum over hundreds of months accumulates noise as if it were
    evidence, and the ensemble would eventually put all of its weight on
    whichever member happened to be luckiest early on.  Second, and more
    importantly, the evidence itself is not stationary: Chapter 3 documents
    that published factor returns decay, and the same is true of a model
    built on price data.  A discounted score is the standard answer to a
    non-stationary benchmark, and its half-life is a stated parameter
    rather than a hidden one.
    """
    names = list(predictions)
    log_score = {name: pd.Series(0.0, index=dates) for name in names}
    weight_rows: list[dict] = []
    combined_rows: list[pd.DataFrame] = []

    decay = 0.5 ** (1.0 / float(half_life)) if half_life and half_life > 0 else 1.0
    cumulative = {name: 0.0 for name in names}

    for date in dates:
        position = calendar.get_indexer([date])[0]
        if position < 0:
            continue

        # Update each member's cumulative log score with the outcome that
        # became known at this date.
        for name in names:
            frame = predictions[name]
            rows = frame[frame["date"] == date]
            if len(rows) == 0:
                continue
            # The formation date whose outcome is realised exactly now.
            source = position - horizons.get(name, 1)
            if source < 0:
                continue
            target = targets[name]
            if source >= len(target.index):
                continue
            realised_here = target.iloc[source]
            joined = rows.set_index("ticker")[["mu", "var_mu"]].join(
                realised_here.rename("y"), how="inner"
            )
            joined = joined.dropna()
            if len(joined) == 0:
                continue
            mu = joined["mu"].to_numpy(dtype=float)
            y = joined["y"].to_numpy(dtype=float)
            var = np.maximum(joined["var_mu"].to_numpy(dtype=float), 1e-10)
            z = (y - mu) / np.sqrt(var)
            score = float(np.mean(-0.5 * z * z))
            # Discounted accumulation: the ensemble remembers a member's
            # record over roughly ``half_life`` months, and forgets it
            # gradually rather than all at once.
            cumulative[name] = decay * cumulative[name] + score
            log_score[name].loc[date] = cumulative[name]

        # The weight of a member is the exponential of its discounted
        # cumulative score, computed only from outcomes that were known
        # before this date.  The maximum is subtracted before exponentiating
        # so that the arithmetic cannot overflow.
        values = np.array([cumulative[name] for name in names], dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        if float(values.max()) - float(values.min()) < 1e-12:
            weights = {name: 1.0 / len(names) for name in names}
        else:
            raw = np.exp(values - values.max()) + 1e-300
            total = float(raw.sum())
            weights = {name: float(raw[i] / total)
                       for i, name in enumerate(names)}
        weight_rows.append({"date": date, **{f"w_{k}": v for k, v in weights.items()}})

        # Combined forecast for this date.
        #
        # Units.  A member with horizon ``h`` forecasts the *h-month* total
        # return, so its posterior mean and variance are in h-month units.
        # The portfolio problem the forecast feeds into is a monthly one:
        # the covariance is estimated from monthly returns and the book
        # rebalances monthly.  Mixing a 3-month mean with a monthly
        # covariance would understate the 3-month members' edge by a
        # factor of three and make the horizons incomparable, so every
        # member is converted to a monthly rate *before* it is combined.
        # The conversion is exact for the mean and first-order for the
        # variance, and it happens before the model-spread term is added,
        # so disagreement is also measured in monthly units.
        per_ticker: dict[str, dict] = {}
        for name in names:
            frame = predictions[name]
            rows = frame[frame["date"] == date]
            h = float(horizons.get(name, 1))
            if not np.isfinite(h) or h < 1.0:
                h = 1.0
            for _, row in rows.iterrows():
                entry = per_ticker.setdefault(row["ticker"],
                                              {"mu": [], "var_mu": [], "w": []})
                mu_h = float(row["mu"])
                var_h = float(row["var_mu"])
                if not np.isfinite(mu_h) or not np.isfinite(var_h):
                    continue
                growth = 1.0 + min(max(mu_h, -0.99), 10.0)
                entry["mu"].append(growth ** (1.0 / h) - 1.0)
                entry["var_mu"].append(max(var_h, 0.0) / (h * h))
                entry["w"].append(weights[name])
        if per_ticker:
            records = []
            for ticker, entry in per_ticker.items():
                w = np.asarray(entry["w"], dtype=float)
                mu = np.asarray(entry["mu"], dtype=float)
                var = np.asarray(entry["var_mu"], dtype=float)
                w = w / w.sum()
                mean = float(w @ mu)
                # Law of total variance: the posterior variance of the mean is
                # the average of the members' own variances *plus* the
                # spread of the members' means.  Model disagreement is
                # uncertainty, and it belongs in the denominator.
                within = float(w @ var)
                between = float(w @ (mu - mean) ** 2)
                records.append({"date": date, "ticker": ticker, "mu": mean,
                                "var_mu": within + between})
            combined_rows.append(pd.DataFrame(records))

    weights_frame = pd.DataFrame(weight_rows).set_index("date") if weight_rows else pd.DataFrame()
    combined_frame = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    return weights_frame, combined_frame
