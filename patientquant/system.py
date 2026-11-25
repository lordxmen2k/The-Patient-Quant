"""The system, end to end.

The system has six moving parts and one rule about how they are allowed to
interact.

    signal      Bayesian panel regression with a posterior over the edge
    tilt        certainty-equivalent Kelly, applied inside each sleeve
    core        hierarchical risk parity across the risky sleeves
    governor    a damped volatility target and a drawdown brake
    frictions   spreads, a no-trade band, and the tax account
    record      measured against the benchmark with the selection tests applied

The rule is that the *strategic* allocation is never set by the model.  The
core comes from risk parity across asset-class sleeves, which uses no
forecast at all; the model is allowed only to tilt inside a sleeve, and only
by a bounded amount.  This is deliberate.  A model that sets the strategic
allocation has to be right about expected returns of asset classes, which is
the hardest thing in the book to be right about and the easiest thing to
overfit.  A model that tilts inside a sleeve can be wrong without changing
what the investor is fundamentally exposed to, and the book can therefore
measure exactly how much the model contributed.

The portfolio is formed at each month end from information available at that
month end, and earns the return realised in the following month.

Two accounts are simulated with identical mechanics and an identical weight
path: a taxable account that pays capital gains and dividend tax, and a
tax-exempt account that does not.  The difference between them is therefore
the tax and nothing else, which is the only way to attribute it honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import costs as pq_costs
from . import evaluation as pq_eval
from . import features as pq_features
from . import models as pq_models
from . import portfolio as pq_portfolio
from . import risk as pq_risk

# --- Sleeves ---------------------------------------------------------------
# A sleeve is a set of funds that face broadly the same macro risk.  The
# strategic allocation is made across sleeves; the model is only ever
# allowed to tilt inside one.
SLEEVES: dict[str, list[str]] = {
    "us_equity": [
        "SPY", "VTI", "IVV", "QQQ", "IWM", "RSP", "MDY", "VUG", "VTV",
        "MGK", "MGY",
    ],
    "intl_equity": ["EFA", "EEM", "VEA", "VWO", "ACWI", "SPDW"],
    "bonds": [
        "SHY", "IEF", "TLT", "BIL", "SHV", "BND", "LQD", "HYG", "JNK",
        "TIP", "MUB", "VCIT", "VGLT", "BNDX",
    ],
    "real_assets": ["GLD", "IAU", "VNQ", "VNQI", "DBC", "PDBC", "RINF"],
    "alts": ["DBMF"],
}

CASH_TICKER = "BIL"
RISK_SLEEVES = list(SLEEVES)

# The strategic policy: how the investor is exposed before the model or the
# risk layer does anything.  It is a policy, stated in advance and not
# derived from the data, and every number in it is argued in the book.
#
#   36%  United States equity
#   24%  equity outside the United States
#   30%  bonds
#   10%  real assets: gold, real estate and commodities
#
# The 60/30/10 split between growth assets, bonds and real assets is a
# house view, and the book reports what the record looks like under other
# splits rather than presenting this one as discovered.
#
# The 60:40 split between US and non-US equity is close to the world's own
# market capitalisation split, so it does not smuggle in a view.
CORE_POLICY: dict[str, float] = {
    "us_equity": 0.36,
    "intl_equity": 0.24,
    "bonds": 0.30,
    "real_assets": 0.10,
}

TICKER_TO_SLEEVE: dict[str, str] = {
    ticker: sleeve for sleeve, tickers in SLEEVES.items() for ticker in tickers
}


@dataclass
class SystemConfig:
    """Every knob the system has, in one place."""

    # Signal
    horizon: int = 3
    min_months: int = 36
    min_rows: int = 180
    ensemble_half_life: float = 60.0   # months of memory in the model average
    initial_capital: float = 1_000_000.0
    configs: tuple[pq_models.Config, ...] = field(default_factory=tuple)

    # Portfolio
    core_policy: dict = field(default_factory=lambda: dict(CORE_POLICY))
    gamma: float = 1.0             # relative risk aversion
    kelly_fraction: float = 0.25   # fraction of full Kelly
    tilt_scale: float = 0.25       # how much of a sleeve a tilt may move
    use_tilt: bool = True          # ablation switch: the model's contribution
    use_governor: bool = True      # ablation switch: the risk layer's

    # Risk
    target_vol: float = 0.12
    damping: float = 0.5
    drawdown_first: float = 0.15
    drawdown_second: float = 0.25

    # Frictions
    no_trade_band: float = 0.005
    spread_liquid: float = 0.0002
    spread_moderate: float = 0.0006

    # Accounting
    tax_schedule: pq_costs.TaxSchedule = field(default_factory=pq_costs.TaxSchedule)

    def label(self) -> str:
        return (f"h{self.horizon} f{self.kelly_fraction} g{self.gamma} "
                f"d{self.damping} tilt{self.tilt_scale}")


def default_configs() -> list[pq_models.Config]:
    """The model set.

    The number of members matters: the Deflated Sharpe Ratio has to be told
    how many configurations were tried, so this list is deliberately fixed
    and its length is reported in the book rather than hidden.
    """
    feature_sets = {
        "full": tuple(pq_features.FEATURE_NAMES),
        "trend": ("mom_12_1", "mom_6_1", "mom_36", "rev_1", "trend_12",
                  "drawdown_12"),
        "risk": ("vol_12", "vol_ratio", "beta_36", "corr_36", "illiq_12",
                 "drawdown_12"),
    }
    configs: list[pq_models.Config] = []
    for name, feats in feature_sets.items():
        for horizon in (1, 3):
            for strength in (0.5, 2.0):
                configs.append(pq_models.Config(
                    name=name, features=feats, horizon=horizon,
                    prior_strength=strength,
                ))
    return configs


# ---------------------------------------------------------------------------
# Strategic allocation
# ---------------------------------------------------------------------------
def sleeve_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weighted return of each sleeve, used for the strategic allocation."""
    returns = panel["returns"]
    out = {}
    for sleeve, tickers in SLEEVES.items():
        present = [t for t in tickers if t in returns.columns]
        if present:
            out[sleeve] = returns[present].mean(axis=1)
    return pd.DataFrame(out)


def core_weights(panel: dict[str, pd.DataFrame], date: pd.Timestamp,
                 policy: dict[str, float] | None = None) -> pd.Series:
    """The strategic allocation: the stated policy, scaled to what exists.

    A sleeve with no listed fund at ``date`` cannot be held, so its weight
    is redistributed over the sleeves that can.  Early in the sample that
    makes the allocation narrower than it later becomes, which is a fact
    about the data rather than a choice.
    """
    policy = policy or CORE_POLICY
    prices = panel["raw_prices"]
    live_row = prices.loc[date] if date in prices.index else pd.Series(dtype=float)

    available = {}
    for sleeve, weight in policy.items():
        members = [t for t in SLEEVES.get(sleeve, [])
                   if t in live_row.index and np.isfinite(live_row[t])]
        if members:
            available[sleeve] = weight

    weights = pd.Series(0.0, index=RISK_SLEEVES)
    if not available:
        return weights
    total = float(sum(available.values()))
    for sleeve, weight in available.items():
        weights[sleeve] = weight / total
    return weights


def within_sleeve_weights(panel: dict[str, pd.DataFrame], sleeve: str,
                           date: pd.Timestamp, window: int = 60) -> pd.Series:
    """Base weights inside one sleeve: inverse volatility of its members.

    Only funds that exist on ``date`` are eligible.  A fund that has not
    listed yet has no price and therefore no weight; it cannot be bought.
    """
    returns = panel["returns"]
    prices = panel["raw_prices"]
    live = prices.loc[date] if date in prices.index else pd.Series(dtype=float)
    tickers = [t for t in SLEEVES[sleeve]
               if t in returns.columns and t in live.index
               and np.isfinite(live[t])]
    if not tickers:
        return pd.Series(dtype=float)

    history = returns.loc[:date, tickers].tail(window)
    if len(history) < 12:
        history = returns.loc[:date, tickers].dropna(how="all")
    if history.empty:
        return pd.Series(1.0 / len(tickers), index=tickers)
    vol = history.std(ddof=1) * np.sqrt(12)
    fallback = float(history.std(ddof=1).median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = float(vol.median()) if np.isfinite(vol.median()) and vol.median() > 0 else 0.01
    vol = vol.replace(0.0, np.nan).fillna(fallback)
    vol = vol.where(np.isfinite(vol) & (vol > 0), fallback)
    weights = 1.0 / vol
    return weights / float(weights.sum())


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
_SIGNAL_CACHE: dict[tuple, dict] = {}


def _signal_key(config: SystemConfig) -> tuple:
    """A cache key for the signal.

    The signal depends on the model set and on the two warm-up parameters,
    and on nothing else.  The strategic allocation, the tilt scale, the
    Kelly fraction, the volatility target and the drawdown brake all act
    *downstream* of it, so two configurations that differ only in those
    share a signal.  Caching on this key is what makes the trial family
    affordable: without it the book would refit every model for every
    member of the family, and the family is the point.
    """
    members = tuple(
        (c.label(), c.horizon, c.features, c.prior_strength, c.prior_mean,
         c.alpha0, c.beta0)
        for c in config.configs
    )
    return (members, config.min_months, config.min_rows,
            config.ensemble_half_life)


def build_signal(panel: dict[str, pd.DataFrame], config: SystemConfig,
                 use_cache: bool = True) -> dict:
    """Run every model configuration and combine them."""
    key = _signal_key(config)
    if use_cache and key in _SIGNAL_CACHE:
        return _SIGNAL_CACHE[key]

    raw_features = pq_features.build_features(panel)
    scaled = pq_features.cross_sectional_scale(raw_features)

    predictions: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.DataFrame] = {}
    horizons: dict[str, int] = {}
    for cfg in config.configs:
        target = pq_features.build_target(panel, cfg.horizon)
        targets[cfg.label()] = target
        horizons[cfg.label()] = cfg.horizon
        frame = pq_models.walk_forward(scaled, target, cfg,
                                       min_months=config.min_months,
                                       min_rows=config.min_rows)
        if len(frame):
            predictions[cfg.label()] = frame

    dates = sorted({pd.Timestamp(d) for frame in predictions.values()
                    for d in frame["date"].unique()})
    date_index = pd.DatetimeIndex(dates)

    weights, combined = pq_models.ensemble_combine(
        predictions, targets, horizons, date_index,
        targets[config.configs[0].label()].index,
        half_life=config.ensemble_half_life,
    )
    signal = {
        "raw_features": raw_features,
        "scaled_features": scaled,
        "predictions": predictions,
        "model_weights": weights,
        "combined": combined,
        "dates": date_index,
        "targets": targets,
    }
    if use_cache:
        _SIGNAL_CACHE[key] = signal
    return signal


def kelly_tilts(combined: pd.DataFrame, date: pd.Timestamp, tickers: list[str],
                covariance: pd.DataFrame, config: SystemConfig) -> pd.Series:
    """Certainty-equivalent Kelly tilts for the assets live on ``date``."""
    out = pd.Series(0.0, index=tickers)
    rows = combined[combined["date"] == date]
    if rows.empty:
        return out

    rows = rows.set_index("ticker").reindex(tickers)
    live = rows["mu"].notna()
    present = [t for t in tickers if bool(live.get(t, False))]
    if not present:
        return out

    mu = rows.loc[present, "mu"].to_numpy(dtype=float)
    var_mu = rows.loc[present, "var_mu"].to_numpy(dtype=float)
    sub = covariance.reindex(index=present, columns=present).fillna(0.0)

    raw = pq_portfolio.kelly_weights(mu, sub, var_mu,
                                     gamma=config.gamma,
                                     fraction=config.kelly_fraction)
    out.loc[present] = raw
    return out


# ---------------------------------------------------------------------------
# Target weights
# ---------------------------------------------------------------------------
def build_target_weights(panel, signal, config, date) -> tuple[pd.Series, dict]:
    """The strategic core, the model tilt, and the risk governor."""
    returns = panel["returns"]
    prices = panel["raw_prices"]
    live_row = prices.loc[date] if date in prices.index else pd.Series(dtype=float)
    tickers = [t for t in returns.columns
               if (t in TICKER_TO_SLEEVE or t == CASH_TICKER)
               and t in live_row.index and np.isfinite(live_row[t])]

    core = core_weights(panel, date, config.core_policy)
    base = pd.Series(0.0, index=tickers)
    for sleeve, sleeve_weight in core.items():
        inside = within_sleeve_weights(panel, sleeve, date)
        if inside.empty:
            continue
        for ticker in inside.index:
            if ticker in base.index:
                base[ticker] = sleeve_weight * float(inside[ticker])

    # --- the model's only job: tilt inside a sleeve ----------------------
    history = returns.loc[:date].tail(36)
    covariance, shrinkage = pq_portfolio.ledoit_wolf_covariance(history)
    if config.use_tilt:
        tilts = kelly_tilts(signal["combined"], date, tickers, covariance, config)
    else:
        tilts = pd.Series(0.0, index=tickers)

    tilted = base.copy()
    for sleeve in SLEEVES:
        members = [t for t in SLEEVES[sleeve] if t in tilted.index]
        if not members:
            continue
        for ticker in members:
            deviation = float(np.clip(tilts.get(ticker, 0.0), -0.5, 0.5))
            tilted[ticker] = tilted[ticker] * (1.0 + config.tilt_scale * deviation)
        sleeve_total = float(tilted[members].sum())
        base_total = float(base[members].sum())
        if sleeve_total > 0 and base_total > 0:
            tilted[members] = tilted[members] / sleeve_total * base_total
    tilted = tilted.where(np.isfinite(tilted), 0.0)

    # --- the risk governor ----------------------------------------------
    risky = tilted.copy()
    risky[CASH_TICKER] = 0.0
    risky = risky.where(np.isfinite(risky), 0.0).clip(lower=0.0)

    lookback = returns.loc[:date, risky.index].tail(36)
    if len(lookback) > 12 and lookback.notna().any().any():
        cov, _ = pq_portfolio.pairwise_covariance(lookback)
    else:
        cov = pd.DataFrame(0.0, index=risky.index, columns=risky.index)

    portfolio_series = (risky * returns.reindex(columns=risky.index).fillna(0.0)).sum(axis=1)
    history_port = portfolio_series.loc[:date].dropna()
    drawdown = pq_risk.trailing_drawdown((1.0 + history_port).cumprod())

    scale, brake = 1.0, 1.0
    if config.use_governor and not cov.empty and np.isfinite(cov.to_numpy()).all():
        scaled, scale, brake = pq_risk.apply_governor(
            risky, cov, config.target_vol, config.damping, drawdown,
            first_level=config.drawdown_first, second_level=config.drawdown_second)
        risky = scaled.clip(lower=0.0)

    total = float(risky.sum())
    if not np.isfinite(total) or total <= 0:
        risky = pd.Series(0.0, index=risky.index)
        total = 0.0
    if total > 1.0:
        risky = risky / total
        total = 1.0

    target = risky.copy()
    target[CASH_TICKER] = 1.0 - total
    target = target.where(np.isfinite(target), 0.0).clip(lower=0.0)
    if float(target.sum()) > 0:
        target = target / float(target.sum())

    diagnostics = {
        "vol_scale": scale,
        "brake": brake,
        "drawdown": drawdown,
        "shrinkage": shrinkage,
        "core": core.to_dict(),
        "gross_exposure": total,
    }
    return target, diagnostics


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------
def _px(prices: pd.Series, ticker: str) -> float:
    """The price of a fund at a date, or zero if it has none."""
    try:
        value = float(prices.get(ticker, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def trade(account: pq_costs.TaxableAccount, target: pd.Series,
          prices: pd.Series, date: pd.Timestamp, band: float,
          spreads: dict[str, float]) -> tuple[float, float]:
    """Move the account towards ``target``, respecting the no-trade band."""
    held = {t: account.shares(t) for t in account.lots if account.shares(t) > 1e-12}
    value = account.total_value({t: _px(prices, t) for t in held})
    if value <= 0:
        return 0.0, 0.0

    current = pd.Series(0.0, index=target.index)
    for ticker, shares in held.items():
        if ticker in current.index:
            current[ticker] = shares * _px(prices, ticker) / value

    desired = target.reindex(current.index).fillna(0.0)
    change = desired - current
    change = change.where(change.abs() > band, 0.0)

    turnover = float(change.abs().sum())

    # Sell first to raise cash, then buy.
    for ticker, delta in change[change < 0].sort_values().items():
        price = _px(prices, ticker)
        if price <= 0:
            continue
        shares = min(account.shares(ticker), abs(delta) * value / price)
        if shares > 1e-9:
            account.sell(ticker, shares, price, date)

    for ticker, delta in change[change > 0].sort_values(ascending=False).items():
        price = _px(prices, ticker)
        if price <= 0:
            continue
        budget = min(delta * value, max(account.cash, 0.0))
        if budget <= 1.0:
            continue
        account.buy(ticker, budget / price, price, date)

    cost = 0.0
    for ticker, delta in change.items():
        if delta == 0:
            continue
        cost += abs(delta) * spreads.get(ticker, 0.0005) / 2.0
    account.cash -= cost * value
    account.prune_windows(date)
    return turnover, cost


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def run(panel: dict[str, pd.DataFrame], config: SystemConfig,
        benchmark: str = "SPY") -> dict:
    """Run the system and return every record the book needs."""
    returns = panel["returns"]
    raw_prices = panel["raw_prices"]
    dividends = panel["dividends"]
    signal = build_signal(panel, config)

    combined = signal["combined"]
    dates = signal["dates"]
    spread_map = pq_costs.default_spreads(
        returns.columns, config.spread_liquid, config.spread_moderate)

    taxable = pq_costs.TaxableAccount(schedule=config.tax_schedule)
    # The tax-exempt account pays no tax, so it has no wash-sale rule to
    # apply: the machinery would take cash out of the account for a tax
    # that is never payable.
    exempt = pq_costs.TaxableAccount(schedule=config.tax_schedule,
                                      wash_sale_rules=False)
    taxable.cash = config.initial_capital
    exempt.cash = config.initial_capital

    all_dates = returns.index
    records: list[dict] = []
    weights_rows: list[pd.Series] = []
    settlements: list[dict] = []

    current_target = pd.Series(0.0, index=returns.columns)
    current_target[CASH_TICKER] = 1.0

    for i, date in enumerate(all_dates):
        if len(dates) == 0 or date < dates[0]:
            continue
        if i + 1 >= len(all_dates):
            break
        next_date = all_dates[i + 1]

        prices_now = raw_prices.loc[date]
        prices_next = raw_prices.loc[next_date]
        div_now = dividends.loc[date]

        # Dividends are received on the positions held at the month end.
        taxable.receive_dividends({t: float(div_now.get(t, 0.0)) for t in taxable.lots})
        exempt.receive_dividends({t: float(div_now.get(t, 0.0)) for t in exempt.lots})

        value_before_taxable = taxable.total_value(
            {t: _px(prices_now, t) for t in taxable.lots})
        value_before_exempt = exempt.total_value(
            {t: _px(prices_now, t) for t in exempt.lots})

        target, diagnostics = build_target_weights(panel, signal, config, date)
        weights_rows.append(target)

        turnover, cost = trade(taxable, target, prices_now, date,
                                config.no_trade_band, spread_map)
        trade(exempt, target, prices_now, date, config.no_trade_band, spread_map)

        value_after_taxable = taxable.total_value(
            {t: _px(prices_next, t) for t in taxable.lots})
        value_after_exempt = exempt.total_value(
            {t: _px(prices_next, t) for t in exempt.lots})

        # The tax for the year is settled when the year turns over.
        tax_paid = 0.0
        if next_date.year != date.year:
            settlement = taxable.settle_year(date.year)
            exempt.settle_year(date.year, pay_tax=False)
            tax_paid = settlement["total_tax"]
            settlements.append(settlement)
            value_after_taxable -= tax_paid

        records.append({
            "date": next_date,
            "taxable_value": value_after_taxable,
            "exempt_value": value_after_exempt,
            "turnover": turnover,
            "cost": cost,
            "tax_paid": tax_paid,
            **{k: v for k, v in diagnostics.items() if not isinstance(v, dict)},
        })

    frame = pd.DataFrame(records).set_index("date")
    frame["taxable_return"] = frame["taxable_value"].pct_change()
    frame["exempt_return"] = frame["exempt_value"].pct_change()

    weights = pd.DataFrame(weights_rows, index=frame.index)
    weights.index = frame.index

    benchmark_returns = returns[benchmark].reindex(frame.index).fillna(0.0)

    return {
        "config": config,
        "signal": signal,
        "record": frame,
        "weights": weights,
        "taxable_returns": frame["taxable_return"].dropna(),
        "exempt_returns": frame["exempt_return"].dropna(),
        "benchmark_returns": benchmark_returns,
        "settlements": settlements,
        "tax_paid_total": float(frame["tax_paid"].sum()),
        "cost_total": float((frame["cost"] * frame["taxable_value"].shift(1)).sum()),
    }
