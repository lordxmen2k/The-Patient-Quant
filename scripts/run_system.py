#!/usr/bin/env python3
"""Run the system and everything needed to judge it.

Produces ``build/numbers.json`` and ``build/series/``: the record, the layer
attribution, the comparators, the sensitivity table, and the selection
tests.  Every figure in the book is drawn from these files and nothing
else.

The trial family is deliberately wide.  A reality check that only covers the
model configurations is not worth running, because those configurations
produce nearly identical portfolios and the check would pass trivially.
The family here includes everything the author actually tried: the model
configurations, the ablations, the policy variants, and the sizing variants.

    python3 scripts/run_system.py            # full run
    python3 scripts/run_system.py --quick    # a smaller family, for testing
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from patientquant import evaluation as pq_eval  # noqa: E402
from patientquant import panel as pq_panel  # noqa: E402
from patientquant import portfolio as pq_portfolio  # noqa: E402
from patientquant import selection as pq_selection  # noqa: E402
from patientquant import system as pq_system  # noqa: E402

OUT_DIR = REPO_ROOT / "build"
SERIES_DIR = OUT_DIR / "series"

ALT_POLICIES = {
    "60/30/10 (the book)": {"us_equity": 0.36, "intl_equity": 0.24,
                            "bonds": 0.30, "real_assets": 0.10},
    "40/20/30/10 defensive": {"us_equity": 0.24, "intl_equity": 0.16,
                              "bonds": 0.30, "real_assets": 0.30},
    "80/15/5 aggressive": {"us_equity": 0.48, "intl_equity": 0.32,
                           "bonds": 0.15, "real_assets": 0.05},
}


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return _clean(obj.to_dict())
    return obj


def make_config(base: pq_system.SystemConfig, **overrides) -> pq_system.SystemConfig:
    fields = dict(base.__dict__)
    fields.update(overrides)
    return pq_system.SystemConfig(**fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="use a smaller family, for testing")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("building the monthly panel ...")
    panel = pq_panel.monthly_panel()
    terms = pq_panel.fund_terms()

    configs = pq_system.default_configs()
    base = pq_system.SystemConfig(configs=tuple(configs))

    # --- the full system --------------------------------------------------
    print(f"running the system ({len(configs)} model configurations) ...")
    full = pq_system.run(panel, base)
    record = full["record"]
    print(f"  record {record.index[0].date()} -> {record.index[-1].date()} "
          f"({len(record)} months)")

    # --- the trial family -------------------------------------------------
    family: dict[str, pq_system.SystemConfig] = {}

    for cfg in configs:
        family[f"model {cfg.label()}"] = make_config(base, configs=(cfg,))

    family["no tilt, no governor"] = make_config(base, use_tilt=False,
                                                 use_governor=False)
    family["tilt only"] = make_config(base, use_governor=False)
    family["governor only"] = make_config(base, use_tilt=False)

    for kelly in (0.10, 0.50):
        family[f"kelly fraction {kelly:.2f}"] = make_config(base,
                                                            kelly_fraction=kelly)
    for tilt in (0.10, 0.50):
        family[f"tilt scale {tilt:.2f}"] = make_config(base, tilt_scale=tilt)
    for vol in (0.10, 0.15):
        family[f"volatility target {vol:.2f}"] = make_config(base, target_vol=vol)
    for half in (24.0, 120.0):
        family[f"ensemble half-life {half:.0f}"] = make_config(
            base, ensemble_half_life=half)
    for label, policy in ALT_POLICIES.items():
        if label.startswith("60/30"):
            continue
        family[f"policy: {label}"] = make_config(base, core_policy=policy)
    if args.quick:
        family = dict(list(family.items())[:8])

    print(f"running {len(family)} members of the trial family ...")
    family_returns: dict[str, pd.Series] = {}
    family_summary: dict[str, dict] = {}
    family_turnover: dict[str, float] = {}
    for label, cfg in family.items():
        variant = pq_system.run(panel, cfg)
        series = variant["exempt_returns"].dropna()
        family_returns[label] = series
        family_summary[label] = pq_eval.summarise(series, label)
        family_turnover[label] = float(variant["record"]["turnover"].mean())
        print(f"  {label:38} CAGR {family_summary[label]['cagr']:7.2%}  "
              f"SR {family_summary[label]['sharpe']:5.2f}  "
              f"DD {family_summary[label]['max_drawdown']:7.2%}")
        del variant
        gc.collect()

    # --- comparators -------------------------------------------------------
    returns = panel["returns"]
    benchmark = full["benchmark_returns"].dropna()

    # An equal-weighted portfolio of every fund that exists: the naive
    # alternative that needs no forecast, no optimiser and no argument.
    live = returns.notna()
    one_over_n = pd.Series(0.0, index=returns.index)
    for date in returns.index:
        row = live.loc[date]
        members = [t for t in returns.columns if bool(row[t])]
        if members:
            one_over_n[date] = float(returns.loc[date, members].mean())
    one_over_n = one_over_n.dropna()

    # The book's own strategic policy, held without rebalancing at all.
    static_policy = pq_system.CORE_POLICY
    policy_series = pd.Series(0.0, index=returns.index)
    for sleeve, weight in static_policy.items():
        members = [t for t in pq_system.SLEEVES[sleeve] if t in returns.columns]
        if not members:
            continue
        policy_series = policy_series + returns[members].mean(axis=1) * weight
    policy_series = policy_series.dropna()

    comparator_returns = {
        "SPY buy and hold": benchmark,
        "System, tax exempt": full["exempt_returns"].dropna(),
        "System, taxable": full["taxable_returns"].dropna(),
        "Equal weight of every fund": one_over_n,
        "Strategic policy, no rebalancing": policy_series,
    }
    summaries = {}
    for name, series in comparator_returns.items():
        common = series.dropna()
        summaries[name] = pq_eval.summarise(common, name)
        summaries[name]["window"] = [str(common.index[0].date()),
                                    str(common.index[-1].date())]

    # --- selection tests ---------------------------------------------------
    print("running the selection tests ...")
    trial_matrix = pd.DataFrame(family_returns)
    trial_matrix = trial_matrix.dropna()

    pbo = pq_selection.pbo_cscv(trial_matrix, n_splits=8, metric="sharpe")

    avg_corr = pq_selection.average_correlation(trial_matrix)
    n_implied = pq_selection.implied_independent_trials(trial_matrix.shape[1],
                                                         avg_corr)

    trial_monthly_sharpes = []
    for column in trial_matrix.columns:
        series = trial_matrix[column]
        trial_monthly_sharpes.append(float(series.mean() / (series.std(ddof=1) + 1e-12)))
    var_sr = float(np.var(trial_monthly_sharpes, ddof=1))

    system_series = full["exempt_returns"].dropna()
    monthly_sr = float(system_series.mean() / system_series.std(ddof=1))

    sr0_closed = pq_selection.expected_max_sharpe(var_sr, int(trial_matrix.shape[1]))
    sr0_implied = pq_selection.expected_max_sharpe(var_sr, max(int(round(n_implied)), 2))
    sr0_mc = pq_selection.expected_max_sharpe_montecarlo(
        var_sr, int(trial_matrix.shape[1]), n_simulations=20000)

    dsr = pq_selection.deflated_sharpe(monthly_sr, sr0_closed, len(system_series),
                                        system_series.to_numpy(dtype=float))
    dsr_implied = pq_selection.deflated_sharpe(monthly_sr, sr0_implied,
                                                len(system_series),
                                                system_series.to_numpy(dtype=float))

    # The family against the benchmark: can any member beat it better than
    # chance would allow?
    benchmark_aligned = benchmark.reindex(trial_matrix.index)
    differences = trial_matrix.sub(benchmark_aligned, axis=0).dropna()
    spa = pq_selection.hansen_spa(differences, n_bootstrap=2000)
    rc = pq_selection.white_reality_check(differences, n_bootstrap=2000)

    alpha = pq_eval.newey_west_mean_tstat(system_series, benchmark)

    # --- sensitivity -------------------------------------------------------
    print("running the sensitivity table ...")
    sensitivity = []
    for label in family:
        stats = family_summary[label]
        sensitivity.append({
            "variant": label,
            "cagr": stats["cagr"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "mean_turnover": family_turnover[label],
        })

    # --- window sensitivity ------------------------------------------------
    # The layer attribution is a statement about a window, and a window is
    # a choice.  This block re-reads the *same* four return paths from a
    # set of later starting dates and recomputes the attribution, so the
    # book can say how much of the model's contribution survives the
    # choice of the starting month and how much is one episode.
    print("running the window sensitivity ...")
    layer_paths = {
        "core": family_returns.get("no tilt, no governor"),
        "core_plus_tilt": family_returns.get("tilt only"),
        "core_plus_governor": family_returns.get("governor only"),
        "full": full["exempt_returns"].dropna(),
    }
    layer_paths = {k: v for k, v in layer_paths.items() if v is not None}
    window_sensitivity = []
    if layer_paths:
        common_index = None
        for value in layer_paths.values():
            common_index = value.index if common_index is None else (
                common_index.intersection(value.index))
        starts = [d for d in common_index[::12]]
        # A window shorter than five years cannot produce a meaningful
        # compound rate, and a one-month window produces exactly zero for
        # every variant, which would look like a finding.
        minimum_months = 60
        for start in starts:
            row = {"start": str(start.date()),
                   "months": int((common_index[-1] - start).days // 30.44 + 1)}
            for name, path in layer_paths.items():
                sliced = path.loc[start:]
                if len(sliced) < minimum_months:
                    continue
                stats = pq_eval.summarise(sliced, name)
                row[f"{name}_cagr"] = stats["cagr"]
                row[f"{name}_sharpe"] = stats["sharpe"]
                row[f"{name}_max_drawdown"] = stats["max_drawdown"]
            if "core_cagr" not in row:
                continue
            if "core_cagr" in row and "core_plus_tilt_cagr" in row:
                row["tilt_contribution_cagr"] = (
                    row["core_plus_tilt_cagr"] - row["core_cagr"])
            if "core_cagr" in row and "core_plus_governor_cagr" in row:
                row["governor_contribution_cagr"] = (
                    row["core_plus_governor_cagr"] - row["core_cagr"])
            window_sensitivity.append(row)
    contribs = [r["tilt_contribution_cagr"] for r in window_sensitivity
                if "tilt_contribution_cagr" in r]
    window_summary = {
        "n_windows": len(window_sensitivity),
        "tilt_contribution_mean": float(np.mean(contribs)) if contribs else None,
        "tilt_contribution_median": float(np.median(contribs)) if contribs else None,
        "tilt_contribution_min": float(np.min(contribs)) if contribs else None,
        "tilt_contribution_max": float(np.max(contribs)) if contribs else None,
        "tilt_contribution_positive_fraction": (
            float(np.mean([c > 0 for c in contribs])) if contribs else None),
    }

    # --- cost sensitivity --------------------------------------------------
    cost_sensitivity = []
    for label, spread in (("2 bp", 0.0002), ("6 bp", 0.0006), ("12 bp", 0.0012)):
        cfg = make_config(base, spread_liquid=spread, spread_moderate=spread * 3)
        variant = pq_system.run(panel, cfg)
        stats = pq_eval.summarise(variant["exempt_returns"].dropna(), label)
        cost_sensitivity.append({
            "spread_label": label, "cagr": stats["cagr"],
            "sharpe": stats["sharpe"], "max_drawdown": stats["max_drawdown"],
        })
        del variant
        gc.collect()

    # --- shrinkage diagnostics ---------------------------------------------
    shrunk, intensity = pq_portfolio.ledoit_wolf_covariance(returns.tail(60))

    # --- friction summary ---------------------------------------------------
    weights = full["weights"]
    initial = base.initial_capital
    tax_total = float(full["tax_paid_total"])
    exempt_cagr = summaries["System, tax exempt"]["cagr"]
    taxable_cagr = summaries["System, tax taxable"]["cagr"] if "System, tax taxable" in summaries else summaries["System, taxable"]["cagr"]

    numbers = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_model_configurations": len(configs),
            "n_trial_family": len(family),
            "panel_start": str(panel["returns"].index[0].date()),
            "panel_end": str(panel["returns"].index[-1].date()),
            "system_start": str(record.index[0].date()),
            "system_end": str(record.index[-1].date()),
            "system_months": int(len(record)),
            "initial_capital": initial,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "summaries": summaries,
        "family_summaries": family_summary,
        "selection": {
            "pbo": pbo,
            "average_correlation_between_trials": avg_corr,
            "implied_independent_trials": n_implied,
            "variance_of_sharpe_across_trials": var_sr,
            "sr0_closed_form": sr0_closed,
            "sr0_implied_independent_trials": sr0_implied,
            "sr0_monte_carlo": sr0_mc,
            "monthly_sharpe": monthly_sr,
            "dsr": dsr,
            "dsr_using_implied_independent_trials": dsr_implied,
            "hansen_spa": spa,
            "white_reality_check": rc,
            "alpha_vs_benchmark": alpha,
        },
        "frictions": {
            "mean_turnover_per_month": float(record["turnover"].mean()),
            "mean_cash_weight": float(weights["BIL"].mean()),
            "total_tax_paid": tax_total,
            "tax_as_fraction_of_initial": tax_total / initial,
            "after_tax_cagr_penalty": exempt_cagr - taxable_cagr,
            "covariance_shrinkage_intensity": intensity,
        },
        "sensitivity": sensitivity,
        "cost_sensitivity": cost_sensitivity,
        "window_sensitivity": window_sensitivity,
        "window_summary": window_summary,
        "terms": {
            "expense_ratio_mean": float(terms["expense_ratio"].mean()),
            "expense_ratio_min": float(terms["expense_ratio"].min()),
            "expense_ratio_max": float(terms["expense_ratio"].max()),
            "n_funds": int(len(terms)),
        },
    }

    (OUT_DIR / "numbers.json").write_text(
        json.dumps(_clean(numbers), indent=2, sort_keys=True) + "\n")

    record.to_csv(SERIES_DIR / "record.csv")
    weights.to_csv(SERIES_DIR / "weights.csv")
    trial_matrix.to_csv(SERIES_DIR / "trials.csv")
    panel["returns"].to_csv(SERIES_DIR / "panel_returns.csv")
    full["signal"]["combined"].to_csv(SERIES_DIR / "combined_forecast.csv",
                                       index=False)
    full["signal"]["model_weights"].to_csv(SERIES_DIR / "model_weights.csv")
    terms.to_csv(SERIES_DIR / "fund_terms.csv")
    pd.Series(one_over_n).to_csv(SERIES_DIR / "one_over_n.csv",
                                  header=["return"])
    pd.Series(policy_series).to_csv(SERIES_DIR / "policy_series.csv",
                                     header=["return"])

    print(f"\nwrote {OUT_DIR / 'numbers.json'} and {SERIES_DIR}")
    print(f"elapsed {round(time.time() - started, 1)}s")
    for name in ("System, tax exempt", "System, taxable", "SPY buy and hold"):
        s = summaries[name]
        print(f"  {name:26} CAGR {s['cagr']:7.2%}  Sharpe {s['sharpe']:5.2f}  "
              f"maxDD {s['max_drawdown']:7.2%}")
    print(f"  PBO {pbo['pbo']:.3f}  DSR {dsr:.4f}  SPA p {spa['pvalue']:.3f}  "
          f"RC p {rc['pvalue']:.3f}  alpha t {alpha['tstat']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
