#!/usr/bin/env python3
"""Write the book's headline numbers into LaTeX macros.

Every number the book states about its own results is defined here, once,
from ``build/numbers.json`` and the canonical series files.  The chapters
reference macros; they never contain a transcribed figure.  If the system
is re-run and a number changes, the book changes with it, and the
verification gate in ``build/verify.py`` fails if a macro and the
underlying data ever disagree.

    python3 build/make_numbers_tex.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

NUMBERS = json.loads((REPO / "build" / "numbers.json").read_text())
SERIES = REPO / "build" / "series"
OUT = REPO / "source" / "generated_numbers.tex"


def pct(value: float, places: int = 2) -> str:
    """A percentage written for a LaTeX paragraph."""
    return f"{value * 100:.{places}f}\\%"


def signed_pct(value: float, places: int = 2) -> str:
    return f"{value * 100:+.{places}f}\\%"


def num(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def main() -> int:
    meta = NUMBERS["meta"]
    summaries = NUMBERS["summaries"]
    family = NUMBERS["family_summaries"]
    selection = NUMBERS["selection"]
    frictions = NUMBERS["frictions"]
    terms = NUMBERS["terms"]
    window = NUMBERS["window_summary"]

    # The benchmark over the whole panel, which is longer than the
    # system's own record and is quoted in the chapter that defines the
    # target.  It is computed here from the canonical panel file.
    panel = pd.read_csv(SERIES / "panel_returns.csv", index_col=0,
                        parse_dates=True)
    spy = panel["SPY"].dropna()
    spy_years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_full_cagr = (1.0 + spy).prod() ** (1.0 / spy_years) - 1.0

    lines: list[str] = []
    lines.append("% ---------------------------------------------------------------------")
    lines.append("%  The Patient Quant — generated headline numbers")
    lines.append("%  Written by build/make_numbers_tex.py from build/numbers.json.")
    lines.append("%  Do not edit: every figure in the text comes from here.")
    lines.append("% ---------------------------------------------------------------------")
    lines.append("")

    def define(name: str, value: str) -> None:
        # A LaTeX control word may not contain a digit: the tokeniser reads
        # "\numSpyW12" as "\numSpyW" followed by the characters "12", so a
        # macro with a digit in its name can never be called, and the
        # definition itself puts the digits into the output stream as
        # stray text.  The generator refuses to write such a name.
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} contains a non-letter; "
                             "LaTeX control words may not contain digits")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # --- the record -------------------------------------------------------
    define("numPanelStart", meta["panel_start"])
    define("numPanelEnd", meta["panel_end"])
    define("numRecordStart", meta["system_start"])
    define("numRecordEnd", meta["system_end"])
    define("numRecordMonths", str(meta["system_months"]))
    define("numNFunds", str(terms["n_funds"]))
    define("numModelConfigs", str(meta["n_model_configurations"]))
    define("numFamilySize", str(meta["n_trial_family"]))
    define("numInitialCapital", f"{meta['initial_capital']:,.0f}")

    # --- the comparators --------------------------------------------------
    for key, prefix in (
        ("SPY buy and hold", "numSpy"),
        ("System, tax exempt", "numFull"),
        ("System, taxable", "numTax"),
        ("Equal weight of every fund", "numEqual"),
        ("Strategic policy, no rebalancing", "numPolicy"),
    ):
        s = summaries[key]
        define(f"{prefix}CAGR", pct(s["cagr"]))
        define(f"{prefix}Sharpe", num(s["sharpe"]))
        define(f"{prefix}SharpeNaive", num(s["sharpe_naive"]))
        define(f"{prefix}DD", pct(s["max_drawdown"]))
        define(f"{prefix}Vol", pct(s["volatility"]))
        define(f"{prefix}Sortino", num(s["sortino"]))
        define(f"{prefix}Calmar", num(s["calmar"]))
        define(f"{prefix}WorstYear", pct(s["worst_12m"]))
        define(f"{prefix}Recovery", str(s["time_to_recovery_months"]))
        define(f"{prefix}Lo", num(s["lo_scale_factor"]))
        define(f"{prefix}RhoOne", num(s["rho1"]))
    define("numSpyFullCAGR", pct(spy_full_cagr))

    # --- the layer attribution -------------------------------------------
    for key, prefix in (
        ("no tilt, no governor", "numCore"),
        ("tilt only", "numTilt"),
        ("governor only", "numGov"),
    ):
        s = family[key]
        define(f"{prefix}CAGR", pct(s["cagr"]))
        define(f"{prefix}Sharpe", num(s["sharpe"]))
        define(f"{prefix}DD", pct(s["max_drawdown"]))
        define(f"{prefix}WorstYear", pct(s["worst_12m"]))
        define(f"{prefix}Recovery", str(s["time_to_recovery_months"]))

    # --- the alternative policies -----------------------------------------
    for key, prefix in (
        ("policy: 40/20/30/10 defensive", "numDef"),
        ("policy: 80/15/5 aggressive", "numAgg"),
    ):
        s = family[key]
        define(f"{prefix}CAGR", pct(s["cagr"]))
        define(f"{prefix}Sharpe", num(s["sharpe"]))
        define(f"{prefix}DD", pct(s["max_drawdown"]))
        define(f"{prefix}WorstYear", pct(s["worst_12m"]))
        define(f"{prefix}Recovery", str(s["time_to_recovery_months"]))

    # --- the selection tests ----------------------------------------------
    pbo = selection["pbo"]
    define("numPBO", num(pbo["pbo"], 3))
    define("numPBOSplits", str(pbo["n_combinations"]))
    define("numPBOTrials", str(pbo["n_trials"]))
    define("numPBOProb", num(pbo["probability_above_zero"], 3))
    define("numDSR", num(selection["dsr"], 4))
    define("numDSRImplied", num(selection["dsr_using_implied_independent_trials"], 4))
    define("numSPAP", num(selection["hansen_spa"]["pvalue"], 3))
    define("numSPAT", num(selection["hansen_spa"]["statistic"], 3))
    define("numRCP", num(selection["white_reality_check"]["pvalue"], 3))
    define("numRCT", num(selection["white_reality_check"]["statistic"], 4))
    define("numAvgCorr", num(selection["average_correlation_between_trials"], 5))
    define("numImpliedTrials", num(selection["implied_independent_trials"], 3))
    define("numVarSR", f"{selection['variance_of_sharpe_across_trials']:.3e}")
    define("numSRZero", num(selection["sr0_closed_form"], 4))
    define("numSRZeroMonteCarlo", num(selection["sr0_monte_carlo"], 4))
    define("numSRZeroImplied",
           num(selection["sr0_implied_independent_trials"], 4))
    define("numAlpha", pct(selection["alpha_vs_benchmark"]["mean"], 3))
    define("numAlphaT", num(selection["alpha_vs_benchmark"]["tstat"], 2))
    define("numAlphaP", num(selection["alpha_vs_benchmark"]["pvalue"], 3))
    define("numAlphaSE", pct(selection["alpha_vs_benchmark"]["hac_se"], 3))
    define("numAlphaMonths", str(selection["alpha_vs_benchmark"]["n"]))

    # --- the frictions -----------------------------------------------------
    define("numTurnover", pct(frictions["mean_turnover_per_month"], 2))
    define("numCash", pct(frictions["mean_cash_weight"], 2))
    define("numTaxTotal", f"{frictions['total_tax_paid']:,.0f}")
    define("numTaxFraction", pct(frictions["tax_as_fraction_of_initial"], 1))
    define("numTaxPenalty", pct(frictions["after_tax_cagr_penalty"], 2))
    define("numShrink", pct(frictions["covariance_shrinkage_intensity"], 2))

    # --- the window sensitivity --------------------------------------------
    define("numWindowCount", str(window["n_windows"]))
    define("numTiltMean", signed_pct(window["tilt_contribution_mean"], 3))
    define("numTiltMedian", signed_pct(window["tilt_contribution_median"], 3))
    define("numTiltMin", signed_pct(window["tilt_contribution_min"], 3))
    define("numTiltMax", signed_pct(window["tilt_contribution_max"], 3))
    define("numTiltPosFrac", pct(window["tilt_contribution_positive_fraction"], 1))

    windows = NUMBERS["window_sensitivity"]
    gov = [w["governor_contribution_cagr"] for w in windows
           if "governor_contribution_cagr" in w]
    define("numGovMean", signed_pct(float(np.mean(gov)), 3))
    define("numGovMin", signed_pct(float(np.min(gov)), 3))
    define("numGovMax", signed_pct(float(np.max(gov)), 3))
    define("numGovPosFrac", pct(float(np.mean([g > 0 for g in gov])), 1))

    # --- the fund terms -----------------------------------------------------
    define("numERMean", f"{terms['expense_ratio_mean'] * 100:.2f}")
    define("numERMin", f"{terms['expense_ratio_min'] * 100:.2f}")
    define("numERMax", f"{terms['expense_ratio_max'] * 100:.2f}")

    # --- the cost sensitivity -----------------------------------------------
    # The basis-point labels are spelled out, because a control word may
    # not contain a digit.
    bp_words = {"2 bp": "TwoBp", "6 bp": "SixBp", "12 bp": "TwelveBp"}
    for row in NUMBERS["cost_sensitivity"]:
        label = bp_words.get(row["spread_label"],
                             row["spread_label"].replace(" ", ""))
        define(f"numCost{label}CAGR", pct(row["cagr"]))
        define(f"numCost{label}Sharpe", num(row["sharpe"]))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(lines) - 7} definitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
