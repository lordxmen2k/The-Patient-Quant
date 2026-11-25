#!/usr/bin/env python3
"""Check the manuscript's headline numbers against the generator.

The rule this script enforces is simple: a number that the book leans on
must be produced by the system run (``scripts/run_system.py``, which writes
``build/numbers.json``), must be rendered into ``source/generated_numbers.tex``
by the one generator that formats it, and must appear in the manuscript
through that macro and not through a number typed by hand.

Three things are checked, in order:

  1. every macro the manuscript uses is defined;
  2. every macro that can be re-derived from the canonical data agrees with
     that data, digit for digit;
  3. the rendered PDF contains the headline numbers as the text prints them,
     so that a macro which was defined but never quoted does not pass
     unnoticed.

    python3 build/check_numbers.py

The script is a companion to ``build/verify.py``: that file runs the whole
gate set against the finished PDF, this one prints the arithmetic so that a
reader can see where each number came from.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBERS = os.path.join(ROOT, "build", "numbers.json")
MACROS = os.path.join(ROOT, "source", "generated_numbers.tex")
PDF = os.path.join(ROOT, "output", "The-Patient-Quant.pdf")

# ----------------------------------------------------------------------
#  The rules.  Each entry re-derives one macro from the canonical data.
#  The tuple is (source, key, field, format).
# ----------------------------------------------------------------------
# A format tag says how a canonical value is rendered in the book.  The
# percentage tags scale by a hundred first, because the canonical data
# stores rates as fractions.
PCT = "pct2"
PCT1 = "pct1"
NUM = "num3"
NUM2 = "num2"
NUM4 = "num4"
NUM5 = "num5"
INT = "int"
PCT3 = "pct3"
PCT1 = "pct1"
PLAIN2 = "plain2"
SIGNED3 = "signed3"
MONEY = "money"
SCI3 = "sci3"
DATE = "date"


def render(value, fmt):
    if fmt == "pct2":
        return f"{value * 100:.2f}\\%"
    if fmt == "pct1":
        return f"{value * 100:.1f}\\%"
    if fmt == "num3":
        return f"{value:.3f}"
    if fmt == "num2":
        return f"{value:.2f}"
    if fmt == "num4":
        return f"{value:.4f}"
    if fmt == "num5":
        return f"{value:.5f}"
    if fmt == "int":
        return f"{int(value)}"
    if fmt == "pct3":
        return f"{value * 100:.3f}\\%"
    if fmt == "pct1":
        return f"{value * 100:.1f}\\%"
    if fmt == "plain2":
        return f"{value * 100:.2f}"
    if fmt == "signed3":
        return f"{value * 100:+.3f}\\%"
    if fmt == "money":
        return f"{value:,.0f}"
    if fmt == "sci3":
        return f"{value:.3e}"
    if fmt == "date":
        return str(value)
    raise KeyError(fmt)


def rules(data):
    summaries = data["summaries"]
    family = data["family_summaries"]
    meta = data["meta"]
    terms = data["terms"]
    selection = data["selection"]
    frictions = data["frictions"]

    out = []

    def add(macro, source, key, field, fmt):
        out.append((macro, source, key, field, fmt))

    # --- the five accounts the book reports on -------------------------
    accounts = {
        "Spy": "SPY buy and hold",
        "Full": "System, tax exempt",
        "Tax": "System, taxable",
        "Equal": "Equal weight of every fund",
        "Policy": "Strategic policy, no rebalancing",
    }
    fields = [
        ("CAGR", "cagr", PCT),
        ("Sharpe", "sharpe", NUM),
        ("SharpeNaive", "sharpe_naive", NUM),
        ("DD", "max_drawdown", PCT),
        ("Vol", "volatility", PCT),
        ("Sortino", "sortino", NUM),
        ("Calmar", "calmar", NUM),
        ("WorstYear", "worst_12m", PCT),
        ("Recovery", "time_to_recovery_months", INT),
        ("Lo", "lo_scale_factor", NUM),
    ]
    for prefix, label in accounts.items():
        for suffix, field, fmt in fields:
            add(f"num{prefix}{suffix}", "summaries", label, field, fmt)

    # --- the layer attributions ----------------------------------------
    layers = {
        "Core": "no tilt, no governor",
        "Tilt": "tilt only",
        "Gov": "governor only",
    }
    for prefix, label in layers.items():
        for suffix, field, fmt in [("CAGR", "cagr", PCT),
                                   ("Sharpe", "sharpe", NUM),
                                   ("DD", "max_drawdown", PCT),
                                   ("WorstYear", "worst_12m", PCT),
                                   ("Recovery", "time_to_recovery_months", INT)]:
            add(f"num{prefix}{suffix}", "family", label, field, fmt)

    # --- the shape of the record ---------------------------------------
    add("numRecordMonths", "meta", None, "system_months", INT)
    add("numFamilySize", "meta", None, "n_trial_family", INT)
    add("numModelConfigs", "meta", None, "n_model_configurations", INT)
    add("numNFunds", "terms", None, "n_funds", INT)

    # --- the selection tests --------------------------------------------
    add("numPBO", "selection.pbo", None, "pbo", NUM)
    add("numPBOSplits", "selection.pbo", None, "n_combinations", INT)
    add("numPBOTrials", "selection.pbo", None, "n_trials", INT)
    add("numPBOProb", "selection.pbo", None, "probability_above_zero", NUM)
    add("numDSR", "selection", None, "dsr", NUM4)
    add("numDSRImplied", "selection", None,
        "dsr_using_implied_independent_trials", NUM4)
    add("numSPAP", "selection.hansen_spa", None, "pvalue", NUM)
    add("numSPAT", "selection.hansen_spa", None, "statistic", NUM)
    add("numRCP", "selection.white_reality_check", None, "pvalue", NUM)
    add("numRCT", "selection.white_reality_check", None, "statistic", NUM4)
    add("numAvgCorr", "selection", None,
        "average_correlation_between_trials", NUM5)
    add("numImpliedTrials", "selection", None, "implied_independent_trials", NUM)
    add("numVarSR", "selection", None, "variance_of_sharpe_across_trials",
        SCI3)
    add("numSRZero", "selection", None, "sr0_closed_form", NUM4)
    add("numSRZeroMonteCarlo", "selection", None, "sr0_monte_carlo", NUM4)
    add("numSRZeroImplied", "selection", None,
        "sr0_implied_independent_trials", NUM4)
    add("numAlpha", "selection.alpha_vs_benchmark", None, "mean", PCT3)
    add("numAlphaT", "selection.alpha_vs_benchmark", None, "tstat", NUM2)
    add("numAlphaP", "selection.alpha_vs_benchmark", None, "pvalue", NUM)
    add("numAlphaSE", "selection.alpha_vs_benchmark", None, "hac_se", PCT3)
    add("numAlphaMonths", "selection.alpha_vs_benchmark", None, "n", INT)

    # --- the frictions ---------------------------------------------------
    add("numTurnover", "frictions", None, "mean_turnover_per_month", PCT)
    add("numCash", "frictions", None, "mean_cash_weight", PCT)
    add("numTaxPenalty", "frictions", None, "after_tax_cagr_penalty", PCT)
    add("numShrink", "frictions", None, "covariance_shrinkage_intensity", PCT)
    add("numTaxTotal", "frictions", None, "total_tax_paid", MONEY)
    add("numTaxFraction", "frictions", None, "tax_as_fraction_of_initial",
        PCT1)

    # --- the window sensitivity ------------------------------------------
    add("numWindowCount", "window_summary", None, "n_windows", INT)
    add("numTiltMean", "window_summary", None, "tilt_contribution_mean",
        SIGNED3)
    add("numTiltMedian", "window_summary", None, "tilt_contribution_median",
        SIGNED3)
    add("numTiltMin", "window_summary", None, "tilt_contribution_min",
        SIGNED3)
    add("numTiltMax", "window_summary", None, "tilt_contribution_max",
        SIGNED3)
    add("numTiltPosFrac", "window_summary", None,
        "tilt_contribution_positive_fraction", PCT1)

    # --- the fund terms ---------------------------------------------------
    add("numERMean", "terms", None, "expense_ratio_mean", PLAIN2)
    add("numERMin", "terms", None, "expense_ratio_min", PLAIN2)
    add("numERMax", "terms", None, "expense_ratio_max", PLAIN2)

    # --- the alternative policies, which are family members ---------------
    for prefix, label in (("numDef", "policy: 40/20/30/10 defensive"),
                          ("numAgg", "policy: 80/15/5 aggressive")):
        for suffix, field, fmt in (("CAGR", "cagr", PCT),
                                   ("Sharpe", "sharpe", NUM),
                                   ("DD", "max_drawdown", PCT),
                                   ("WorstYear", "worst_12m", PCT),
                                   ("Recovery", "time_to_recovery_months", INT)):
            add(f"{prefix}{suffix}", "family", label, field, fmt)

    # --- the window sensitivity, aggregated over the windows ---------------
    windows = data["window_sensitivity"]
    gov = [w["governor_contribution_cagr"] for w in windows
           if "governor_contribution_cagr" in w]
    if gov:
        add("numGovMean", "computed", None, "gov_mean", SIGNED3)
        add("numGovMin", "computed", None, "gov_min", SIGNED3)
        add("numGovMax", "computed", None, "gov_max", SIGNED3)
        add("numGovPosFrac", "computed", None, "gov_positive", PCT1)

    # --- the cost sensitivity ----------------------------------------------
    bp_words = {"2 bp": "TwoBp", "6 bp": "SixBp", "12 bp": "TwelveBp"}
    for row in data["cost_sensitivity"]:
        label = bp_words[row["spread_label"]]
        add(f"numCost{label}CAGR", "cost", row["spread_label"], "cagr", PCT)
        add(f"numCost{label}Sharpe", "cost", row["spread_label"], "sharpe", NUM)

    # --- the shape of the record -------------------------------------------
    add("numRecordMonths", "meta", None, "system_months", INT)
    add("numFamilySize", "meta", None, "n_trial_family", INT)
    add("numModelConfigs", "meta", None, "n_model_configurations", INT)
    add("numNFunds", "terms", None, "n_funds", INT)
    add("numInitialCapital", "meta", None, "initial_capital", MONEY)

    # --- the dates of the panel and of the record ---------------------------
    add("numPanelStart", "meta", None, "panel_start", DATE)
    add("numPanelEnd", "meta", None, "panel_end", DATE)
    add("numRecordStart", "meta", None, "system_start", DATE)
    add("numRecordEnd", "meta", None, "system_end", DATE)
    return out


def load_numbers():
    with open(NUMBERS, encoding="utf-8") as fh:
        return json.load(fh)


def load_macros():
    with open(MACROS, encoding="utf-8") as fh:
        text = fh.read()
    return dict(re.findall(r"\\newcommand\{\\(num[A-Za-z0-9]+)\}\{([^}]*)\}",
                            text))


def manuscript_text():
    """The manuscript's own prose, with the macros left in place."""
    parts = []
    chapters = os.path.join(ROOT, "source", "chapters")
    for name in sorted(os.listdir(chapters)):
        if name.endswith(".tex"):
            with open(os.path.join(chapters, name), encoding="utf-8") as fh:
                parts.append(fh.read())
    for name in ("frontmatter.tex", "backmatter.tex"):
        path = os.path.join(ROOT, "source", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                parts.append(fh.read())
    return "\n".join(parts)


def _gov(data):
    """The governor's contribution to growth, window by window."""
    return [w["governor_contribution_cagr"] for w in data["window_sensitivity"]
            if "governor_contribution_cagr" in w]


def resolve(data, source, key, field):
    """The canonical value behind one rule."""
    if source == "summaries":
        return data["summaries"][key][field]
    if source == "family":
        return data["family_summaries"][key][field]
    if source == "meta":
        return data["meta"][field]
    if source == "terms":
        return data["terms"][field]
    if source == "selection":
        return data["selection"][field]
    if source == "selection.pbo":
        return data["selection"]["pbo"][field]
    if source == "selection.alpha_vs_benchmark":
        return data["selection"]["alpha_vs_benchmark"][field]
    if source == "selection.hansen_spa":
        return data["selection"]["hansen_spa"][field]
    if source == "selection.white_reality_check":
        return data["selection"]["white_reality_check"][field]
    if source == "window_summary":
        return data["window_summary"][field]
    if source == "cost":
        for row in data["cost_sensitivity"]:
            if row["spread_label"] == key:
                return row[field]
        raise KeyError(key)
    if source == "meta":
        return data["meta"][field]
    if source == "computed":
        if field == "gov_mean":
            return sum(_gov(data)) / len(_gov(data))
        if field == "gov_min":
            return min(_gov(data))
        if field == "gov_max":
            return max(_gov(data))
        if field == "gov_positive":
            return sum(1 for g in _gov(data) if g > 0) / len(_gov(data))
        raise KeyError(field)
    if source == "frictions":
        return data["frictions"][field]
    raise KeyError(source)


def main() -> int:
    data = load_numbers()
    macros = load_macros()
    text = manuscript_text()

    failures = []
    checked = 0

    # ---- 1. every macro used is defined --------------------------------
    used = set(re.findall(r"\\(num[A-Za-z0-9]+)", text))
    for macro in sorted(used - set(macros)):
        failures.append(f"{macro}: used in the manuscript, never defined")

    # ---- 2. every derivable macro agrees with the canonical data -------
    # The benchmark over the whole panel is computed from the canonical
    # panel file, in the same way the generator computes it.
    panel_path = os.path.join(ROOT, "build", "series", "panel_returns.csv")
    if os.path.exists(panel_path):
        import pandas as pd
        panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
        spy = panel["SPY"].dropna()
        years = (spy.index[-1] - spy.index[0]).days / 365.25
        full = (1.0 + spy).prod() ** (1.0 / years) - 1.0
        expected = f"{full * 100:.2f}\\%"
        actual = macros.get("numSpyFullCAGR", "").strip()
        if actual != expected:
            failures.append(f"numSpyFullCAGR: macro {actual!r} vs "
                            f"re-derived {expected!r}")

    for macro, source, key, field, fmt in rules(data):
        if macro not in macros:
            continue
        try:
            value = resolve(data, source, key, field)
        except KeyError:
            continue
        expected = render(value, fmt)
        actual = macros[macro].strip()
        if actual != expected:
            failures.append(
                f"{macro}: macro {actual!r} vs re-derived {expected!r}")
        checked += 1

    # ---- 3. the headline numbers are in the rendered PDF ----------------
    headline = []
    for name in ("numSpyCAGR", "numFullCAGR", "numFullDD", "numSpyDD",
                 "numCoreCAGR", "numTiltCAGR", "numGovCAGR", "numPBO",
                 "numFamilySize"):
        if name in macros:
            headline.append(macros[name].strip().replace("\\%", "%"))

    if os.path.exists(PDF):
        proc = subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                              text=True)
        rendered = proc.stdout if proc.returncode == 0 else ""
        for token in headline:
            if token and token not in rendered:
                failures.append(f"{token}: not found in the rendered PDF")
    else:
        failures.append("the finished PDF is missing; cannot check the "
                        "rendered text")

    for line in failures:
        print(f"[FAIL] {line}")
    print()
    print(f"{checked - len([f for f in failures if ' vs re-derived' in f])}"
          f"/{checked} headline macros re-derived and matched; "
          f"{len(used)} macros used by the manuscript; "
          f"{len(headline)} headline numbers checked in the rendered PDF.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
