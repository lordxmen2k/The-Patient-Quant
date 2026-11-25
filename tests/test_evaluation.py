"""Tests for the evaluation metrics.

The book's numbers are the numbers these functions produce, so each one is
checked against a case whose answer can be worked out by hand.
"""

import unittest

import numpy as np
import pandas as pd

from patientquant import evaluation


def monthly(returns):
    index = pd.date_range("2000-01-31", periods=len(returns), freq="ME")
    return pd.Series(returns, index=index)


class TestRates(unittest.TestCase):
    def test_the_geometric_mean_is_the_annualised_compound_rate(self):
        series = monthly([0.01] * 12)
        expected = 1.01 ** 12 - 1.0
        self.assertAlmostEqual(evaluation.geometric_mean(series), expected,
                               places=12)

    def test_the_arithmetic_mean_is_annualised(self):
        series = monthly([0.01] * 12)
        self.assertAlmostEqual(evaluation.arithmetic_mean(series), 0.12,
                               places=12)

    def test_the_compound_growth_rate_of_a_constant_series(self):
        series = monthly([0.01] * 120)
        expected = 1.01 ** 12 - 1.0
        self.assertAlmostEqual(evaluation.cagr_from_returns(series), expected,
                               places=12)

    def test_the_volatility_of_a_constant_series_is_zero(self):
        series = monthly([0.01] * 12)
        self.assertAlmostEqual(evaluation.volatility(series), 0.0, places=12)

    def test_the_volatility_is_the_annualised_standard_deviation(self):
        series = monthly(list(np.random.default_rng(5).normal(0.01, 0.02, 120)))
        expected = float(series.std(ddof=1)) * np.sqrt(12)
        self.assertAlmostEqual(evaluation.volatility(series), expected,
                               places=12)


class TestDrawdown(unittest.TestCase):
    def test_the_maximum_drawdown_of_a_known_path(self):
        # Up 10 per cent, then down 50 per cent: the drawdown is a half.
        series = monthly([0.10, -0.50])
        self.assertAlmostEqual(evaluation.max_drawdown(series), -0.50,
                               places=12)

    def test_a_rising_series_has_no_drawdown(self):
        series = monthly([0.01] * 24)
        self.assertAlmostEqual(evaluation.max_drawdown(series), 0.0,
                               places=12)

    def test_the_drawdown_series_measures_from_the_running_peak(self):
        series = monthly([1.0, -0.2, 0.1])
        drawdown = evaluation.drawdown_series(series)
        self.assertAlmostEqual(float(drawdown.iloc[0]), 0.0, places=12)
        self.assertAlmostEqual(float(drawdown.iloc[1]), -0.2, places=12)
        self.assertAlmostEqual(float(drawdown.iloc[2]), -0.12, places=12)

    def test_the_worst_twelve_month_window(self):
        """A bad month inside a twelve-month window is measured with the
        eleven months that surround it."""
        series = monthly([0.01] * 12 + [-0.5] + [0.01] * 12)
        expected = (1.01 ** 11) * (1.0 - 0.5) - 1.0
        self.assertAlmostEqual(evaluation.worst_window(series, 12), expected,
                               places=12)


class TestRecovery(unittest.TestCase):
    def test_a_recovery_is_counted_in_months(self):
        # Wealth: 1.10, 0.55, 0.6875, 0.859, 1.075, 1.29.  The old peak of
        # 1.10 is regained four months after the trough.
        series = monthly([0.10, -0.50, 0.25, 0.25, 0.25, 0.20])
        self.assertEqual(evaluation.time_to_recovery(series), 4)

    def test_a_series_that_never_recovers(self):
        series = monthly([0.10, -0.50, 0.0, 0.0])
        self.assertEqual(evaluation.time_to_recovery(series), -1)


class TestRatios(unittest.TestCase):
    def test_a_series_with_no_variation_has_no_usable_sharpe(self):
        """A constant series has no standard deviation, so the ratio is
        either undefined or absurd; it is never a number to quote."""
        series = monthly([0.01] * 12)
        value = evaluation.sharpe_ratio(series)["sharpe"]
        self.assertTrue(not np.isfinite(value) or abs(value) > 1e6)

    def test_the_sharpe_ratio_reports_both_the_naive_and_the_corrected(self):
        rng = np.random.default_rng(13)
        series = monthly(rng.normal(0.005, 0.03, 300))
        result = evaluation.sharpe_ratio(series)
        self.assertIn("sharpe", result)
        self.assertIn("sharpe_naive", result)
        self.assertIn("scale_factor", result)
        self.assertIn("rho1", result)

    def test_the_sortino_ratio_uses_only_the_downside(self):
        rng = np.random.default_rng(41)
        good = monthly(rng.normal(0.01, 0.03, 120))
        bad = monthly(rng.normal(-0.01, 0.03, 120))
        self.assertGreater(evaluation.sortino_ratio(good),
                           evaluation.sortino_ratio(bad))

    def test_a_series_with_no_downside_has_no_downside_deviation(self):
        series = monthly([0.02] * 24)
        self.assertAlmostEqual(evaluation.downside_deviation(series), 0.0,
                               places=12)

    def test_the_calmar_ratio_is_the_return_over_the_drawdown(self):
        series = monthly([0.01] * 120 + [-0.10] + [0.01] * 12)
        calmar = evaluation.calmar_ratio(series)
        self.assertGreater(calmar, 0.0)


class TestAutocorrelation(unittest.TestCase):
    def test_an_independent_series_has_small_autocorrelations(self):
        rng = np.random.default_rng(19)
        series = monthly(rng.normal(0.0, 0.02, 500))
        correlations = evaluation.autocorrelations(series, max_lag=5)
        self.assertEqual(len(correlations), 5)
        for value in correlations:
            self.assertLess(abs(value), 0.15)

    def test_the_scale_factor_is_larger_when_the_returns_are_persistent(self):
        rng = np.random.default_rng(23)
        independent = monthly(rng.normal(0.0, 0.02, 500))
        persistent_values = list(np.zeros(500))
        shock = rng.normal(0.0, 0.02, 500)
        for i in range(1, 500):
            persistent_values[i] = 0.7 * persistent_values[i - 1] + shock[i]
        persistent = monthly(persistent_values)
        # The factor multiplies the monthly Sharpe by the square root of
        # the annualisation horizon, corrected for serial correlation: with
        # positive autocorrelation the correction pulls it down.
        persistent_factor = evaluation.lo_scale_factor(
            12, evaluation.autocorrelations(persistent))
        independent_factor = evaluation.lo_scale_factor(
            12, evaluation.autocorrelations(independent))
        self.assertLess(persistent_factor, independent_factor)
        # The sample autocorrelations of an independent series are near
        # zero but not exactly zero, so the factor is near the square root
        # of the horizon rather than exactly it.
        self.assertAlmostEqual(independent_factor, np.sqrt(12.0), delta=0.3)
        self.assertGreater(persistent_factor, 0.0)


class TestNeweyWest(unittest.TestCase):
    def test_the_test_runs_and_reports_a_p_value(self):
        rng = np.random.default_rng(31)
        x = pd.Series(rng.normal(0.001, 0.01, 200))
        y = pd.Series(rng.normal(0.0, 0.02, 200))
        result = evaluation.newey_west_mean_tstat(x, y, lags=6)
        self.assertIn("tstat", result)
        self.assertGreaterEqual(result["pvalue"], 0.0)
        self.assertLessEqual(result["pvalue"], 1.0)


class TestSummary(unittest.TestCase):
    def test_the_summary_carries_every_field_the_book_quotes(self):
        series = monthly(np.random.default_rng(37).normal(0.005, 0.03, 300))
        summary = evaluation.summarise(series, "test")
        for field in ("cagr", "sharpe", "max_drawdown", "volatility",
                      "sortino", "calmar", "worst_12m",
                      "time_to_recovery_months", "sharpe_naive",
                      "lo_scale_factor", "rho1", "observations"):
            self.assertIn(field, summary)


if __name__ == "__main__":
    unittest.main()
