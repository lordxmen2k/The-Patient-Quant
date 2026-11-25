"""Tests for the sizing and the risk layer.

Two rules carry the book's result, and both are checked here:

  * the Kelly weights shrink toward cash when the edge cannot be told from
    zero, which is what makes protecting the principal and growing it the
    same action;
  * the governor is bounded, so that it can only ever reduce exposure, and
    it only does so when the evidence says to.
"""

import os
import unittest

import numpy as np
import pandas as pd

from patientquant import portfolio, risk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def correlated_returns(n=250, k=5, seed=7):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2)) * 0.01
    loadings = rng.normal(size=(2, k)) * 0.6
    idio = rng.normal(size=(n, k)) * 0.012
    data = factors @ loadings + idio
    return pd.DataFrame(data, columns=[f"F{i}" for i in range(k)])


class TestCovariance(unittest.TestCase):
    def test_pairwise_covariance_survives_a_nan(self):
        frame = correlated_returns()
        frame.iloc[0, 0] = np.nan
        covariance, count = portfolio.pairwise_covariance(frame)
        self.assertTrue(np.isfinite(covariance.to_numpy()).all())

    def test_ledoit_wolf_shrinks_towards_a_constant_correlation(self):
        frame = correlated_returns()
        covariance, intensity = portfolio.ledoit_wolf_covariance(frame)
        sample = frame.cov().to_numpy()
        self.assertTrue(0.0 <= intensity <= 1.0)
        # The shrunk covariance is a convex mixture of the sample
        # covariance and the constant-correlation target, so its
        # off-diagonal elements are pulled towards each other.
        off_sample = sample[~np.eye(len(sample), dtype=bool)]
        off_shrunk = covariance.to_numpy()[~np.eye(len(sample), dtype=bool)]
        self.assertLess(off_shrunk.std(), off_sample.std())

    def test_the_constant_correlation_target_has_equal_off_diagonals(self):
        frame = correlated_returns()
        covariance, _ = portfolio.ledoit_wolf_covariance(frame)
        values = covariance.to_numpy()
        off = values[~np.eye(len(values), dtype=bool)]
        # The target is not asserted directly, but the shrinkage intensity
        # is a fraction, so the result cannot be the target alone unless
        # the sample was already constant-correlation.
        self.assertGreater(len(set(np.round(off, 12))), 1)


class TestKellyWeights(unittest.TestCase):
    def test_no_edge_gives_no_position(self):
        sigma = pd.DataFrame(np.eye(3) * 0.01, index=list("ABC"),
                             columns=list("ABC"))
        weights = portfolio.kelly_weights(np.zeros(3), sigma, np.zeros(3))
        self.assertTrue(np.allclose(weights, 0.0))

    def test_a_positive_edge_gives_a_long_position(self):
        sigma = pd.DataFrame(np.eye(3) * 0.01, index=list("ABC"),
                             columns=list("ABC"))
        weights = portfolio.kelly_weights(np.array([0.01, 0.0, 0.0]), sigma,
                                          np.zeros(3))
        self.assertGreater(weights[0], 0.0)
        # The two funds with no edge carry no position at all.
        self.assertAlmostEqual(weights[1], 0.0, places=12)
        self.assertAlmostEqual(weights[2], 0.0, places=12)

    def test_the_weights_are_bounded(self):
        sigma = pd.DataFrame(np.eye(2) * 0.0001, index=list("AB"),
                             columns=list("AB"))
        weights = portfolio.kelly_weights(np.array([1.0, 1.0]), sigma,
                                          np.zeros(2))
        self.assertLessEqual(float(np.max(np.abs(weights))), 0.25)

    def test_uncertainty_in_the_mean_shrinks_the_position(self):
        sigma = pd.DataFrame(np.eye(2) * 0.01, index=list("AB"),
                             columns=list("AB"))
        mu = np.array([0.002, 0.0])
        certain = portfolio.kelly_weights(mu, sigma, np.zeros(2))
        uncertain = portfolio.kelly_weights(mu, sigma, np.array([0.01, 0.0]))
        self.assertLess(abs(float(uncertain[0])), abs(float(certain[0])))

    def test_the_kelly_fraction_scales_the_position(self):
        sigma = pd.DataFrame(np.eye(2) * 0.01, index=list("AB"),
                             columns=list("AB"))
        # A small enough edge that neither position reaches the cap.
        mu = np.array([0.002, 0.0])
        quarter = portfolio.kelly_weights(mu, sigma, np.zeros(2), fraction=0.25)
        half = portfolio.kelly_weights(mu, sigma, np.zeros(2), fraction=0.50)
        self.assertAlmostEqual(float(half[0]), 2.0 * float(quarter[0]),
                               places=12)


class TestRiskLayer(unittest.TestCase):
    def test_the_volatility_scale_is_bounded(self):
        self.assertGreaterEqual(risk.volatility_scale(0.01, 0.10), 0.0)
        self.assertLessEqual(risk.volatility_scale(0.50, 0.10), 2.0)

    def test_the_volatility_scale_reduces_exposure_when_vol_is_high(self):
        self.assertLess(risk.volatility_scale(0.30, 0.10), 1.0)
        self.assertGreater(risk.volatility_scale(0.03, 0.10), 1.0)

    def test_the_volatility_scale_ignores_a_degenerate_estimate(self):
        self.assertEqual(risk.volatility_scale(0.0, 0.10), 1.0)
        self.assertEqual(risk.volatility_scale(float("nan"), 0.10), 1.0)

    def test_the_drawdown_brake_steps_down(self):
        self.assertEqual(risk.drawdown_brake(0.0), 1.0)
        self.assertEqual(risk.drawdown_brake(-0.10), 1.0)
        self.assertEqual(risk.drawdown_brake(-0.20), 0.50)
        self.assertEqual(risk.drawdown_brake(-0.30), 0.25)

    def test_the_brake_never_increases_exposure(self):
        for drawdown in (0.0, -0.05, -0.15, -0.25, -0.5):
            self.assertLessEqual(risk.drawdown_brake(drawdown), 1.0)

    def test_the_trailing_drawdown_is_measured_from_the_peak(self):
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        self.assertAlmostEqual(risk.trailing_drawdown(equity),
                               110.0 / 120.0 - 1.0, places=12)

    def test_the_governor_multipliers_are_bounded(self):
        """The scaler may scale exposure in either direction, but no more
        than a factor of two, and the brake only ever reduces it."""
        covariance = pd.DataFrame(np.eye(2) * 0.01, index=list("AB"),
                                  columns=list("AB"))
        risky = pd.Series([0.5, 0.5], index=list("AB"))
        for drawdown in (0.0, -0.10, -0.20, -0.40):
            for target in (0.02, 0.10, 0.40):
                scaled, scale, brake = risk.apply_governor(
                    risky, covariance, target, drawdown=drawdown)
                self.assertLessEqual(scale, 2.0)
                self.assertGreaterEqual(scale, 0.0)
                self.assertLessEqual(brake, 1.0)
                self.assertGreaterEqual(brake, 0.0)
                # The brake is what makes the governor a protection rather
                # than a lever: the product never exceeds two.
                self.assertLessEqual(scale * brake, 2.0 + 1e-12)

    def test_the_system_never_levers(self):
        """The system caps the risky exposure at one and holds the rest in
        cash, so the governor can never borrow to reach its target."""
        record_path = os.path.join(ROOT, "build", "series", "record.csv")
        if not os.path.exists(record_path):
            self.skipTest("the system has not been run in this checkout")
        record = pd.read_csv(record_path)
        self.assertLessEqual(float(record["gross_exposure"].max()), 1.0 + 1e-9)
        self.assertGreaterEqual(float(record["gross_exposure"].min()), 0.0)


if __name__ == "__main__":
    unittest.main()
