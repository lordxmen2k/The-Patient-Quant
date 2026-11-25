"""Tests for the statistical functions the book prints.

Every estimator in this package is written from its objective function
rather than taken from a library, so the tests here check the written
versions against values that can be computed by hand or that are standard
enough to be quoted: the normal and Student-t quantiles, the incomplete
beta function behind the Student-t CDF, and the symmetry relations that
must hold whatever the implementation.
"""

import math
import unittest

from patientquant import stats


class TestNormal(unittest.TestCase):
    def test_norm_cdf_at_the_mean(self):
        self.assertAlmostEqual(stats.norm_cdf(0.0), 0.5, places=12)

    def test_norm_cdf_known_values(self):
        # One standard deviation: 0.8413447460685429.
        self.assertAlmostEqual(stats.norm_cdf(1.0), 0.8413447460685429,
                               places=12)
        # Two standard deviations: 0.9772498680518208.
        self.assertAlmostEqual(stats.norm_cdf(2.0), 0.9772498680518208,
                               places=12)
        self.assertAlmostEqual(stats.norm_cdf(-1.959963984540054), 0.025,
                               places=12)

    def test_norm_cdf_is_symmetric(self):
        for x in (0.1, 0.7, 1.3, 2.2, 3.0):
            self.assertAlmostEqual(stats.norm_cdf(x) + stats.norm_cdf(-x), 1.0,
                                   places=12)

    def test_norm_sf_is_the_complement(self):
        for x in (0.0, 0.5, 1.25, 4.0):
            self.assertAlmostEqual(stats.norm_sf(x), 1.0 - stats.norm_cdf(x),
                                   places=12)

    def test_norm_pdf_integrates_to_one(self):
        # Simpson's rule over a wide interval; the tolerance is loose
        # because the rule is, but the identity is exact.
        n, lo, hi = 200000, -12.0, 12.0
        h = (hi - lo) / n
        total = 0.0
        for i in range(n + 1):
            x = lo + i * h
            w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
            total += w * stats.norm_pdf(x)
        self.assertAlmostEqual(total * h / 3.0, 1.0, places=8)

    def test_norm_ppf_inverts_norm_cdf(self):
        for p in (0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999):
            x = stats.norm_ppf(p)
            self.assertAlmostEqual(stats.norm_cdf(x), p, places=10)

    def test_norm_ppf_known_quantiles(self):
        self.assertAlmostEqual(stats.norm_ppf(0.975), 1.959963984540054,
                               places=9)
        self.assertAlmostEqual(stats.norm_ppf(0.95), 1.6448536269514722,
                               places=9)

    def test_norm_ppf_is_symmetric(self):
        for p in (0.01, 0.05, 0.2, 0.35):
            self.assertAlmostEqual(stats.norm_ppf(p),
                                   -stats.norm_ppf(1.0 - p), places=10)


class TestErf(unittest.TestCase):
    def test_erf_at_zero(self):
        self.assertAlmostEqual(stats.erf(0.0), 0.0, places=14)

    def test_erf_known_values(self):
        self.assertAlmostEqual(stats.erf(1.0), 0.8427007929497149, places=12)
        self.assertAlmostEqual(stats.erf(0.5), 0.5204998778130465, places=12)

    def test_erf_is_odd(self):
        for x in (0.1, 0.9, 2.5):
            self.assertAlmostEqual(stats.erf(x), -stats.erf(-x), places=12)


class TestBeta(unittest.TestCase):
    def test_betainc_at_the_endpoints(self):
        self.assertAlmostEqual(stats.betainc(2.0, 3.0, 0.0), 0.0, places=12)
        self.assertAlmostEqual(stats.betainc(2.0, 3.0, 1.0), 1.0, places=12)

    def test_betainc_known_value(self):
        # I_{0.5}(1,1) = 0.5 by symmetry.
        self.assertAlmostEqual(stats.betainc(1.0, 1.0, 0.5), 0.5, places=12)

    def test_betainc_is_monotone(self):
        previous = -1.0
        for i in range(1, 50):
            value = stats.betainc(2.5, 1.5, i / 50.0)
            self.assertGreater(value, previous)
            previous = value


class TestStudentT(unittest.TestCase):
    def test_t_cdf_at_the_mean(self):
        self.assertAlmostEqual(stats.t_cdf(0.0, 5.0), 0.5, places=12)

    def test_t_cdf_is_symmetric(self):
        for t in (0.4, 1.1, 2.5):
            self.assertAlmostEqual(stats.t_cdf(t, 7.0) + stats.t_cdf(-t, 7.0),
                                   1.0, places=11)

    def test_t_cdf_tends_to_the_normal(self):
        # With a large number of degrees of freedom the Student-t CDF is
        # the normal CDF to within the difference of the two densities.
        for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
            self.assertAlmostEqual(stats.t_cdf(x, 5000.0),
                                   stats.norm_cdf(x), places=4)

    def test_t_cdf_known_value(self):
        # t with 10 degrees of freedom at 2.2281388519649385 is 0.975.
        self.assertAlmostEqual(stats.t_cdf(2.2281388519649385, 10.0), 0.975,
                               places=10)

    def test_t_ppf_inverts_t_cdf(self):
        for p in (0.01, 0.1, 0.5, 0.9, 0.99):
            t = stats.t_ppf(p, 12.0)
            self.assertAlmostEqual(stats.t_cdf(t, 12.0), p, places=10)

    def test_t_ppf_is_symmetric(self):
        for p in (0.02, 0.1, 0.25):
            self.assertAlmostEqual(stats.t_ppf(p, 6.0),
                                   -stats.t_ppf(1.0 - p, 6.0), places=10)

    def test_t_pdf_integrates_to_one(self):
        n, lo, hi = 200000, -60.0, 60.0
        h = (hi - lo) / n
        total = 0.0
        for i in range(n + 1):
            x = lo + i * h
            w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
            total += w * stats.t_pdf(x, 8.0)
        self.assertAlmostEqual(total * h / 3.0, 1.0, places=8)


class TestNumericalStability(unittest.TestCase):
    def test_norm_ppf_never_returns_infinity_in_range(self):
        for p in (1e-12, 1e-9, 1 - 1e-12):
            value = stats.norm_ppf(p)
            self.assertTrue(math.isfinite(value))

    def test_erf_does_not_overflow(self):
        self.assertAlmostEqual(stats.erf(40.0), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
