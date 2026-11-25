"""Tests for the selection tests.

The book's least flattering result is its selection test: the procedure
that chose the system is worse than a coin flip.  That result is only
worth reporting if the machinery that produces it is right, so these tests
check the machinery against cases whose answer is known.

The four tests are the probability of backtest overfitting by CSCV, the
deflated Sharpe ratio, the Hansen superior predictive ability test and the
White reality check.
"""

import unittest

import numpy as np
import pandas as pd

from patientquant import selection


def returns(n=1000, seed=3):
    """A series of monthly returns with no fat tails and no skew."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.005, 0.04, size=n)


def performance_matrix(n=240, m=6, seed=11, skill=0.0):
    """A matrix of monthly returns for ``m`` candidate strategies.

    ``skill`` is the monthly edge given to the first strategy; with no
    skill every candidate has the same distribution and the selection
    procedure has nothing to find.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.01, 0.04, size=(n, m))
    if skill:
        data[:, 0] += skill
    index = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.DataFrame(data, index=index,
                        columns=[f"s{i}" for i in range(m)])


class TestImpliedTrials(unittest.TestCase):
    def test_the_average_correlation_is_between_minus_one_and_one(self):
        for seed in (1, 2, 3, 4, 5):
            value = selection.average_correlation(performance_matrix(seed=seed))
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_identical_candidates_have_a_correlation_of_one(self):
        frame = performance_matrix()
        identical = pd.DataFrame(
            np.tile(frame.iloc[:, 0].to_numpy(), (4, 1)).T,
            columns=list("abcd"))
        self.assertAlmostEqual(selection.average_correlation(identical), 1.0,
                               places=6)

    def test_more_trials_means_fewer_independent_ones(self):
        self.assertLess(selection.implied_independent_trials(25, 0.99),
                        selection.implied_independent_trials(25, 0.50))
        self.assertLess(selection.implied_independent_trials(25, 0.50),
                        selection.implied_independent_trials(25, 0.0))

    def test_perfect_correlation_leaves_one_independent_trial(self):
        self.assertAlmostEqual(
            selection.implied_independent_trials(25, 1.0), 1.0, places=6)

    def test_no_correlation_leaves_every_trial_independent(self):
        self.assertAlmostEqual(
            selection.implied_independent_trials(25, 0.0), 25.0, places=6)


class TestPBO(unittest.TestCase):
    def test_the_probability_is_a_probability(self):
        result = selection.pbo_cscv(performance_matrix())
        self.assertGreaterEqual(result["pbo"], 0.0)
        self.assertLessEqual(result["pbo"], 1.0)

    def test_the_test_is_run_on_the_declared_family(self):
        matrix = performance_matrix()
        result = selection.pbo_cscv(matrix)
        self.assertEqual(result["n_trials"], matrix.shape[1])
        self.assertGreater(result["n_combinations"], 0)

    def test_a_matrix_with_no_skill_is_a_coin_flip(self):
        """With every candidate drawn from the same distribution there is
        nothing to select, so the procedure should land near one half.

        The estimate is noisy for a single draw, so the test averages over
        several independent families.
        """
        values = [selection.pbo_cscv(performance_matrix(m=10, seed=seed))["pbo"]
                  for seed in range(8)]
        mean = float(np.mean(values))
        self.assertGreater(mean, 0.25)
        self.assertLess(mean, 0.75)

    def test_the_result_reports_the_probability_above_zero(self):
        result = selection.pbo_cscv(performance_matrix())
        self.assertGreaterEqual(result["probability_above_zero"], 0.0)
        self.assertLessEqual(result["probability_above_zero"], 1.0)


class TestDeflatedSharpe(unittest.TestCase):
    def test_a_sharpe_above_the_benchmark_is_not_deflated_away(self):
        value = selection.deflated_sharpe(0.5, 0.1, 1000, returns())
        self.assertGreater(value, 0.9)

    def test_a_sharpe_at_the_benchmark_is_a_coin_flip(self):
        """The DSR is the probability that the Sharpe beats the expected
        maximum under the null, so at the benchmark it is one half."""
        at_benchmark = selection.deflated_sharpe(0.1, 0.1, 1000, returns())
        self.assertAlmostEqual(at_benchmark, 0.5, places=6)
        below = selection.deflated_sharpe(0.05, 0.1, 1000, returns())
        self.assertLess(below, 0.5)

    def test_more_trials_make_the_test_harder(self):
        """More candidate trials means a higher benchmark to beat."""
        one = selection.expected_max_sharpe(1.0, 1)
        many = selection.expected_max_sharpe(1.0, 25)
        self.assertGreater(many, one)
        easy = selection.deflated_sharpe(0.4, one, 1000, returns())
        hard = selection.deflated_sharpe(0.4, many, 1000, returns())
        self.assertLess(hard, easy)

    def test_more_observations_make_the_test_easier(self):
        short = selection.deflated_sharpe(0.3, 0.1, 100, returns(100))
        long = selection.deflated_sharpe(0.3, 0.1, 2000, returns(2000))
        self.assertGreater(long, short)

    def test_the_expected_maximum_sharpe_grows_with_the_trials(self):
        one = selection.expected_max_sharpe(1.0, 1)
        many = selection.expected_max_sharpe(1.0, 100)
        self.assertGreater(many, one)


class TestRealityChecks(unittest.TestCase):
    def test_the_white_reality_check_runs(self):
        matrix = performance_matrix()
        result = selection.white_reality_check(matrix)
        self.assertGreaterEqual(result["pvalue"], 0.0)
        self.assertLessEqual(result["pvalue"], 1.0)
        self.assertIn("statistic", result)

    def test_the_hansen_spa_runs(self):
        matrix = performance_matrix()
        result = selection.hansen_spa(matrix)
        self.assertGreaterEqual(result["pvalue"], 0.0)
        self.assertLessEqual(result["pvalue"], 1.0)

    def test_a_best_strategy_that_beats_the_others(self):
        """With one candidate clearly ahead, the tests should not report
        that nothing is better than the average."""
        matrix = performance_matrix(skill=0.03)
        spa = selection.hansen_spa(matrix)
        self.assertLess(spa["pvalue"], 0.5)


class TestStationaryBootstrap(unittest.TestCase):
    def test_the_indices_stay_in_range(self):
        rng = np.random.default_rng(7)
        indices = selection.stationary_bootstrap_indices(100, 10.0, rng)
        self.assertEqual(len(indices), 100)
        self.assertGreaterEqual(min(indices), 0)
        self.assertLess(max(indices), 100)


if __name__ == "__main__":
    unittest.main()
