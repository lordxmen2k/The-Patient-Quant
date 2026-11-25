"""Tests for the panel and the features.

Two failures are worth more than the whole system, and both are checked
here:

  * a fund that did not exist yet must not carry a zero return for the
    months before its inception, because a zero return makes a fund look
    like it had no volatility and no beta, which is the most attractive
    possible appearance for a new fund to have;
  * the target of a training row must not be knowable at the date the row
    is dated, so the horizon has to be respected.
"""

import os
import unittest

import numpy as np
import pandas as pd

from patientquant import features, models, panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache", "manifest.json")

HAVE_CACHE = os.path.exists(CACHE)


@unittest.skipUnless(HAVE_CACHE, "the data cache is not in this checkout")
class TestPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = panel.monthly_panel()

    def test_the_panel_has_every_fund(self):
        self.assertGreaterEqual(self.panel["returns"].shape[1], 60)

    def test_the_panel_covers_the_whole_history(self):
        returns = self.panel["returns"]
        self.assertGreaterEqual(len(returns), 400)

    def test_a_fund_is_not_credited_with_zero_returns_before_it_existed(self):
        """The newest funds are the test: they have the fewest months, so
        a zero-filled history would make them look like they had neither
        volatility nor beta."""
        returns = self.panel["returns"]
        for ticker in returns.columns:
            series = returns[ticker].dropna()
            if len(series) < 12:
                continue
            # The first month of a fund's life may be zero (the vendor
            # reports it so); after that, a month of exactly zero for
            # every fund at once is the signature of a zero fill.
            self.assertFalse(series.iloc[1:].eq(0.0).all(),
                             f"{ticker} looks zero-filled")

    def test_the_live_window_is_not_zero_filled(self):
        returns = self.panel["returns"]
        last = returns.iloc[-1]
        live = last.dropna()
        self.assertGreater(len(live), 10)
        self.assertFalse(live.eq(0.0).all())

    def test_the_panel_is_reproducible(self):
        again = panel.monthly_panel()
        self.assertTrue(np.allclose(
            self.panel["returns"].fillna(0.0).to_numpy(),
            again["returns"].fillna(0.0).to_numpy(), equal_nan=True))


@unittest.skipUnless(HAVE_CACHE, "the data cache is not in this checkout")
class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = panel.monthly_panel()
        cls.raw = features.build_features(cls.panel)

    def test_every_feature_is_defined_for_every_fund_that_exists(self):
        for name, frame in self.raw.items():
            returns = self.panel["returns"][frame.columns]
            for ticker in frame.columns:
                series = frame[ticker].dropna()
                if len(series) == 0:
                    continue
                # A feature cannot be known before the fund has a price.
                first = returns[ticker].dropna().index[0]
                self.assertGreaterEqual(series.index[0], first,
                                        f"{name} is known before {ticker} exists")

    def test_a_feature_is_never_defined_for_a_fund_with_no_history(self):
        for name, frame in self.raw.items():
            returns = self.panel["returns"][frame.columns]
            for ticker in frame.columns:
                if returns[ticker].dropna().empty:
                    self.assertTrue(frame[ticker].dropna().empty,
                                    f"{name} has values for {ticker} "
                                    "before it existed")

    def test_the_target_is_the_forward_return(self):
        target = features.build_target(self.panel, horizon=3)
        returns = self.panel["returns"]
        dates = target.index
        for ticker in list(target.columns)[:5]:
            series = target[ticker].dropna()
            if len(series) == 0:
                continue
            date = series.index[-1]
            position = dates.get_loc(date)
            if position + 3 >= len(dates):
                continue
            forward = returns[ticker].loc[dates[position + 1:position + 4]]
            expected = float(np.prod(1.0 + forward.to_numpy(dtype=float)) - 1.0)
            self.assertAlmostEqual(float(series.iloc[-1]), expected, places=10)


@unittest.skipUnless(HAVE_CACHE, "the data cache is not in this checkout")
class TestNoLookAhead(unittest.TestCase):
    def test_the_training_set_ends_before_the_horizon_boundary(self):
        """Independently rebuild the eligible training rows for a
        formation date and compare the count with the model's."""
        built = panel.monthly_panel()
        raw = features.build_features(built)
        scaled = features.cross_sectional_scale(raw)
        config = models.Config(name="full", horizon=3)
        target = features.build_target(built, config.horizon)
        frame = models.walk_forward(scaled, target, config,
                                    min_months=36, min_rows=180)
        self.assertGreater(len(frame), 0)

        cols = list(config.features)
        stacked = pd.concat(
            [scaled[name].stack(future_stack=True).rename(name) for name in cols],
            axis=1).dropna()
        stacked = stacked.reindex(sorted(stacked.index,
                                         key=lambda ix: (ix[0], ix[1])))
        y_long = target.stack(future_stack=True).reindex(stacked.index)
        stacked = stacked[y_long.notna()]
        positions = target.index.get_indexer(stacked.index.get_level_values(0))
        recorded = frame.set_index(["date", "ticker"])["n_train"].to_dict()

        for date in frame["date"].unique()[:12]:
            i = int(target.index.get_indexer([pd.Timestamp(date)])[0])
            eligible = int((positions <= i - config.horizon).sum())
            got = recorded.get((pd.Timestamp(date),
                                frame[frame["date"] == date]["ticker"].iloc[0]))
            self.assertIsNotNone(got)
            self.assertEqual(int(got), eligible)


if __name__ == "__main__":
    unittest.main()
