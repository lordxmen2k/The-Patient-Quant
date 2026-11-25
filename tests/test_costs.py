"""Tests for the tax model.

The cost of tax is the book's second subject: the system is reported twice,
once in a tax-exempt account and once in a taxable one, and the difference
between them is what the investor actually keeps.  These tests check the
rules the book states --- first-in first-out, the 30-day wash sale in both
directions, the $3,000 capital-loss offset, the holding-period rule and the
annual settlement --- against hand-worked cases.
"""

import unittest

import pandas as pd

from patientquant.costs import TaxableAccount, TaxSchedule, spread_cost


def day(year, month, day_number):
    return pd.Timestamp(year=year, month=month, day=day_number)


class TestTaxSchedule(unittest.TestCase):
    def test_dividend_rate_is_the_qualified_rate_plus_niit(self):
        schedule = TaxSchedule()
        self.assertAlmostEqual(schedule.marginal_on_dividend(), 0.188, places=6)

    def test_dividend_rate_follows_the_schedule(self):
        schedule = TaxSchedule(qualified_dividend=0.0, net_investment_income=0.0)
        self.assertAlmostEqual(schedule.marginal_on_dividend(), 0.0, places=9)


class TestWashSale(unittest.TestCase):
    """Section 1091: a loss sale followed by a purchase within 30 days."""

    def test_a_repurchase_disallows_the_loss(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        result = account.sell("SPY", 100.0, 90.0, day(2020, 2, 1))
        self.assertAlmostEqual(result["short"] + result["long"], -1000.0,
                               places=6)
        self.assertAlmostEqual(account.wash_sale_losses, 0.0, places=6)

        # Buying again inside the window disallows the loss and adds it to
        # the basis of the replacement shares.
        account.buy("SPY", 100.0, 95.0, day(2020, 2, 20))
        self.assertGreater(account.wash_sale_losses, 0.0)
        # The whole loss is disallowed, whatever the replacement shares
        # cost: the rule turns on the timing, not on the price.
        self.assertAlmostEqual(account.wash_sale_losses, 1000.0, places=6)
        # And the basis of the replacement shares is raised by the loss.
        self.assertAlmostEqual(account.lots["SPY"][-1].price, 105.0, places=6)

    def test_a_purchase_outside_the_window_is_clean(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        account.sell("SPY", 100.0, 90.0, day(2020, 2, 1))
        account.buy("SPY", 100.0, 95.0, day(2020, 4, 1))
        self.assertAlmostEqual(account.wash_sale_losses, 0.0, places=6)

    def test_a_purchase_before_the_sale_is_also_caught(self):
        """The rule runs in both directions: a purchase followed by a sale
        of the same fund inside 30 days is a wash as well."""
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        account.buy("SPY", 100.0, 100.0, day(2020, 3, 1))
        account.sell("SPY", 100.0, 90.0, day(2020, 3, 10))
        self.assertGreater(account.wash_sale_losses, 0.0)


class TestLotsAndGains(unittest.TestCase):
    def test_lots_are_matched_first_in_first_out(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        account.buy("SPY", 100.0, 120.0, day(2021, 1, 1))
        result = account.sell("SPY", 100.0, 130.0, day(2022, 1, 1))
        # The first lot is matched first, and it was bought in 2020 and
        # sold in 2022, so the gain is long-term.
        self.assertAlmostEqual(result["long"], 3000.0, places=6)
        self.assertAlmostEqual(result["short"], 0.0, places=6)

    def test_a_sale_inside_the_holding_period_is_short_term(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        result = account.sell("SPY", 100.0, 110.0, day(2020, 6, 1))
        self.assertAlmostEqual(result["short"], 1000.0, places=6)
        self.assertAlmostEqual(result["long"], 0.0, places=6)

    def test_a_sale_after_a_year_is_long_term(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        result = account.sell("SPY", 100.0, 110.0, day(2021, 1, 2))
        self.assertAlmostEqual(result["long"], 1000.0, places=6)
        self.assertAlmostEqual(result["short"], 0.0, places=6)


class TestSettlement(unittest.TestCase):
    def test_a_long_term_gain_pays_the_long_term_rate(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        account.sell("SPY", 100.0, 110.0, day(2021, 6, 1))
        tax = account.settle_year(2021)
        self.assertAlmostEqual(tax["total_tax"], 1000.0 * 0.15, places=4)

    def test_a_short_term_gain_pays_the_ordinary_rate(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        account.sell("SPY", 100.0, 110.0, day(2020, 6, 1))
        tax = account.settle_year(2020)
        self.assertAlmostEqual(tax["total_tax"], 1000.0 * 0.32, places=4)

    def test_a_net_loss_offsets_ordinary_income_up_to_the_cap(self):
        """A capital loss has no effect on the portfolio's own tax bill.

        It is usable against the investor's other income, up to $3,000 a
        year, and the rest is carried forward.  The model reports how much
        ordinary income the loss offsets rather than pretending the
        portfolio has any.
        """
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 1000.0, 100.0, day(2020, 1, 1))
        account.sell("SPY", 1000.0, 90.0, day(2020, 6, 1))
        tax = account.settle_year(2020)
        self.assertAlmostEqual(tax["total_tax"], 0.0, places=6)
        # $10,000 of loss: $3,000 offsets ordinary income, which at the
        # 32 per cent marginal rate is $960 of tax saved, and $7,000 is
        # carried forward.
        self.assertAlmostEqual(tax["ordinary_offset"], 3000.0 * 0.32, places=4)
        self.assertAlmostEqual(account.loss_carryforward, 7000.0, places=4)

    def test_a_carryforward_loss_reduces_the_next_year_gain(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 1000.0, 100.0, day(2020, 1, 1))
        account.sell("SPY", 1000.0, 90.0, day(2020, 6, 1))
        account.settle_year(2020)
        account.buy("SPY", 100.0, 100.0, day(2021, 1, 1))
        account.sell("SPY", 100.0, 110.0, day(2021, 6, 1))
        tax = account.settle_year(2021)
        # The $7,000 carried forward is applied to the $1,000 gain, so the
        # gain is wiped out.  The $3,000 cap then applies again in the new
        # year, so $3,000 is carried forward again.
        self.assertAlmostEqual(tax["total_tax"], 0.0, places=6)
        self.assertAlmostEqual(account.loss_carryforward, 3000.0, places=4)

    def test_a_tax_exempt_account_pays_nothing_but_still_clears(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        account.sell("SPY", 100.0, 110.0, day(2021, 6, 1))
        tax = account.settle_year(2021, pay_tax=False)
        self.assertAlmostEqual(tax["total_tax"], 0.0, places=6)

    def test_dividends_are_taxed_as_they_are_received(self):
        account = TaxableAccount()
        account.cash = 100_000.0
        account.buy("SPY", 100.0, 100.0, day(2020, 1, 1))
        account.receive_dividends({"SPY": 2.0})
        self.assertAlmostEqual(account.dividend_paid, 200.0, places=6)
        tax = account.settle_year(2020)
        self.assertAlmostEqual(tax["total_tax"], 200.0 * 0.188, places=4)


class TestSpreadCost(unittest.TestCase):
    def test_the_cost_is_the_weight_change_times_the_spread(self):
        changes = pd.Series({"SPY": 0.10, "TLT": -0.10})
        spreads = {"SPY": 0.0002, "TLT": 0.0004}
        cost = spread_cost(changes, spreads)
        # A one-way trade crosses half the spread; a round trip crosses
        # the whole of it.
        expected = (0.10 * 0.0002 + 0.10 * 0.0004) / 2.0
        self.assertAlmostEqual(cost, expected, places=12)

    def test_a_round_trip_costs_twice_as_much_as_a_one_way(self):
        changes = pd.Series({"SPY": 0.10})
        spreads = {"SPY": 0.0002}
        one_way = spread_cost(changes, spreads)
        two_way = spread_cost(pd.Series({"SPY": 0.20}), spreads)
        self.assertAlmostEqual(two_way, 2.0 * one_way, places=12)
        # The stated convention: the spread is crossed twice on a round
        # trip, so a one-way change of 10 per cent costs 5 per cent of a
        # spread.
        self.assertAlmostEqual(one_way, 0.05 * 0.0002, places=12)

    def test_no_change_costs_nothing(self):
        cost = spread_cost(pd.Series({"SPY": 0.0}), {"SPY": 0.0002})
        self.assertAlmostEqual(cost, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
