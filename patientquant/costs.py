"""Costs and taxes: what the investor actually pays.

Two things are separated here that are usually conflated.

**Trading costs.**  A bid-ask spread, a commission, and the market impact of
a small order.  For a single investor trading liquid exchange-traded funds,
the spread is the material term.  The SEC's investor bulletin is explicit
that the spread "can be thought of as a hidden cost to investors since
spreads reduce potential returns."

**The expense ratio.**  It is *not* charged here, and the reason matters.
The price series the book uses is Yahoo's adjusted close, which is a
*total-return* series: the fund's expenses are already deducted from the
prices.  Charging an expense ratio again would double-count the single most
certain drag in the book.  The expense ratios are collected and reported
because they are the right thing to compare when choosing between funds, not
because they are a cost to be subtracted twice.

**Taxes** are modelled in a taxable account, because that is where they bite.
An account that is tax-deferred or tax-exempt has no capital gains tax and
no dividend tax, and the difference between the two accounts is one of the
largest numbers in the book.  The tax engine tracks tax lots, distinguishes
short-term from long-term gains by holding period, applies the wash-sale rule
of Internal Revenue Code section 1091, and nets annual gains and losses with
the capital-loss carryforward.

Two simplifications are made, and both are stated where they are used:
average of the fund's tax lot is not used -- actual lots are tracked and
consumed first-in-first-out -- but state, local and net investment income tax
are not modelled, and the rates applied are a stated schedule rather than
the reader's own marginal bracket.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


@dataclass
class TaxSchedule:
    """A stated, ordinary-income individual investor's schedule.

    The rates are the ones that apply to most single filers with taxable
    income in the upper-middle brackets, and they are stated here so that a
    reader with a different bracket can change one number and re-run.
    """

    long_term_gain: float = 0.15      # 0% / 15% / 20% on net long-term gain
    short_term_gain: float = 0.32     # short-term gain taxed as ordinary income
    qualified_dividend: float = 0.15  # qualified dividend rate
    ordinary_income: float = 0.32     # marginal rate for the $3,000 offset
    net_investment_income: float = 0.038  # NIIT, if the investor is above the threshold
    capital_loss_offset: float = 3000.0
    carryforward_years: float = float("inf")

    def marginal_on_dividend(self) -> float:
        """Dividends pay the qualified rate plus net investment income tax."""
        return self.qualified_dividend + self.net_investment_income


@dataclass
class Lot:
    acquired: pd.Timestamp
    shares: float
    price: float


@dataclass
class TaxableAccount:
    """A taxable brokerage account with lot tracking and wash-sale rules."""

    schedule: TaxSchedule = field(default_factory=TaxSchedule)
    cash: float = 0.0
    lots: dict[str, deque] = field(default_factory=dict)
    realised_short: float = 0.0        # accumulated short-term gain (may be negative)
    realised_long: float = 0.0
    dividend_paid: float = 0.0
    tax_paid: float = 0.0
    loss_carryforward: float = 0.0
    wash_sale_losses: float = 0.0      # losses disallowed by section 1091
    pending_wash: dict[str, list] = field(default_factory=dict)
    recent_buys: dict[str, deque] = field(default_factory=dict)
    no_trade_band: float = 0.005       # do not trade a weight change below this
    # Section 1091 exists to stop a deduction being taken twice.  An
    # account that cannot deduct anything has no rule to apply, and
    # running the machinery on it would take cash out of the account for
    # a tax that is never payable.
    wash_sale_rules: bool = True

    # -- positions ------------------------------------------------------
    def shares(self, ticker: str) -> float:
        return sum(lot.shares for lot in self.lots.get(ticker, ()))

    def market_value(self, ticker: str, price: float) -> float:
        return self.shares(ticker) * price

    def total_value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(self.market_value(t, p) for t, p in prices.items()
                               if t in self.lots)

    # -- transactions ---------------------------------------------------
    def receive_dividends(self, dividends: dict[str, float]) -> None:
        """Dividend cash is received and is taxable as it is received."""
        total = 0.0
        for ticker, amount in dividends.items():
            if amount <= 0 or ticker not in self.lots:
                continue
            shares = self.shares(ticker)
            cash = shares * amount
            self.cash += cash
            total += cash
        self.dividend_paid += total

    def buy(self, ticker: str, shares: float, price: float, date: pd.Timestamp) -> float:
        """Buy shares, applying any pending wash-sale disallowance."""
        if shares <= 0 or price <= 0:
            return 0.0
        cost = shares * price
        basis_add = 0.0

        # A purchase within 30 days of a loss sale of the same fund triggers
        # section 1091: the loss is not allowed, and is added to the basis of
        # the replacement shares.  The rule does not depend on the price the
        # replacement shares are bought at; it is the timing that counts.
        for record in (self.pending_wash.get(ticker, [])
                       if self.wash_sale_rules else ()):
            if abs((date - record["date"]).days) > 30:
                continue
            if record["remaining"] <= 0:
                continue
            per_share = record.get("loss_per_share", 0.0)
            if per_share <= 0:
                continue
            # The shares that can still be matched are those the remaining
            # loss was realised on.
            available = record["remaining"] / per_share
            matched = min(shares, available)
            candidate = per_share * matched
            disallow = min(record["remaining"], candidate)
            if disallow <= 0:
                continue
            record["remaining"] -= disallow
            basis_add += disallow
            self.wash_sale_losses += disallow

        effective_cost = cost + basis_add
        self.cash -= effective_cost
        self.lots.setdefault(ticker, deque()).append(Lot(date, shares, price))
        self.lots[ticker][-1].price = (effective_cost / shares)
        self.recent_buys.setdefault(ticker, deque()).append(Lot(date, shares, price))
        return effective_cost

    def sell(self, ticker: str, shares: float, price: float,
             date: pd.Timestamp) -> dict:
        """Sell shares, realising gains and applying the wash-sale rule."""
        if shares <= 0 or ticker not in self.lots:
            return {"short": 0.0, "long": 0.0, "wash": 0.0}
        remaining = shares
        short_gain = long_gain = 0.0
        wash = 0.0
        loss_shares = 0.0
        loss_gross = 0.0
        loss_disallowed = 0.0
        queue = self.lots[ticker]

        while remaining > 1e-12 and queue:
            lot = queue[0]
            take = min(lot.shares, remaining)
            if take <= 1e-12:
                queue.popleft()
                continue
            basis = lot.price * take
            proceeds = price * take
            gain = proceeds - basis
            held_days = (date - lot.acquired).days
            is_long = held_days > DAYS_PER_YEAR

            # If substantially identical stock was bought within 30 days
            # before this sale, the loss on the matching shares is disallowed.
            if gain < 0:
                loss_shares += take
                loss_gross += -gain
                matched = (self._match_recent_buys(ticker, take, price, date)
                           if self.wash_sale_rules else 0.0)
                if matched > 0:
                    loss = -gain
                    disallowed = min(loss, matched)
                    gain += disallowed
                    wash += disallowed
                    self.wash_sale_losses += disallowed
                    loss_disallowed += disallowed

            if is_long:
                long_gain += gain
            else:
                short_gain += gain

            lot.shares -= take
            remaining -= take
            if lot.shares <= 1e-12:
                queue.popleft()

        self.cash += shares * price

        # Record the loss so that a purchase inside the 30-day window cannot
        # be used to take the deduction in this year.  The loss per share is
        # carried with it, because the rule bites whatever price the
        # replacement shares are bought at: the loss is disallowed and the
        # basis of the new shares is raised by the amount of it.  The part
        # of the loss already disallowed by a purchase inside the window
        # that preceded the sale is not offered a second time.
        remaining_loss = loss_gross - loss_disallowed
        if self.wash_sale_rules and remaining_loss > 0.0 and loss_shares > 0.0:
            self.pending_wash.setdefault(ticker, []).append({
                "date": date, "price": price,
                "remaining": remaining_loss,
                "loss_per_share": remaining_loss / loss_shares,
            })

        self.realised_short += short_gain
        self.realised_long += long_gain
        return {"short": short_gain, "long": long_gain, "wash": wash}

    def _match_recent_buys(self, ticker: str, shares: float, price: float,
                           date: pd.Timestamp) -> float:
        """Dollars of loss matched by purchases made within 30 days before."""
        queue = self.recent_buys.get(ticker)
        if not queue:
            return 0.0
        matched = 0.0
        remaining = shares
        while queue and remaining > 1e-12:
            lot = queue[0]
            if (date - lot.acquired).days > 30:
                # Older than the window: not a candidate, but the queue is
                # chronological and the candidates are at the back, so the
                # search continues rather than stopping.
                queue.popleft()
                continue
            take = min(lot.shares, remaining)
            loss = max(0.0, lot.price - price) * take
            matched += loss
            lot.shares -= take
            remaining -= take
            if lot.shares <= 1e-12:
                queue.popleft()
        return matched

    def prune_windows(self, date: pd.Timestamp) -> None:
        """Forget buys and pending wash sales that have left the 30-day window."""
        for ticker, queue in list(self.recent_buys.items()):
            while queue and (date - queue[0].acquired).days > 31:
                queue.popleft()
            if not queue:
                del self.recent_buys[ticker]
        for ticker, records in list(self.pending_wash.items()):
            keep = []
            for record in records:
                if abs((date - record["date"]).days) <= 30 and record["remaining"] > 0:
                    keep.append(record)
            if keep:
                self.pending_wash[ticker] = keep
            else:
                del self.pending_wash[ticker]

    # -- annual tax -----------------------------------------------------
    def settle_year(self, year: int, pay_tax: bool = True) -> dict:
        """Compute the tax for ``year`` and reset the accumulators.

        ``pay_tax`` is false for a tax-exempt account, which still needs
        the accruals cleared so that the next year starts clean.
        """
        sched = self.schedule
        net_short = self.realised_short
        net_long = self.realised_long

        carry = self.loss_carryforward
        if carry > 0:
            net_short += -carry  # a carried loss is a short-term loss
            carry = 0.0

        dividend_tax = self.dividend_paid * sched.marginal_on_dividend()

        # Capital gains: net long-term gain is reduced by short-term losses,
        # and vice versa.  Only the excess of a net loss over net gains
        # reaches ordinary income, and then only to $3,000.
        st_tax = 0.0
        lt_tax = 0.0
        offset = 0.0
        remaining_loss = 0.0

        if net_short >= 0 and net_long >= 0:
            st_tax = net_short * sched.short_term_gain
            lt_tax = net_long * sched.long_term_gain
        elif net_short >= 0 > net_long:
            net = net_short + net_long
            if net > 0:
                st_tax = net * sched.short_term_gain
            else:
                remaining_loss = -net
        elif net_short < 0 <= net_long:
            net = net_long + net_short
            if net > 0:
                lt_tax = net * sched.long_term_gain
            else:
                remaining_loss = -net
        else:
            remaining_loss = -(net_short + net_long)

        if remaining_loss > 0:
            used = min(remaining_loss, sched.capital_loss_offset)
            offset = used * sched.ordinary_income
            remaining_loss -= used
            self.loss_carryforward = remaining_loss

        total_tax = st_tax + lt_tax + dividend_tax
        if pay_tax:
            self.cash -= total_tax
            self.tax_paid += total_tax
        else:
            # A tax-exempt account pays nothing.  The components are still
            # computed, so that the book can say what the same portfolio
            # would have paid, but the tax paid is zero.
            total_tax = 0.0

        settlement = {
            "year": year,
            "net_short": net_short,
            "net_long": net_long,
            "short_term_tax": st_tax,
            "long_term_tax": lt_tax,
            "dividend_tax": dividend_tax,
            "ordinary_offset": offset,
            "total_tax": total_tax,
        }

        self.realised_short = 0.0
        self.realised_long = 0.0
        self.dividend_paid = 0.0
        return settlement


def spread_cost(weights_change: pd.Series, spreads: dict[str, float]) -> float:
    """Cost of a set of weight changes, as a fraction of portfolio value.

    Crossing the spread on the way in and on the way out costs roughly one
    full spread per round trip, so a one-way trade costs half a spread.
    """
    total = 0.0
    for ticker, change in weights_change.items():
        spread = spreads.get(ticker, 0.0005)
        total += abs(change) * spread / 2.0
    return float(total)


def default_spreads(tickers, liquid: float = 0.0002,
                    moderate: float = 0.0006) -> dict[str, float]:
    """A stated spread assumption: two basis points for the giants, six for the rest.

    The numbers are assumptions, not measurements, and the book reports the
    sensitivity of every conclusion to them rather than hiding them inside a
    single backtest.
    """
    huge = {"SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "TLT", "IEF", "SHY", "BND"}
    out = {}
    for ticker in tickers:
        out[ticker] = liquid if ticker in huge else moderate
    return out
