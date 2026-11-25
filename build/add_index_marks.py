#!/usr/bin/env python3
"""
Insert index marks into the book chapters.

The placement engine lives in build/index_marks.py.  This script holds the
table of terms, the chapter each is indexed from, and writes the marks.

Run from the repository root:  python3 build/add_index_marks.py
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from index_marks import add_mark  # noqa: E402

CHAPTERS = pathlib.Path("source/chapters")

# (chapter number, canonical index entry)
#
# The order within a chapter matters: a shorter term that is a substring of
# a longer one in the same chapter must come after it, or the longer term
# finds its occurrence already marked and is lost.
MARKS = [
    # Part I — the question
    (1, "Single investor"), (1, "Principal"), (1, "Machine learning"),
    (1, "Portfolio"), (1, "Investor"), (1, "Alpha"),
    (2, "Arithmetic mean"), (2, "Geometric mean"), (2, "Volatility"),
    (2, "Compound growth"), (2, "Drawdown"), (2, "Variance"),
    (2, "Maximum drawdown"),
    (3, "Capital asset pricing model"), (3, "Factor"),
    (3, "Replication"), (3, "Active management"),
    (3, "Decay"), (3, "Posterior uncertainty"),
    (4, "Passive"), (4, "Multiple testing"), (4, "Data snooping"),
    (4, "Backtest overfitting"), (4, "Standard error"),
    (4, "Replication crises"), (4, "Momentum"),
    (5, "DBOT"), (5, "Language model"), (5, "Deep reinforcement learning"),
    (5, "Regime robustness"), (5, "Macro"), (5, "White space"),
    # Part II — the evidence
    (6, "Risk premium"), (6, "Premium"), (6, "Equities"), (6, "Dividend"),
    (7, "Sequence risk"), (7, "Withdrawal rate"),
    (7, "Ruin probability"), (7, "Samuelson"),
    (8, "Myopic loss aversion"), (8, "Evaluation horizon"),
    (8, "Equity premium"), (8, "Prospect theory"), (8, "Loss aversion"),
    (9, "Disposition effect"), (9, "Overtrading"), (9, "Herding"),
    (9, "Information cascade"), (9, "Illusion of control"),
    (9, "Odean"), (9, "No-trade band"),
    (10, "Bid-ask spread"), (10, "Commissions"), (10, "Turnover"),
    (10, "Expense ratio"), (10, "Spread"),
    (10, "Short-term gain"), (10, "Qualified dividend"),
    (10, "Exchange-traded fund"),
    (11, "Capacity"), (11, "Market impact"), (11, "Institution"),
    (11, "Size"),
    # Part III — the data
    (12, "Survivorship bias"), (12, "Revision"), (12, "Vendor"),
    (12, "Cache"), (12, "Inception"), (12, "Point-in-time data"),
    (12, "Constraint"),
    (13, "Live window"), (13, "Universe"), (13, "Fund"), (13, "Leverage"),
    (14, "Momentum"), (14, "Reversal"), (14, "Volatility"),
    (14, "Beta"), (14, "Correlation"), (14, "Feature"),
    (14, "Volume"), (14, "Moving average"), (14, "Liquidity"),
    (15, "Forward return"), (15, "Target"), (15, "Horizon"),
    (16, "Look-ahead"), (16, "Leakage"), (16, "Purge"),
    (16, "Embargo"), (16, "Survivorship"),
    # Part IV — the model
    (17, "Panel regression"), (17, "Conjugate prior"), (17, "Posterior"),
    (17, "Prior"), (17, "Conjugate update"),
    (18, "Normal-inverse-gamma"), (18, "Hyperparameters"),
    (18, "Posterior predictive"),
    (19, "Model averaging"), (19, "Ensemble"), (19, "Half-life"),
    (19, "Cumulative"),
    (20, "Edge"), (20, "Uncertainty"), (20, "Calibration"),
    (20, "Posterior distribution"), (20, "Evidence"),
    (20, "Belief"),
    (21, "Lo's correction"), (21, "Autocorrelation"),
    (21, "Standard error"), (21, "Sharpe ratio"),
    (22, "Deflated Sharpe ratio"),
    (22, "Probability of backtest overfitting"),
    (22, "Reality check"), (22, "Superior predictive ability"),
    (22, "Combinatorially symmetric cross-validation"),
    (22, "Bootstrap"), (22, "Cross-validation"), (22, "Alignment"),
    # Part V — the portfolio
    (23, "Strategic allocation"), (23, "Policy"), (23, "Sleeve"),
    (24, "Bonds"), (24, "Real assets"), (24, "Diversification"),
    (25, "Ledoit and Wolf"), (25, "Shrinkage intensity"),
    (25, "Hierarchical risk parity"), (25, "Risk parity"),
    (25, "Equal weighting"), (25, "Optimisation"),
    (25, "Covariance"), (25, "Compounding"),
    (26, "Kelly criterion"), (26, "Certainty-equivalent"),
    (26, "Uncertain mean"), (26, "Growth"),
    (27, "Tilt"), (27, "Sizing"), (27, "Bound"),
    (28, "Volatility scaling"), (28, "Drawdown brake"), (28, "Damping"),
    (28, "Exposure"), (28, "Governor"),
    # Part VI — the system
    (29, "Wash sale"), (29, "Holding period"), (29, "Realised gains"),
    (29, "Tax"),
    (29, "Lot"), (29, "Capital gains"), (29, "Net investment income tax"),
    (29, "Tax schedule"), (29, "Capital losses"),
    (30, "System"), (30, "Monthly loop"), (30, "Cash"),
    (31, "Attribution"), (31, "Layer"), (31, "Contribution"),
    (31, "Turnover"), (31, "Ablation"),
    (32, "Record"), (32, "Compound growth"), (32, "Recovery"),
    (32, "Comparator"), (32, "Benchmark"), (32, "Rebalancing"),
    (33, "Trial family"), (33, "Implied number of independent trials"),
    (33, "Reality check"), (33, "Deflated Sharpe ratio"),
    (34, "Window sensitivity"), (34, "Starting month"),
    (34, "Sensitivity"),
    (35, "Taxable account"), (35, "Tax-exempt account"),
    (35, "Long-term gain"), (35, "No-trade band"),
    (36, "Regime"), (36, "Failure mode"), (36, "Wrong"),
    (36, "Override"), (36, "Discounted score"), (36, "Tail risk"),
    (36, "Threshold"),
    (37, "Regulation FD"), (37, "Pattern day trader"), (37, "Margin"),
    (37, "Insider trading"), (37, "Rule"), (37, "Law"),
    (38, "Recommendation"), (38, "Verdict"), (38, "Advice"),
]


def strip_marks(text: str) -> str:
    """Remove every index mark already in the text.

    The script is idempotent this way: it strips first and then places,
    so it can be re-run after the table of terms changes without the
    marks accumulating.
    """
    return re.sub(r"\\idx\{[^{}]*\}", "", text)


def main() -> None:
    total_added = 0
    missing = []
    by_chapter: dict[int, list[str]] = {}
    for num, term in MARKS:
        by_chapter.setdefault(num, []).append(term)

    for num, terms in sorted(by_chapter.items()):
        path = CHAPTERS / f"ch{num:02d}.tex"
        if not path.exists():
            for term in terms:
                missing.append((num, term))
            continue
        text = strip_marks(path.read_text(encoding="utf-8"))
        for term in terms:
            text, ok = add_mark(text, term)
            if ok:
                total_added += 1
            else:
                missing.append((num, term))
        path.write_text(text, encoding="utf-8")

    print(f"index marks added: {total_added}")
    if missing:
        print("terms not found (and therefore not marked):")
        for num, term in missing:
            print(f"  ch{num:02d}: {term}")


if __name__ == "__main__":
    main()
