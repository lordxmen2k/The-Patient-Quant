"""Fund-level facts: expense ratio, inception date, net assets, yield, turnover.

These are not prices.  They are the *terms* of the instruments, and they are
the numbers an investor pays whether the model is right or wrong.

They come from Yahoo's ``quoteSummary`` endpoint, which requires a session
cookie and a crumb token.  Both are obtained once per run and reused, and
every response is cached with its SHA-256 so that the terms used in the book
can be re-derived from the bytes that were fetched.

Two of these fields matter more than the rest:

``expense_ratio``
    A certain, unavoidable drag.  It is paid whether the model works or not.

``inception_date``
    The boundary between *live* evidence and *backtest* evidence.  A factor
    premium published in 1997 has fifteen years of investable ETF history
    behind it, not twenty-nine.  A book that draws no line here is selling
    a backtest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache" / "fundamentals"
MANIFEST_PATH = REPO_ROOT / "data" / "cache" / "fundamentals_manifest.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

MODULES = "defaultKeyStatistics,fundProfile,summaryDetail"


@dataclass
class Fundamentals:
    ticker: str
    expense_ratio: float | None = None
    inception_date: str | None = None
    total_net_assets: float | None = None
    annual_turnover: float | None = None
    dividend_yield: float | None = None
    category: str | None = None
    family: str | None = None
    sha256: str | None = None

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "expense_ratio": self.expense_ratio,
            "inception_date": self.inception_date,
            "total_net_assets": self.total_net_assets,
            "annual_turnover": self.annual_turnover,
            "dividend_yield": self.dividend_yield,
            "category": self.category,
            "family": self.family,
        }


def _run(cmd: list[str], timeout: int = 45) -> bytes:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:2])} -> {result.stderr.decode()[:160]}")
    return result.stdout


def _cookie_jar() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jar = CACHE_DIR / "_cookies.txt"
    if not jar.exists():
        _run(["curl", "-sS", "-c", str(jar), "-A", USER_AGENT, "https://fc.yahoo.com"])
    return jar


def _crumb(jar: Path) -> str:
    out = _run([
        "curl", "-sS", "-b", str(jar), "-A", USER_AGENT,
        "-H", "Accept: application/json,text/plain,*/*",
        "-H", "Referer: https://finance.yahoo.com/",
        "https://query2.finance.yahoo.com/v1/test/getcrumb",
    ])
    token = out.decode("utf-8").strip()
    if not token or token == "null":
        raise RuntimeError("could not obtain a Yahoo crumb token")
    return token


def _raw_value(node, key: str):
    """Pull ``key`` out of Yahoo's ``{'raw': x, 'fmt': y}`` wrappers."""
    if not isinstance(node, dict):
        return None
    value = node.get(key)
    if isinstance(value, dict):
        return value.get("raw")
    return value


def fetch(ticker: str, *, refresh: bool = False, pause: float = 1.2) -> Fundamentals:
    """Fetch and cache the fund-level facts for one ticker."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ticker}.json"

    if path.exists() and not refresh:
        return _parse(ticker, json.loads(path.read_text()))

    jar = _cookie_jar()
    token = _crumb(jar)
    url = (
        "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{ticker}?modules={MODULES}&crumb={token}"
    )
    raw = _run([
        "curl", "-sS", "-b", str(jar), "-A", USER_AGENT,
        "-H", "Accept: application/json,text/plain,*/*",
        "-H", "Referer: https://finance.yahoo.com/",
        url,
    ])
    payload = json.loads(raw.decode("utf-8"))
    path.write_bytes(raw)
    time.sleep(pause)
    return _parse(ticker, payload)


def _parse(ticker: str, payload: dict) -> Fundamentals:
    result = (payload.get("quoteSummary", {}).get("result") or [{}])[0]
    stats = result.get("defaultKeyStatistics", {}) or {}
    profile = result.get("fundProfile", {}) or {}
    fees = profile.get("feesExpensesInvestment", {}) or {}
    summary = result.get("summaryDetail", {}) or {}

    inception = _raw_value(stats, "fundInceptionDate")
    inception_date = (
        time.strftime("%Y-%m-%d", time.gmtime(inception)) if inception else None
    )
    net_assets = _raw_value(fees, "totalNetAssets")
    if net_assets is not None:
        net_assets = float(net_assets) * 1e6  # Yahoo reports USD millions

    return Fundamentals(
        ticker=ticker,
        expense_ratio=_raw_value(fees, "annualReportExpenseRatio"),
        inception_date=inception_date,
        total_net_assets=net_assets,
        annual_turnover=_raw_value(fees, "annualHoldingsTurnover"),
        dividend_yield=_raw_value(summary, "trailingAnnualDividendYield"),
        category=profile.get("categoryName"),
        family=profile.get("family"),
        sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    )


def load_all(tickers, *, refresh: bool = False, pause: float = 1.2) -> list[Fundamentals]:
    """Fetch fundamentals for a list of tickers, skipping failures."""
    out: list[Fundamentals] = []
    for ticker in tickers:
        try:
            out.append(fetch(ticker, refresh=refresh, pause=pause))
        except Exception as exc:  # noqa: BLE001
            print(f"  {ticker:<6} fundamentals failed: {exc}")
    return out
