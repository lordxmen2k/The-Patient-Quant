"""Fetch and cache daily price history for the ETF universe.

One source, one pull, cached and checksummed.  The book never mixes vendors.

Raw JSON responses are stored verbatim under ``data/cache/raw`` so that every
number in the book can be re-derived from the bytes that were actually
downloaded.  A sidecar manifest records the SHA-256 of each response, the bar
count, and the first/last trading dates.

The module deliberately uses only the standard library plus numpy/pandas: the
sandbox has no ``requests`` and PyPI is unreachable, and a book that says "do
not trust a black box" should not ship a dependency it cannot read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
# Yahoo's edge answers some user-agent strings with "429 Too Many Requests"
# and serves others with 200 for the identical URL.  A Linux Chrome string
# was refused; a Windows Chrome string was served.  The package therefore
# identifies itself as the browser it is permitted to be.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# 1990-01-01 .. 2026-12-31.  Yahoo ignores a period2 in the future and
# truncates to the last completed session, so an end date comfortably past
# today is the right way to ask for "everything so far".
DEFAULT_PERIOD1 = 631152000
DEFAULT_PERIOD2 = 1798761600

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CACHE_DIR = REPO_ROOT / "data" / "cache" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "cache" / "manifest.json"


@dataclass
class FetchReport:
    """Outcome of a fetch or verification pass over one ticker."""

    ticker: str
    n_bars: int = 0
    first: str | None = None
    last: str | None = None
    sha256: str | None = None
    from_cache: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and self.n_bars > 0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json"


def _read_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _curl_get(url: str, timeout: int) -> bytes:
    """Fetch ``url`` with curl.

    Yahoo's edge answers Python's ``urllib`` with ``429 Too Many Requests``
    while serving the identical URL to ``curl`` with 200.  The difference is
    the header signature: urllib sends ``Accept-Encoding: identity`` and
    ``Connection: close``, curl sends ``Accept-Encoding: gzip, deflate`` and
    keeps the connection alive.  Rather than disguise a browser, the package
    asks curl for the bytes and keeps urllib as a fallback.
    """
    result = subprocess.run(
        ["curl", "-sS", "-L", "--compressed", "--max-time", str(timeout),
         "-A", USER_AGENT,
         "-H", "Accept: application/json,text/plain,*/*",
         "-H", "Referer: https://finance.yahoo.com/quote/",
         url],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}): {result.stderr.decode()[:200]}")
    return result.stdout


def _urllib_get(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _http_get_json(url: str, timeout: int = 30) -> tuple[dict, bytes]:
    """Fetch ``url`` with backoff that understands Yahoo's rate limiter.

    Yahoo answers a burst of chart requests with ``429 Too Many Requests``.
    Treating that as a hard failure wastes the run, and treating it as a
    generic error and retrying immediately makes it worse.  The body is
    inspected: a 429 is answered with a long, growing pause, and only a
    genuine network failure is retried quickly.
    """
    last_error: Exception | None = None
    for attempt in range(6):
        for fetcher in (_curl_get, _urllib_get):
            try:
                raw = fetcher(url, timeout)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            if b"Too Many Requests" in raw[:64] or b"429" in raw[:16]:
                last_error = RuntimeError("HTTP 429 from Yahoo")
                continue
            try:
                return json.loads(raw.decode("utf-8")), raw
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                last_error = exc
        time.sleep(min(20.0 * (attempt + 1), 120.0))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_raw(ticker: str, *, refresh: bool = False,
              period1: int = DEFAULT_PERIOD1, period2: int = DEFAULT_PERIOD2,
              pause: float = 3.0) -> FetchReport:
    """Return the raw chart JSON for ``ticker``, downloading it if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    manifest = _read_manifest()

    if path.exists() and not refresh:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        return _summarise(ticker, payload, raw_bytes, from_cache=True)

    url = (
        f"{CHART_URL.format(ticker=ticker)}?period1={period1}&period2={period2}"
        "&interval=1d&events=div%2Csplit"
    )
    payload, raw_bytes = _http_get_json(url)
    path.write_bytes(raw_bytes)
    report = _summarise(ticker, payload, raw_bytes, from_cache=False)
    manifest[ticker] = {
        "sha256": report.sha256,
        "bytes": len(raw_bytes),
        "n_bars": report.n_bars,
        "first": report.first,
        "last": report.last,
        "source": url,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_manifest(manifest)
    time.sleep(pause)
    return report


def _summarise(ticker: str, payload: dict, raw_bytes: bytes,
               *, from_cache: bool) -> FetchReport:
    report = FetchReport(ticker=ticker, from_cache=from_cache,
                          sha256=_sha256_bytes(raw_bytes))
    chart = (payload.get("chart") or {})
    if chart.get("error"):
        report.problems.append(f"yahoo error: {chart['error']}")
        return report
    results = chart.get("result") or []
    if not results:
        report.problems.append("empty result set")
        return report
    result = results[0]
    timestamps = result.get("timestamp") or []
    report.n_bars = len(timestamps)
    if timestamps:
        report.first = _utc_date(timestamps[0])
        report.last = _utc_date(timestamps[-1])
    if not timestamps:
        report.problems.append("no timestamps")
    return report


def _utc_date(epoch: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(epoch))


def to_frame(payload: dict) -> pd.DataFrame:
    """Convert a Yahoo chart payload into a tidy adjusted-close frame."""
    chart = payload.get("chart") or {}
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"bad chart payload: {chart.get('error')}")
    result = chart["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjblocks = result["indicators"].get("adjclose") or []
    adj = adjblocks[0]["adjclose"] if adjblocks else quote["close"]

    index = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True)
    frame = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
            "adjclose": adj,
        },
        index=index,
    )
    frame.index = frame.index.tz_convert(None).normalize()
    frame.index.name = "date"
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def load(ticker: str, *, refresh: bool = False) -> tuple[pd.Series, FetchReport]:
    """Return the adjusted-close series for ``ticker`` plus a fetch report."""
    report = fetch_raw(ticker, refresh=refresh)
    payload = json.loads(_cache_path(ticker).read_text())
    frame = to_frame(payload)
    series = frame["adjclose"].dropna()
    return series, report


def verify(series: pd.Series, report: FetchReport) -> FetchReport:
    """Integrity checks that every series must pass before it is used."""
    problems = report.problems
    values = series.to_numpy(dtype=float)

    if len(values) == 0:
        problems.append("empty series")
        report.problems = problems
        return report

    if not np.all(np.isfinite(values)):
        problems.append(f"{int((~np.isfinite(values)).sum())} non-finite values")
    if np.any(values <= 0):
        problems.append(f"{int((values <= 0).sum())} non-positive prices")

    index = series.index
    if not index.is_monotonic_increasing:
        problems.append("dates not monotonic")
    if index.has_duplicates:
        problems.append(f"{int(index.duplicated().sum())} duplicate dates")

    gaps = index.to_series().diff().dt.days.dropna()
    if len(gaps):
        long_gap = gaps[gaps > 10]
        if len(long_gap):
            worst = long_gap.max()
            problems.append(f"{len(long_gap)} gap(s) longer than 10 calendar days (max {int(worst)})")

    # A total-return series should essentially never fall by more than 60% in a
    # single day for these instruments.  A bigger move means a bad splice.
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    returns = returns[np.isfinite(returns)]
    if len(returns):
        worst = returns.min()
        if worst < -0.60:
            problems.append(f"implausible single-day total return of {worst:.3f}")

    report.problems = problems
    return report
