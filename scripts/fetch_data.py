#!/usr/bin/env python3
"""Download and verify every price series the book uses.

    python3 scripts/fetch_data.py            # use the cache, fetch what is missing
    python3 scripts/fetch_data.py --refresh  # re-download everything
    python3 scripts/fetch_data.py --verify   # re-run integrity checks only

The script prints a table and writes ``data/cache/integrity.json``.  A ticker
that fails a check is reported loudly: the book will not silently use a broken
series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from patientquant import data as pq_data  # noqa: E402
from patientquant.universe import TICKERS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download every series")
    parser.add_argument("--verify", action="store_true",
                        help="re-download and verify every series")
    parser.add_argument("--offline", action="store_true",
                        help="verify the cached series without touching the network")
    parser.add_argument("--tickers", nargs="*", default=None, help="restrict to these tickers")
    args = parser.parse_args()

    tickers = args.tickers or TICKERS
    rows = []
    failures = []

    for ticker in tickers:
        try:
            if args.offline:
                report = pq_data.fetch_raw(ticker)
                series, _ = pq_data.load(ticker)
            elif args.verify:
                report = pq_data.fetch_raw(ticker, refresh=True)
                series, _ = pq_data.load(ticker)
            else:
                report = pq_data.fetch_raw(ticker, refresh=args.refresh)
                series, _ = pq_data.load(ticker)
            report = pq_data.verify(series, report)
        except Exception as exc:  # noqa: BLE001
            failures.append(ticker)
            rows.append({
                "ticker": ticker, "n_bars": 0, "first": "", "last": "",
                "status": "FETCH FAILED", "detail": str(exc)[:80],
            })
            print(f"  {ticker:<6} FETCH FAILED  {exc}", flush=True)
            continue

        status = "ok" if report.ok else "PROBLEM"
        if not report.ok:
            failures.append(ticker)
        rows.append({
            "ticker": ticker,
            "n_bars": report.n_bars,
            "first": report.first or "",
            "last": report.last or "",
            "status": status,
            "detail": "; ".join(report.problems),
        })
        print(
            f"  {ticker:<6} {report.n_bars:>6} bars  {report.first} -> {report.last}"
            f"  {status}{'  ' + '; '.join(report.problems) if report.problems else ''}",
            flush=True,
        )

    out = REPO_ROOT / "data" / "cache" / "integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n")

    print(f"\n{len(tickers) - len(failures)}/{len(tickers)} series passed every check")
    if failures:
        print("failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
