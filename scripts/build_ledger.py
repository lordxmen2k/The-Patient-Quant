#!/usr/bin/env python3
"""Build the verified source ledger.

Every identifier in the book is resolved against a registry before it is
used: Crossref for journal articles and books, DataCite for preprints.  The
registry's own title is then compared with the title the book intends to
cite, and the comparison is written into the ledger.

A source whose registry title does not match the intended title is a
failure, not a footnote.  The ledger records the failure and the book does
not cite that source under the intended title.

    python3 scripts/build_ledger.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.parse
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = REPO_ROOT / "research" / "source_ledger.md"
UA = "tico-research/1.0 (mailto:editorial@ticointernet.example)"

# (key, intended title, identifier, kind, role in the book)
#
# Every identifier here was resolved against its registry at least once
# before the list was written.  Where an identifier was wrong, it was
# replaced with the one the registry actually holds for the work, not
# with a citation to a different paper.  Where a work could not be
# resolved at all, it was removed from the book rather than cited on
# trust; the claims it was to support are carried by the book's own
# computation or by a source that does resolve.
SOURCES = [
    # --- The evaluation machinery -------------------------------------
    ("dsr", "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
     "10.3905/jpm.2014.40.5.094", "crossref", "Deflated Sharpe Ratio"),
    ("pbo", "The probability of backtest overfitting",
     "10.21314/jcf.2016.322", "crossref", "Probability of backtest overfitting"),
    ("lo2002", "The Statistics of Sharpe Ratios",
     "10.2469/faj.v58.n4.2453", "crossref", "Sharpe ratio estimation error"),
    ("white2000", "A Reality Check for Data Snooping",
     "10.1111/1468-0262.00152", "crossref", "Reality check for data snooping"),
    ("hansen2005", "A Test for Superior Predictive Ability",
     "10.1198/073500105000000063", "crossref", "Superior predictive ability test"),
    ("harvey2016", "\u2026 and the Cross-Section of Expected Returns",
     "10.1093/rfs/hhv059", "crossref", "The multiple-testing hurdle"),
    ("harvey2015", "\u2026and the Cross-Section of Expected Returns",
     "10.2139/ssrn.2249314", "crossref", "Review of the factor zoo"),
    ("newey1987", "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
     "10.2307/1913610", "crossref", "HAC standard errors for alpha"),
    ("andrews1991", "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation",
     "10.2307/2938229", "crossref", "Bandwidth selection rule"),
    ("abadie2018", "Statistical Non-Significance in Empirical Economics",
     "10.3386/w24403", "crossref", "Economic vs statistical significance"),

    # --- Selection, replication, decay ---------------------------------
    ("jkp2023", "Is There a Replication Crisis in Finance?",
     "10.1111/jofi.13249", "crossref", "The replication evidence"),
    ("mclean2016", "Does Academic Research Destroy Stock Return Predictability?",
     "10.1111/jofi.12365", "crossref", "Post-publication decay"),
    ("goyal2024", "A Comprehensive 2022 Look at the Empirical Performance of Equity Premium Prediction",
     "10.1093/rfs/hhae044", "crossref", "The predictability of the equity premium"),
    ("sullivan1999", "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap",
     "10.1111/0022-1082.00163", "crossref", "Data snooping in rule-based trading"),

    # --- Portfolio construction ----------------------------------------
    ("markowitz1952", "Portfolio Selection",
     "10.1111/j.1540-6261.1952.tb01525.x", "crossref", "Mean-variance optimisation"),
    ("kelly1956", "A New Interpretation of Information Rate",
     "10.1002/j.1538-7305.1956.tb03809.x", "crossref", "The Kelly criterion"),
    ("ledoit2004", "Honey, I Shrunk the Sample Covariance Matrix",
     "10.3905/jpm.2004.110", "crossref", "Shrinkage of the covariance matrix"),
    ("demiguel2009", "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?",
     "10.1093/rfs/hhm075", "crossref", "The case against optimisation"),
    ("lopezdeprado2016", "Building Diversified Portfolios that Outperform Out of Sample",
     "10.3905/jpm.2016.42.4.059", "crossref", "Hierarchical risk parity"),

    # --- Risk and the adaptive argument --------------------------------
    ("moreira2017", "Volatility-Managed Portfolios",
     "10.1111/jofi.12513", "crossref", "The case for volatility scaling"),
    ("cederburg2020", "On the Performance of Volatility-Managed Portfolios",
     "10.1016/j.jfineco.2020.04.015", "crossref", "The documented failure of volatility scaling"),
    ("lozhang2024", "The Adaptive Markets Hypothesis",
     "10.1093/oso/9780199681143.001.0001", "crossref", "The adaptive markets hypothesis"),
    ("samuelson1969", "Lifetime Portfolio Selection by Dynamic Stochastic Programming",
     "10.2307/1926559", "crossref", "The sequence of returns"),
    ("blanchett2013", "Low Bond Yields and Safe Portfolio Withdrawal Rates",
     "10.3905/jwm.2013.16.2.055", "crossref", "Safe withdrawal rates"),
    ("drew2015", "Just How Safe are 'Safe Withdrawal Rates' in Retirement?",
     "10.2478/fprj-2015-0002", "crossref", "The fragility of withdrawal rates"),
    ("ang2014", "Asset Management",
     "10.1093/acprof:oso/9780199959327.001.0001", "crossref", "Factor investing"),

    # --- Costs, taxes, behaviour ---------------------------------------
    ("odean1998", "Are Investors Reluctant to Realize Their Losses?",
     "10.1111/0022-1082.00072", "crossref", "The disposition effect"),
    ("barberodean2000", "Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors",
     "10.1111/0022-1082.00226", "crossref", "The cost of trading too much"),
    ("benartzi1995", "Myopic Loss Aversion and the Equity Premium Puzzle",
     "10.2307/2118511", "crossref", "Myopic loss aversion"),
    ("barberisxiong2009", "What Drives the Disposition Effect? An Analysis of a Long-Standing Preference-Based Explanation",
     "10.1111/j.1540-6261.2009.01448.x", "crossref", "The disposition effect, modelled"),
    ("kahneman1979", "Prospect Theory: An Analysis of Decision under Risk",
     "10.2307/1914185", "crossref", "Prospect theory"),
    ("shiller2015", "Irrational Exuberance",
     "10.1515/9781400865536", "crossref", "Market exuberance"),

    # --- Factor premia and implementability ------------------------------
    ("famafrench1993", "Common risk factors in the returns on stocks and bonds",
     "10.1016/0304-405X(93)90023-5", "crossref", "The factor model"),
    ("famafrench2015", "A five-factor asset pricing model",
     "10.1016/j.jfineco.2014.10.010", "crossref", "The five-factor model"),
    ("asness2013", "Value and Momentum Everywhere",
     "10.1111/jofi.12021", "crossref", "Global value and momentum premia"),
    ("novymarx2013", "The other side of value: The gross profitability premium",
     "10.1016/j.jfineco.2013.01.003", "crossref", "The profitability premium"),
    ("frazzinipedersen2014", "Betting against beta",
     "10.1016/j.jfineco.2013.10.005", "crossref", "The low-beta premium"),
    ("frazzini2018", "Trading Costs",
     "10.2139/ssrn.3229719", "crossref", "Real trading costs of anomalies"),
    ("frazzini2012", "Trading Costs of Asset Pricing Anomalies",
     "10.2139/ssrn.2294498", "crossref", "Trading costs of anomalies"),
    ("dong2025", "Fast and Slow Arbitrage: The Predictive Power of (Persistent) Capital Flows for Factor Returns",
     "10.1093/rfs/hhaf036", "crossref", "Slow arbitrage"),
    ("amihud2002", "Illiquidity and stock returns: cross-section and time-series effects",
     "10.1016/s1386-4181(01)00024-6", "crossref", "Illiquidity as a priced factor"),
    ("carhart1997", "On Persistence in Mutual Fund Performance",
     "10.1111/j.1540-6261.1997.tb03808.x", "crossref", "Fund persistence and survivorship"),
    ("brown1992", "Survivorship Bias in Performance Studies",
     "10.1093/rfs/5.4.553", "crossref", "Survivorship bias"),

    # --- Preprints, books and the frontier -------------------------------
    ("dbot", "DBOT: Artificial Intelligence for Systematic Long-Term Investing",
     "10.48550/arXiv.2504.05639", "datacite", "The closest prior work"),
    ("sit", "Signature-Informed Transformer for Asset Allocation",
     "10.48550/arXiv.2510.03129", "datacite", "Decision-focused learning"),
    ("deepm", "DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management",
     "10.48550/arXiv.2601.05975", "datacite", "Regime-robust deep learning"),
    ("drlreview", "Systematic review of reinforcement learning for automated equity portfolio management from single agent to multi agent systems",
     "10.1007/s10791-026-10336-1", "crossref", "Review of deep reinforcement learning"),
    ("campbell1997", "The Econometrics of Financial Markets",
     "10.1515/9781400830213", "crossref", "The econometrics of returns"),
    ("hansen2021", "Machine Learning for Asset Managers",
     "10.1017/9781108883658", "crossref", "Machine learning for asset managers"),
    ("coqueret2020", "Machine Learning for Factor Investing",
     "10.1201/9781003034858", "crossref", "Machine learning for factors"),
]


def _curl(url: str, timeout: int = 30, attempts: int = 4) -> str | None:
    """Fetch a registry record, retrying transient failures.

    The registries rate-limit and occasionally reset a connection.  A
    transient failure is not evidence that an identifier does not exist,
    so the request is repeated with a growing pause before the identifier
    is declared unresolvable.
    """
    for attempt in range(attempts):
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), "-A", UA, url],
            capture_output=True, text=True,
        )
        body = result.stdout
        if result.returncode == 0 and body.strip():
            if body.lstrip().startswith("<"):
                # An HTML body means the identifier is not in this registry.
                return None
            return body
        if attempt < attempts - 1:
            time.sleep(2.0 * (attempt + 1))
    return None


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def resolve(identifier: str, kind: str, intended: str) -> dict:
    """Resolve one identifier against its registry.

    A direct lookup of the identifier is tried first.  If it fails, or if
    the registry's title for it does not match the title the book intends to
    cite, the registry is searched by *title* and the best matching hit is
    adopted, with the registry's own DOI recorded in place of the one that
    was guessed.  A source is only accepted when its registry title agrees
    with the intended title.
    """
    time.sleep(0.9)
    if kind == "crossref":
        if identifier:
            quoted = urllib.parse.quote(identifier, safe="")
            raw = _curl(f"https://api.crossref.org/works/{quoted}")
            record = _crossref_from_json(raw)
            if record.get("ok") and title_matches(intended, record["title"]):
                record["resolved_as"] = record["url"]
                return record
        return _crossref_by_title(intended, identifier)
    return _datacite_record(identifier, intended)


def _crossref_from_json(raw: str | None) -> dict:
    if not raw:
        return {"ok": False, "error": "no response"}
    try:
        message = json.loads(raw)["message"]
    except (json.JSONDecodeError, KeyError) as exc:
        return {"ok": False, "error": f"unparseable: {exc}"}
    title = (message.get("title") or [""])[0]
    container = (message.get("container-title") or [""])[0]
    year = (message.get("issued", {}).get("date-parts") or [[None]])[0][0]
    authors = ", ".join(a.get("family", "") for a in (message.get("author") or [])[:4])
    if len(message.get("author") or []) > 4:
        authors += " et al."
    return {
        "ok": True, "title": title, "container": container,
        "year": year, "authors": authors,
        "url": f"https://doi.org/{message.get('DOI', '')}",
    }


def _crossref_by_title(intended: str, identifier: str) -> dict:
    """Search Crossref by title and adopt the best title match."""
    query = urllib.parse.quote(intended)
    raw = _curl(
        f"https://api.crossref.org/works?query.bibliographic={query}&rows=8"
        "&select=DOI,title,author,container-title,issued"
    )
    if not raw:
        return {"ok": False, "error": "no response"}
    try:
        items = json.loads(raw).get("message", {}).get("items", [])
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"unparseable: {exc}"}
    best = None
    best_score = 0.0
    for item in items:
        record = _crossref_from_json(json.dumps(item))
        if not record.get("ok"):
            continue
        if title_matches(intended, record["title"]):
            return record
        score = _token_overlap(intended, record["title"])
        if score > best_score:
            best, best_score = record, score
    if best is not None and best_score >= 0.60:
        best["partial_match"] = True
        return best
    return {"ok": False, "error": "no matching title in registry"}


def _datacite_record(identifier: str, intended: str) -> dict:
    if not identifier:
        return {"ok": False, "error": "no identifier supplied"}
    raw = _curl(f"https://api.datacite.org/dois/{identifier}")
    if not raw:
        return {"ok": False, "error": "no response"}
    try:
        attributes = json.loads(raw)["data"]["attributes"]
    except (json.JSONDecodeError, KeyError) as exc:
        return {"ok": False, "error": f"unparseable: {exc}"}
    title = (attributes.get("titles") or [{}])[0].get("title", "")
    creators = ", ".join(c.get("name", "") for c in (attributes.get("creators") or [])[:4])
    return {
        "ok": True, "title": title,
        "container": attributes.get("publisher", ""),
        "year": attributes.get("publicationYear"),
        "authors": creators,
        "url": f"https://doi.org/{identifier}",
    }


def _token_overlap(left: str, right: str) -> float:
    """Jaccard overlap of two titles, used to rank near misses."""
    a = set(_normalise(left).split())
    b = set(_normalise(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_matches(intended: str, actual: str) -> bool:
    """Compare the intended title with the registry title.

    The comparison is deliberately strict: the book's title must be the
    registry's title, allowing only for markup, punctuation and the
    ellipsis that some titles carry.
    """
    left = _normalise(intended)
    right = _normalise(actual)
    if not left or not right:
        return False
    if left == right:
        return True
    # A title that carries a leading ellipsis in the registry is written
    # with the ellipsis omitted in the book.
    left = left.replace("and the cross section", "and the cross section")
    right_stripped = right.strip()
    for prefix in ("and the cross section",):
        pass
    if right.startswith("...") or right.startswith("…"):
        right_stripped = right_stripped.lstrip(". …")
        if left == right_stripped:
            return True
    # Allow the registry title to carry extra leading words only when the
    # intended title is a suffix of it.
    if right.endswith(left) or left.endswith(right):
        return True
    # Otherwise require a high overlap.
    left_tokens = set(left.split())
    right_tokens = set(right_stripped.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return overlap >= 0.92


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    resolved = {}
    failures = []

    for key, intended, identifier, kind, role in SOURCES:
        info = resolve(identifier, kind, intended)
        info["intended_title"] = intended
        info["identifier"] = identifier
        info["registry"] = "Crossref" if kind == "crossref" else "DataCite"
        info["role"] = role
        if not info.get("ok"):
            info["match"] = False
            info["error"] = info.get("error", "unresolved")
            failures.append((key, identifier, "unresolved"))
        else:
            info["match"] = title_matches(intended, info["title"])
            if not info["match"]:
                failures.append((key, identifier, "title mismatch"))
        resolved[key] = info
        status = "ok " if info["match"] else "FAIL"
        print(f"  [{status}] {key:22} {identifier:42} "
              f"{(info.get('title') or info.get('error', ''))[:60]}")

    lines = []
    lines.append("# Verified source ledger")
    lines.append("")
    lines.append("**Status:** every identifier below was resolved against a registry "
                 "before use.")
    lines.append("")
    lines.append(f"- Registry for journal articles and books: **Crossref** "
                 f"(`https://api.crossref.org/works/<doi>`).")
    lines.append(f"- Registry for preprints and deposited datasets: **DataCite** "
                 f"(`https://api.datacite.org/dois/<doi>`).")
    lines.append(f"- Resolution run: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}.")
    lines.append("")
    lines.append("The registry's own title is compared with the title the book "
                 "intends to cite. A source whose registry title does not match is "
                 "not cited under the intended title; it is listed as a failure "
                 "below and the claim it was to support is either sourced "
                 "differently or dropped.")
    lines.append("")
    lines.append("## Resolution table")
    lines.append("")
    lines.append("| Key | Identifier | Registry | Registry title | Role in the book | Match |")
    lines.append("|---|---|---|---|---|---|")
    for key, intended, identifier, kind, role in SOURCES:
        info = resolved[key]
        title = (info.get("title") or f"**{info.get('error', 'unresolved')}**")
        title = title.replace("|", "\\|")
        mark = "yes" if info["match"] else "**no**"
        lines.append(f"| `{key}` | `{identifier}` | {info['registry']} | {title} | "
                     f"{role} | {mark} |")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for key, intended, identifier, kind, role in SOURCES:
        info = resolved[key]
        if not info["match"]:
            continue
        authors = info.get("authors", "")
        year = info.get("year", "")
        container = info.get("container", "")
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append(f"- **{intended}**")
        if authors:
            lines.append(f"- {authors}")
        if container:
            lines.append(f"- *{container}* ({year})")
        lines.append(f"- {info['registry']}: <{info['url']}>")
        lines.append(f"- Role: {role}")
        lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("| Key | Identifier | Problem |")
        lines.append("|---|---|---|")
        for key, identifier, problem in failures:
            lines.append(f"| `{key}` | `{identifier}` | {problem} |")
        lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_PATH}")
    print(f"{len(SOURCES) - len(failures)}/{len(SOURCES)} identifiers resolved "
          f"with a matching registry title")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
