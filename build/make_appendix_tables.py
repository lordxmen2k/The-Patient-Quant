#!/usr/bin/env python3
"""Generate the book's appendix tables from the canonical data files.

Every table in the appendices is written here from the files the system
produced, so that no number in an appendix is transcribed.  Three files
are written into ``source/``:

    generated_appendix_data.tex      Appendix A, the data and its provenance
    generated_appendix_universe.tex  Appendix C, the universe and its terms
    generated_appendix_ledger.tex    Appendix E, the source ledger

    python3 build/make_appendix_tables.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from patientquant.universe import BY_TICKER, UNIVERSE  # noqa: E402

SOURCE = REPO / "source"
CACHE = REPO / "data" / "cache"


def latex_escape(text: str) -> str:
    text = str(text)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(old, new)
    return text


def main() -> int:
    numbers = json.loads((REPO / "build" / "numbers.json").read_text())
    manifest = json.loads((CACHE / "manifest.json").read_text())
    integrity = json.loads((CACHE / "integrity.json").read_text())
    terms = pd.read_csv(REPO / "build" / "series" / "fund_terms.csv",
                        index_col=0)
    meta = numbers["meta"]

    # ------------------------------------------------------------------
    # Appendix A: the data and its provenance
    # ------------------------------------------------------------------
    rows = []
    rows.append("% ---------------------------------------------------------------------")
    rows.append("%  Appendix A — generated from data/cache/manifest.json")
    rows.append("% ---------------------------------------------------------------------")
    rows.append(r"\begin{tabularx}{\textwidth}{@{}P{0.62in}P{0.92in}P{0.92in}Y@{}}")
    rows.append(r"\toprule")
    rows.append(r"Ticker & First bar & Last bar & Bars \\")
    rows.append(r"\midrule")
    for ticker in sorted(manifest):
        entry = manifest[ticker]
        rows.append(
            f"{latex_escape(ticker)} & {latex_escape(entry['first'])} & "
            f"{latex_escape(entry['last'])} & {entry['n_bars']} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabularx}")
    (SOURCE / "generated_appendix_data.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Appendix C: the universe and its terms
    # ------------------------------------------------------------------
    from patientquant import panel as pq_panel
    panel = pq_panel.monthly_panel()
    returns = panel["returns"]
    rows = []
    rows.append("% ---------------------------------------------------------------------")
    rows.append("%  Appendix C — generated from build/series/fund_terms.csv")
    rows.append("% ---------------------------------------------------------------------")
    rows.append(r"\begin{tabularx}{\textwidth}{@{}P{0.46in}P{1.06in}P{0.62in}P{0.48in}P{0.50in}P{0.54in}Y@{}}")
    rows.append(r"\toprule")
    rows.append(r"Ticker & Name & Sleeve & Expense & Listed & Live from & First return \\")
    rows.append(r"\midrule")
    ordered = sorted(UNIVERSE, key=lambda i: (i.group, i.ticker))
    for instrument in ordered:
        ticker = instrument.ticker
        if ticker not in terms.index:
            continue
        row = terms.loc[ticker]
        expense = row.get("expense_ratio")
        expense_text = (f"{expense * 100:.2f}\\%"
                        if expense is not None and pd.notna(expense) else "---")
        inception = row.get("inception_date")
        inception_text = (pd.Timestamp(inception).strftime("%Y-%m-%d")
                          if inception is not None and pd.notna(inception)
                          else "---")
        series = returns[ticker].dropna()
        live_from = (series.index[0].strftime("%Y-%m-%d") if len(series)
                     else "---")
        rows.append(
            f"{latex_escape(ticker)} & {latex_escape(instrument.name)} & "
            f"{latex_escape(instrument.group)} & {expense_text} & "
            f"{inception_text} & {live_from} & "
            f"{len(series)} months \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabularx}")
    (SOURCE / "generated_appendix_universe.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Appendix E: the source ledger
    # ------------------------------------------------------------------
    ledger = (REPO / "research" / "source_ledger.md").read_text(encoding="utf-8")
    table_rows = []
    # A ledger row may wrap over several physical lines when the registry
    # title carries markup, so the rows are joined before they are split
    # into cells.
    blocks: list[str] = []
    current: list[str] = []
    for line in ledger.splitlines():
        if line.startswith("| `"):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current:
            # A row is finished when its last physical line closes the
            # table cell; a registry title carrying markup can wrap over
            # several lines.
            if line.strip().endswith("|"):
                current.append(line)
                blocks.append("\n".join(current))
                current = []
            else:
                current.append(line)
    if current:
        blocks.append("\n".join(current))
    for block in blocks:
        cells = [c.strip() for c in block.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        key, identifier, registry, title, role, match = cells[:6]
        identifier_clean = identifier.replace("`", "")
        title_clean = re.sub(r"<[^>]+>", "", title).strip()
        title_clean = " ".join(title_clean.split())
        role_clean = " ".join(role.replace("**", "").split())
        match_clean = match.replace("**", "").strip()
        rows_line = (
            f"\\code{{{latex_escape(identifier_clean)}}} & "
            f"{latex_escape(title_clean)} & {latex_escape(role_clean)} & "
            f"{match_clean} \\\\")
        table_rows.append(rows_line)

    entries = []
    for match in re.finditer(r"^### `([^`]+)`\n(.*?)(?=^### |\Z)",
                             ledger, re.M | re.S):
        key = match.group(1)
        body = match.group(2)
        title = re.search(r"^- \*\*(.+?)\*\*$", body, re.M)
        authors = re.search(r"^- (?!.*\*)(.+)$", body, re.M)
        venue = re.search(r"^- \*(.+?)\* \((\d{4})\)$", body, re.M)
        url = re.search(r"- (Crossref|DataCite): <(.+?)>", body)
        role = re.search(r"^- Role: (.+)$", body, re.M)
        entries.append({
            "key": key,
            "title": title.group(1) if title else "",
            "authors": authors.group(1).strip() if authors else "",
            "venue": venue.group(1) if venue else "",
            "year": venue.group(2) if venue else "",
            "url": url.group(2) if url else "",
            "role": role.group(1).strip() if role else "",
        })

    out = []
    out.append("% ---------------------------------------------------------------------")
    out.append("%  Appendix E — generated from research/source_ledger.md")
    out.append("% ---------------------------------------------------------------------")
    out.append(r"\begin{tabular}{@{}P{1.32in}P{2.00in}P{0.92in}P{0.36in}@{}}")
    out.append(r"\toprule")
    out.append(r"Identifier & Registry title & Role in the book & Match \\")
    out.append(r"\midrule")
    out.extend(table_rows)
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append("")
    for entry in entries:
        out.append(r"\noindent\textbf{" + latex_escape(entry["key"]) + r"}\quad "
                   + latex_escape(entry["authors"]) + r" \emph{"
                   + latex_escape(entry["venue"]) + r"} (" + entry["year"] + r"). "
                   + latex_escape(entry["title"]) + r". \url{"
                   + entry["url"].replace("https://doi.org/", "") + r"}. "
                   + latex_escape(entry["role"]) + r"\par")
    (SOURCE / "generated_appendix_ledger.tex").write_text(
        "\n".join(out) + "\n", encoding="utf-8")

    print(f"wrote three appendix tables: {len(table_rows)} ledger rows, "
          f"{len(entries)} ledger entries, "
          f"{len(manifest)} cached series, "
          f"{int(integrity['rows'][0]['n_bars'] > 0)} checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
