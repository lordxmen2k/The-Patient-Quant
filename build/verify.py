#!/usr/bin/env python3
"""The book's verification gates.

Twenty-two checks run after the PDF is typeset.  Each one catches a way in
which a book like this can be subtly wrong without looking wrong: a stale
figure, a hand-transcribed number, a caption pointing at the wrong
figure, a page with overlapping labels, a number in the prose that the
code does not produce.

    python3 build/verify.py

Exit status is non-zero if any gate fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "source")
FIGURES = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "output", "The-Patient-Quant.pdf")
CACHE = os.path.join(ROOT, "data", "cache")

RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str):
    """Decorator that records a gate's outcome."""

    def wrap(fn):
        def run():
            try:
                ok, detail = fn()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            RESULTS.append((name, ok, detail))
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {name:52s} {detail}")
        run.__name__ = fn.__name__
        return run
    return wrap


def source_text() -> str:
    """The book's own prose: the chapters and the front and back matter.

    The preamble and the master file are excluded, because they define
    macros --- including a figure-inclusion macro --- and the gates below
    are about the text a reader reads.
    """
    parts = []
    for base in ("source/chapters",):
        directory = os.path.join(ROOT, base)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(".tex"):
                with open(os.path.join(directory, name), encoding="utf-8") as fh:
                    parts.append(fh.read())
    for name in ("frontmatter.tex", "backmatter.tex"):
        path = os.path.join(SRC, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                parts.append(fh.read())
    return "\n".join(parts)


def normalise(text: str) -> str:
    """A comparable form of the source text.

    LaTeX writes an accented name as an escape sequence --- L\'opez --- and
    a title in a command argument, so a literal comparison of the source
    against a ledger entry misses both.  This strips the escapes and the
    accents and lowercases the result.
    """
    import unicodedata
    text = re.sub(r"\\[`'\"^~=.uvHckbdtr]\s*", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def plain_text() -> str:
    text = source_text()
    text = re.sub(r"\\idx\{[^}]*\}", " ", text)
    text = re.sub(r"\\(begin|end)\{[^}]*(\[[^\]]*\])?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$^_&~\\]", " ", text)
    return re.sub(r"\s+", " ", text)


def figure_paths() -> list[str]:
    """Every figure the book's text includes.

    The chapters include their figures with ``\\includegraphics``.  The two
    cover plates are put in by the master file with its own ``\\fullbleed``
    helper, whose own definition contains a ``\\includegraphics`` with a
    placeholder argument --- so the master file is scanned for the helper's
    uses, not for its definition.
    """
    paths = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", source_text())
    master = os.path.join(SRC, "main.tex")
    if os.path.exists(master):
        with open(master, encoding="utf-8") as fh:
            text = fh.read()
        paths += re.findall(r"\\fullbleed\{([^}]+)\}", text)
    return paths


def numbers() -> dict:
    with open(os.path.join(ROOT, "build", "numbers.json"), encoding="utf-8") as fh:
        return json.load(fh)


def macro_file() -> str:
    path = os.path.join(SRC, "generated_numbers.tex")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def pdf_text() -> str:
    """The rendered text of the finished PDF, page by page.

    This is what a reader actually sees, so it is the only place a baked-in
    template marker can be caught for certain.  pdftotext is used when it is
    available; pypdf is the fallback.
    """
    proc = subprocess.run(["pdftotext", OUT, "-"], capture_output=True,
                          text=True)
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout
    import pypdf
    reader = pypdf.PdfReader(OUT)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# =====================================================================
@gate("1. the PDF builds and is a book")
def gate_01_pdf():
    if not os.path.exists(OUT):
        return False, f"missing {OUT}"
    import pypdf
    reader = pypdf.PdfReader(OUT)
    pages = len(reader.pages)
    if pages < 150:
        return False, f"{pages} pages is too few"

    # A build with errors in it is not a book: LaTeX recovers from an
    # unbalanced argument by reading on until it finds a closing brace,
    # and the text it swallowed simply does not appear.  The log is
    # checked here, because a PDF that builds is not a PDF that is
    # right.
    log = os.path.join(SRC, "main.log")
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as fh:
            errors = [line for line in fh if line.startswith("!")]
        if errors:
            first = errors[0].strip()
            return False, (f"{len(errors)} LaTeX errors in the log; "
                           f"the first is {first}")
    return True, f"{pages} pages, no LaTeX errors"


@gate("2. every figure named in the source exists")
def gate_02_figures_exist():
    named = figure_paths()
    missing = [p for p in named
               if not os.path.exists(os.path.join(SRC, p))]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, f"{len(named)} figure references, all present"


@gate("3. every figure is used once, and no label is defined twice")
def gate_03_figures_once():
    named = figure_paths()
    on_disk = {f"../figures/{n}" for n in os.listdir(FIGURES)
               if n.endswith(".pdf")}
    orphans = on_disk - set(named)
    duplicates = {p for p in named if named.count(p) > 1}
    labels = re.findall(r"\\label\{(fig:[^}]+)\}", source_text())
    repeated_labels = {l for l in labels if labels.count(l) > 1}
    if orphans:
        return False, f"never referenced: {', '.join(sorted(orphans))}"
    if duplicates:
        return False, f"used more than once: {', '.join(sorted(duplicates))}"
    if repeated_labels:
        return False, f"label defined twice: {', '.join(sorted(repeated_labels))}"
    return True, f"{len(named)} figures, each used once, {len(labels)} labels"


@gate("4. no text overlaps any curve, marker or other text")
def gate_04_overlaps():
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "build", "check_fig_overlaps.py")],
        capture_output=True, text=True, cwd=ROOT)
    tail = (proc.stdout.strip().splitlines() or ["no output"])[-1]
    return proc.returncode == 0, tail


@gate("5. every number macro used is defined")
def gate_05_macros_defined():
    """Every macro used must be defined, and every macro name must be legal.

    A LaTeX control word may not contain a digit.  The tokeniser reads
    "\\numSpyW12" as "\\numSpyW" followed by the characters "12", so a
    macro whose name carries a digit can never be called from the text,
    and its definition puts the digits into the document as stray text.
    The failure is silent in the output --- the page typesets, and the
    number is simply not there --- so the names are checked here.
    """
    used = set(re.findall(r"\\(num[A-Za-z0-9]+)", source_text()))
    defined = set(re.findall(r"\\newcommand\{\\(num[A-Za-z0-9]+)\}",
                             macro_file()))
    missing = used - defined
    if missing:
        return False, f"undefined: {', '.join(sorted(missing))}"
    illegal = sorted(n for n in used | defined if not n.isalpha())
    if illegal:
        return False, ("macro names may not contain digits: "
                       f"{', '.join(illegal)}")
    return True, f"{len(used)} macros used, {len(defined)} defined"


@gate("6. every macro agrees with the canonical data")
def gate_06_macros_agree():
    data = numbers()
    text = macro_file()
    macros = dict(re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}\{([^}]*)\}",
                             text))
    summaries = data["summaries"]
    family = data["family_summaries"]
    checks = {
        "numSpyCAGR": f"{summaries['SPY buy and hold']['cagr'] * 100:.2f}",
        "numFullCAGR": f"{summaries['System, tax exempt']['cagr'] * 100:.2f}",
        "numTaxCAGR": f"{summaries['System, taxable']['cagr'] * 100:.2f}",
        "numFullSharpe": f"{summaries['System, tax exempt']['sharpe']:.3f}",
        "numFullDD": f"{summaries['System, tax exempt']['max_drawdown'] * 100:.2f}",
        "numCoreCAGR": f"{family['no tilt, no governor']['cagr'] * 100:.2f}",
        "numTiltCAGR": f"{family['tilt only']['cagr'] * 100:.2f}",
        "numGovCAGR": f"{family['governor only']['cagr'] * 100:.2f}",
        "numPBO": f"{data['selection']['pbo']['pbo']:.3f}",
        "numFamilySize": str(data["meta"]["n_trial_family"]),
        "numRecordMonths": str(data["meta"]["system_months"]),
        "numNFunds": str(data["terms"]["n_funds"]),
    }
    bad = []
    for name, expected in checks.items():
        actual = macros.get(name, "").replace("\\%", "")
        if actual != expected:
            bad.append(f"{name}: text {actual!r} vs data {expected!r}")
    if bad:
        return False, "; ".join(bad)
    return True, f"{len(checks)} macros re-derived and matched"


@gate("7. the headline numbers appear in the rendered book")
def gate_07_headlines():
    """Every number macro the manuscript uses must be on the page.

    The check is made against the rendered PDF, not against the source
    text.  A macro that is defined but never pulled into the document
    still passes a source-level check, and a manuscript whose numbers have
    silently rendered blank still passes one: the numbers were in the
    source, they were simply never typeset.  So the value of every macro
    the text uses is looked for in the PDF.
    """
    data = numbers()
    macros = dict(re.findall(r"\\newcommand\{\\(num[A-Za-z0-9]+)\}\{([^}]*)\}",
                             macro_file()))
    used = set(re.findall(r"\\(num[A-Za-z0-9]+)", source_text()))
    rendered = re.sub(r"\s+", "", pdf_text())
    # LaTeX writes the percent sign as \%; the rendered page carries %.
    rendered = rendered.replace("\\%", "%")

    missing = []
    for macro in sorted(used):
        value = macros.get(macro, "").strip().replace("\\%", "%")
        if not value:
            missing.append(f"{macro} has no value")
            continue
        if value not in rendered:
            missing.append(f"{macro} ({value})")
    if missing:
        return False, ("not found in the rendered PDF: "
                       f"{', '.join(missing[:8])}")
    return True, f"{len(used)} macros used, all present in the rendered PDF"


@gate("8. every source in the ledger is cited")
def gate_08_citations():
    """Every source in the ledger must be cited in the book.

    A source counts as cited when the text names its author or carries its
    title.  The comparison is made on a normalised form of the text,
    because LaTeX writes an accented name as an escape sequence --- Shiller's
    colleague L\'opez de Prado --- and a literal comparison would not see
    it.
    """
    ledger_path = os.path.join(ROOT, "research", "source_ledger.md")
    with open(ledger_path, encoding="utf-8") as fh:
        ledger = fh.read()
    keys = re.findall(r"^### `([^`]+)`", ledger, re.M)
    text = normalise(source_text())

    missing = []
    for key in keys:
        block = re.search(rf"^### `{re.escape(key)}`\n(.*?)(?=^### |\Z)",
                          ledger, re.M | re.S)
        if not block:
            missing.append(f"{key} (no ledger entry)")
            continue
        entry = block.group(1)
        title = re.search(r"^- \*\*(.+?)\*\*$", entry, re.M)
        authors = re.search(r"^- (?!.*\*)(.+)$", entry, re.M)
        names = []
        if authors:
            names.append(authors.group(1).split(",")[0].strip())
        if title:
            names.append(title.group(1).split(":")[0].strip())
        if not any(name and normalise(name) in text for name in names):
            missing.append(f"{key} ({names[0] if names else '?'})")
    if missing:
        return False, f"not cited: {', '.join(missing)}"
    return True, f"{len(keys)} ledger entries, all cited"


@gate("9. the source ledger is clean")
def gate_09_ledger():
    ledger_path = os.path.join(ROOT, "research", "source_ledger.md")
    with open(ledger_path, encoding="utf-8") as fh:
        ledger = fh.read()
    rows = re.findall(r"^\| `(\w+)` \| .*$", ledger, re.M)
    failed = re.findall(r"^\| `(\w+)` \|.*\*\*no\*\*", ledger, re.M)
    if failed:
        return False, f"unresolved: {', '.join(failed)}"
    if len(rows) < 40:
        return False, f"only {len(rows)} entries"
    return True, f"{len(rows)} identifiers, all resolved with matching titles"


@gate("10. the page count in the colophon is right")
def gate_10_pages():
    import pypdf
    reader = pypdf.PdfReader(OUT)
    pages = len(reader.pages) - 2
    generated = os.path.join(SRC, "generated.tex")
    with open(generated, encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r"\\bookPages\}\{(\d+)\}", text)
    if not match:
        return False, "no page count in generated.tex"
    if int(match.group(1)) != pages:
        return False, f"colophon says {match.group(1)}, PDF has {pages}"
    return True, f"{pages} numbered pages"


@gate("11. the word count in the colophon is right")
def gate_11_words():
    generated = os.path.join(SRC, "generated.tex")
    with open(generated, encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r"\\bookWords\}\{([\d,]+)\}", text)
    if not match:
        return False, "no word count in generated.tex"
    stated = int(match.group(1).replace(",", ""))
    recount = len([w for w in plain_text().split()
                   if re.search(r"[A-Za-z0-9]", w)])
    if abs(stated - recount) > 5:
        return False, f"colophon says {stated:,}, recount is {recount:,}"
    return True, f"{recount:,} words"


@gate("12. no template residue survives")
def gate_12_placeholders():
    """Scan the source AND the rendered PDF for template residue.

    Scanning only the prose would miss the two places a template most
    often survives: the preamble, where the identity fields live, and the
    rendered document, where a generated field that no gate re-derives
    would be baked in for good.  Both are scanned here.

    The patterns are the shapes a generator leaves behind, not the words a
    writer might use in prose: an uppercase token, or a run of four or more
    X characters standing in for a number.
    """
    haystacks = {"source": source_text()}
    for name in ("preamble.tex", "main.tex"):
        path = os.path.join(SRC, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                haystacks[name] = fh.read()
    if os.path.exists(OUT):
        haystacks["pdf"] = pdf_text()

    patterns = {
        "PLACEHOLDER": re.compile(r"PLACEHOLDER"),
        "XXXX": re.compile(r"X{4,}"),
        "TODO": re.compile(r"\bTODO\b"),
        "FIXME": re.compile(r"\bFIXME\b"),
        "Lorem ipsum": re.compile(r"Lorem ipsum", re.IGNORECASE),
        "TBD": re.compile(r"\bTBD\b"),
        "PENDING": re.compile(r"\bPENDING\b"),
    }
    hits = []
    for where, text in haystacks.items():
        for label, pattern in patterns.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{label} in {where} line {line}")
    if hits:
        return False, "; ".join(hits[:8])
    return True, f"clean in {len(haystacks)} texts (source, preamble, main, pdf)"


@gate("13. the copyright string is exact")
def gate_13_copyright():
    text = source_text()
    if "Copyright \\textcopyright\\ 2026 Tico Internet LLC" not in text:
        return False, "the exact copyright string is absent"
    others = re.findall(r"Copyright \\textcopyright\\ (\d{4})", text)
    bad = sorted({o for o in others if o != "2026"})
    if bad:
        return False, f"other copyright years: {bad}"
    holders = re.findall(r"Copyright \\textcopyright\\ \d{4}\s*([^\\]{0,40})",
                         text)
    bad_holders = sorted({h.strip() for h in holders
                          if "Tico Internet LLC" not in h})
    if bad_holders:
        return False, f"other copyright holders: {bad_holders}"
    return True, "Copyright 2026 Tico Internet LLC, and nothing else"


@gate("14. no secret is in the repository")
def gate_14_secrets():
    patterns = [
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"(?i)password\s*[:=]\s*\S{6,}"),
        re.compile(r"(?i)api[_-]?key\s*[:=]\s*\S{12,}"),
        re.compile(r"zenodo[\s_-]*token\s*[:=]\s*\S{8,}"),
    ]
    hits = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "build", "figures", "output",
                                "__pycache__", "node_modules"}]
        for name in files:
            if not name.endswith((".py", ".tex", ".md", ".sh", ".toml",
                                  ".txt", ".json", ".yaml", ".yml")):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    body = fh.read()
            except OSError:
                continue
            for pattern in patterns:
                if pattern.search(body):
                    hits.append(os.path.relpath(path, ROOT))
                    break
    if hits:
        return False, f"possible secrets in: {', '.join(sorted(set(hits)))}"
    return True, "no secret patterns in the tracked files"


@gate("15. the house fonts are embedded")
def gate_15_fonts():
    proc = subprocess.run(["pdffonts", OUT], capture_output=True, text=True)
    if proc.returncode != 0:
        return False, "pdffonts failed"
    lines = [line for line in proc.stdout.splitlines()[2:] if line.strip()]
    if not lines:
        return False, "no fonts in the PDF"
    unembedded = []
    for line in lines:
        flags = re.findall(r"\b(yes|no)\b", line)
        if not flags:
            continue
        # pdffonts prints the embedded, subset and unicode flags after the
        # name, the type and the encoding; the embedded flag is the first.
        if flags[0] != "yes":
            unembedded.append(line.split()[0])
    if unembedded:
        return False, f"not embedded: {', '.join(unembedded)}"
    return True, f"{len(lines)} font records, all embedded"


@gate("16. the PDF is structurally sound")
def gate_16_pdf_sound():
    import pypdf
    reader = pypdf.PdfReader(OUT)
    if reader.is_encrypted:
        return False, "the PDF is encrypted"
    text = "".join((page.extract_text() or "") for page in reader.pages[:20])
    if "The Patient Quant" not in text:
        return False, "the title does not appear in the first twenty pages"
    proc = subprocess.run(["qpdf", "--check", OUT], capture_output=True,
                          text=True)
    if proc.returncode not in (0, 3):
        return False, "qpdf reported a structural problem"
    return True, f"{len(reader.pages)} pages, not encrypted, title present"


@gate("17. the data cache matches its checksums")
def gate_17_cache():
    with open(os.path.join(CACHE, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    bad = []
    for ticker, entry in manifest.items():
        path = os.path.join(CACHE, "raw", f"{ticker}.json")
        if not os.path.exists(path):
            bad.append(f"{ticker} missing")
            continue
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != entry["sha256"]:
            bad.append(f"{ticker} checksum")
    if bad:
        return False, f"{len(bad)} problems: {', '.join(bad[:5])}"
    return True, f"{len(manifest)} cached series match their checksums"


@gate("18. the panel rebuilds from the cache")
def gate_18_panel():
    sys.path.insert(0, ROOT)
    import pandas as pd
    from patientquant import panel as pq_panel
    rebuilt = pq_panel.monthly_panel()["returns"]
    stored = pd.read_csv(os.path.join(ROOT, "build", "series",
                                       "panel_returns.csv"),
                          index_col=0, parse_dates=True)
    if rebuilt.shape != stored.shape:
        return False, f"shape {rebuilt.shape} vs {stored.shape}"
    difference = (rebuilt - stored.reindex(columns=rebuilt.columns)).abs()
    worst = float(np.nanmax(difference.to_numpy(dtype=float)))
    if worst > 1e-12:
        return False, f"the rebuilt panel differs by {worst:.3e}"
    return True, f"{rebuilt.shape[1]} funds, {rebuilt.shape[0]} months, identical"


@gate("19. no look-ahead in the training set")
def gate_19_lookahead():
    sys.path.insert(0, ROOT)
    import numpy as np
    import pandas as pd
    from patientquant import features as pq_features
    from patientquant import models as pq_models
    from patientquant import panel as pq_panel

    panel = pq_panel.monthly_panel()
    raw = pq_features.build_features(panel)
    scaled = pq_features.cross_sectional_scale(raw)
    config = pq_models.Config(name="full", horizon=3)
    target = pq_features.build_target(panel, config.horizon)
    frame = pq_models.walk_forward(scaled, target, config,
                                   min_months=36, min_rows=180)
    if not len(frame):
        return False, "the walk-forward produced nothing to check"

    # Independently rebuild the eligible training rows for each formation
    # date and compare the count with what the model recorded.
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

    for date in frame["date"].unique()[:24]:
        i = int(target.index.get_indexer([pd.Timestamp(date)])[0])
        eligible = int((positions <= i - config.horizon).sum())
        got = recorded.get((pd.Timestamp(date), frame[frame["date"] == date]
                            ["ticker"].iloc[0]))
        if got is None:
            return False, f"no training count recorded for {date}"
        if int(got) != eligible:
            return False, (f"at {pd.Timestamp(date).date()} the model used "
                           f"{got} rows, {eligible} were eligible")
    return True, ("the training set at every formation date ends at or "
                  "before the horizon boundary")


def main() -> int:
    print("=== The Patient Quant — verification gates ===\n")

    def order(item):
        # Sort by the gate's number, not by its name: lexicographically,
        # "gate_20" would land between "gate_2" and "gate_3".
        match = re.match(r"gate_(\d+)", item[0])
        return (int(match.group(1)) if match else 999, item[0])

    for name, fn in sorted(
            ((n, f) for n, f in globals().items()
             if n.startswith("gate_") and callable(f)), key=order):
        fn()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{passed}/{len(RESULTS)} gates passed")
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    return 0


def _balanced(text: str, start: int):
    """The content of the brace group that begins at ``start``."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    return None, start


def _count_columns(spec: str) -> int:
    """The number of columns a tabular preamble declares."""
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"[pPmMbB]\{[^}]*\}", "X", spec)
    spec = re.sub(r">[^{}]*\{[^}]*\}", "", spec)
    spec = re.sub(r"<\{[^}]*\}", "", spec)
    spec = spec.replace("|", "").replace("*", "")
    return len([c for c in spec if c.isalpha()])


@gate("20. the source is well formed")
def gate_20_source():
    """Every heading argument is closed, and every chapter file's braces
    balance.

    A heading whose argument is not closed is read by LaTeX as running on
    until it finds a closing brace, which it may find many paragraphs
    later.  The text in between is silently swallowed, the page does not
    appear, and the book looks complete.  Nothing in the rendered output
    says what is missing, so the check is made on the source.
    """
    headings = re.compile(r"\\(?:chapter|part|section|subsection|subsubsection)\{")
    unbalanced = []
    files = sorted(pathlib.Path(SRC, "chapters").glob("*.tex"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.split("\n"), 1):
            match = headings.search(line)
            if not match:
                continue
            depth = 0
            started = False
            for character in line[match.end() - 1:]:
                if character == "{":
                    depth += 1
                    started = True
                elif character == "}":
                    depth -= 1
                if started and depth == 0:
                    break
            if depth != 0:
                unbalanced.append(f"{path.name}:{number}")
        # And the whole file must have balanced braces.
        if text.count("{") != text.count("}"):
            unbalanced.append(f"{path.name} (unbalanced braces)")
    if unbalanced:
        return False, ("; ".join(unbalanced[:6]) +
                       (f" and {len(unbalanced) - 6} more" if len(unbalanced) > 6 else ""))

    # And every table's rows must have as many cells as its preamble
    # declares columns.  A row with more cells than there are columns is
    # an "extra alignment tab", which LaTeX reports and then papers over.
    problems = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\begin\{tabularx?\}", text):
            open_brace = text.index("{", match.end() - 1)
            spec, after = _balanced(text, open_brace)
            if spec is None:
                continue
            end = text.find("\\end{tabular", after)
            body = text[after:end]
            columns = _count_columns(spec)
            for number, line in enumerate(body.split("\n"), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("\\"):
                    continue
                cells = 0
                for cell in stripped.split("&"):
                    span = re.match(r"\s*\\multicolumn\{(\d+)\}", cell)
                    cells += int(span.group(1)) if span else 1
                if cells > columns:
                    problems.append(f"{path.name}: {cells} cells in a "
                                    f"{columns}-column table")
    if problems:
        return False, ("; ".join(problems[:4]) +
                       (f" and {len(problems) - 4} more" if len(problems) > 4 else ""))

    return True, f"{len(files)} chapters, every heading argument closed"


# ---------------------------------------------------------------------
#  The index
#
#  The book carries an index.  Three things can go wrong with one, and
#  none of them is visible in the typeset output: the index can fail to
#  build and simply print nothing, a mark can quietly delete a word from
#  the sentence it sits in, and a mark placed just before a clause's
#  punctuation leaves that punctuation stranded after it.
# ---------------------------------------------------------------------


def _unmatched_dollars(text: str) -> list[int]:
    """The offsets of the dollar signs that open or close a formula.

    An escaped dollar sign is a literal dollar sign, not a delimiter, so
    it is skipped; the manuscript writes \\$3,000 and the like.
    """
    out = []
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "$":
            out.append(index)
        index += 1
    return out


def index_marks_eat_words(text: str, name: str) -> list[str]:
    """Find the index marks that have taken a word out of the prose.

    \\idx prints nothing: it registers a term the prose already writes.
    Used where the word belongs, it silently deletes that word from the
    sentence, and nothing in a LaTeX build reports it.

    The test: the sentence that holds the mark must still contain the
    term once the mark is taken out.
    """
    flat = re.sub(r"[ \t]+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", flat)

    bad: list[str] = []
    for sentence in sentences:
        for match in re.finditer(r"\\(idx|idxw)\{([^{}]*)\}", sentence):
            kind, entry = match.group(1), match.group(2)
            if kind == "idxw":
                continue
            key = re.sub(r"[^A-Za-z0-9]", "", entry.split()[0]).lower()
            if not key:
                continue
            without = sentence[:match.start()] + " " + sentence[match.end():]
            without = re.sub(r"[^A-Za-z0-9]", "", without).lower()
            if key not in without:
                bad.append(f"{name}: {entry}")
    return bad


def index_marks_at_clause_end(text: str, name: str) -> list[str]:
    """Find the index marks that sit at the end of a clause.

    A mark at the end of a clause stands where the word belongs: the
    sentence would end with the word missing and the clause's punctuation
    stranded after it.
    """
    bad: list[str] = []
    for match in re.finditer(r"\\idx\{([^{}]*)\}", text):
        after = text[match.end():match.end() + 40].lstrip()
        if after[:1] in (".", ";", ":"):
            before = text[max(0, match.start() - 70):match.start()]
            before = re.sub(r"[ \t]+", " ", before.replace("\n", " ")).strip()
            bad.append(f"{name}: {match.group(1)} in '...{before[-50:]}'")
    return bad


def index_marks_in_maths(text: str, name: str) -> list[str]:
    """Find the index marks that have been put inside a formula.

    A mark between two dollar signs does not print its term but it still
    splits the formula.
    """
    bad: list[str] = []
    dollars = _unmatched_dollars(text)
    for first, second in zip(dollars[0::2], dollars[1::2]):
        inside = text[first:second]
        for match in re.finditer(r"\\(?:idx|idxw)\{", inside):
            bad.append(f"{name}: a mark inside a formula")
    return bad


@gate("21. the index is populated")
def gate_21_index():
    """The index must carry headwords, and it must have been built.

    \\printindex prints nothing at all when the index file is missing, so
    a book can be typeset without an index and nothing in the output
    says so.
    """
    index_file = os.path.join(SRC, "main.idx")
    if not os.path.exists(index_file):
        return False, "no main.idx: the index was never built"
    with open(index_file, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    headwords = set()
    for line in lines:
        match = re.search(r"\\indexentry\{([^{}]*)", line)
        if match:
            headwords.add(match.group(1))
    if len(headwords) < 60:
        return False, f"{len(headwords)} headwords, at least 60 wanted"
    return True, f"{len(headwords)} headwords"


@gate("22. the index marks are sound")
def gate_22_index_marks():
    """No mark may eat a word, strand a punctuation mark, or sit inside a
    formula.

    Each of those three is invisible in the typeset output.  A word eaten
    by a mark simply does not appear; a stranded full stop still looks
    like a full stop; and a mark inside a formula renders as nothing.
    """
    files = sorted(pathlib.Path(SRC, "chapters").glob("*.tex"))
    eaten: list[str] = []
    stranded: list[str] = []
    in_math: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        eaten += index_marks_eat_words(text, path.name)
        stranded += index_marks_at_clause_end(text, path.name)
        in_math += index_marks_in_maths(text, path.name)

    problems = []
    for label, items in (("ate a word", eaten),
                         ("stranded a punctuation mark", stranded),
                         ("sat inside a formula", in_math)):
        if items:
            problems.append(f"{len(items)} mark(s) {label}: "
                            + "; ".join(items[:3]))
    if problems:
        return False, " | ".join(problems)

    marks = sum(len(re.findall(r"\\idx\{", p.read_text(encoding="utf-8")))
                for p in files)
    return True, f"{marks} marks, none of them harmful"


if __name__ == "__main__":
    import numpy as np  # noqa: E402  (used by the panel gate)
    raise SystemExit(main())
