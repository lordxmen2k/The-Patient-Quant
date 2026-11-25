# Build notes — The Patient Quant

This file records how the book is produced and what was repaired on the way, so that
a later edition can be rebuilt from the same materials and so that a reader who
doubts a number can find where it was computed. It is production documentation and
is deliberately not part of the book.

## Pipeline

```sh
./build.sh            # the book
build/make_covers.sh  # the covers (separate documents)
```

`build.sh` runs, in order:

1. the word count, from the book's own LaTeX source with the control words stripped.
2. `scripts/run_system.py` — the single number generator. It builds the monthly panel
   from the data cache, runs the model over the walk-forward window, runs the
   system end to end for every member of the declared trial family, applies the
   costs and the tax, runs the selection tests, and writes `build/numbers.json` and
   the series under `build/series/`.
3. `build/make_numbers_tex.py` — the 134 headline macros, written from
   `build/numbers.json` into `source/generated_numbers.tex`.
4. `build/make_appendix_tables.py` — the three appendix tables.
5. `build/make_figures.py` — the eighteen charts, then
   `build/check_fig_overlaps.py` over them.
6. `build/make_covers.sh` — the cover plates and their typography, if they are not
   already present.
7. XeLaTeX twice, `makeindex`, XeLaTeX twice.
8. the page count, then `source/generated.tex` is rewritten with the real page and
   word counts, then XeLaTeX twice more.
9. the PDF is copied to `output/The-Patient-Quant.pdf`.
10. `python3 build/verify.py` — the twenty gates — and then
    `python3 build/check_numbers.py`.

`SKIP_RUN=1 ./build.sh` reuses `build/numbers.json` instead of re-running the system.
`COVERS=1 ./build.sh` rebuilds the cover plates even when they are already there.

## One generator per artefact

Every figure, every appendix table and every macro has exactly one generator, and
the generators are never chained. A second copy of a generator is a race that fires
in the wrong order, and on an earlier book that is exactly what overwrote good
figures with broken ones. `source/generated_numbers.tex` is written only by
`build/make_numbers_tex.py`, `source/generated_appendix_*.tex` only by
`build/make_appendix_tables.py`, and `figures/fig_*.pdf` only by
`build/make_figures.py`.

## The index

The book carries an index, and it is built the way the reference volume built
its own: `build/add_index_marks.py` holds the table of terms and the chapter
each is indexed from, `build/index_marks.py` finds a place for each term, and
`makeindex` writes the page numbers. The placement engine is deliberately
conservative, because an index mark is invisible and the three ways it can go
wrong are all invisible too:

* a mark placed where the word belongs deletes the word from the sentence, and
  nothing in a LaTeX build reports the deletion;
* a mark placed just before a clause's punctuation strands that punctuation;
* a mark placed between two dollar signs splits the formula.

Two marks therefore never land where they would do one of those three things,
and gate 22 checks the manuscript for all three after the typesetting.

Four faults in the first placement engine were found and fixed, and each of
them had silently cost index entries:

* the search did not know that LaTeX comments begin with a percent sign, and
  twenty-six marks went into the comment line that names each chapter — the
  one line of the file that is never typeset. The search now masks the
  comments before it looks for anything;
* the search counted every dollar sign when it decided whether it was inside
  a formula, so a written-out `\$3,000` in Chapter 29 made everything after
  it look like mathematics. Escaped dollar signs are now skipped;
* the search did not know that the manuscript writes a hyphen as two hyphens,
  so `bid--ask spread` was not found by a search for `bid-ask spread`;
* the search did not know that a heading argument is a moving argument, in
  which LaTeX refuses an `\index`. A mark in a heading is now written after
  the heading's closing brace.

Where the book discussed a concept without naming it, the prose was changed
to name it — *active management*, *data snooping*, *multiple testing*,
*point-in-time data*, *moving average*, *leakage*, *purge and embargo*,
*hyperparameters*, *posterior predictive*, *combinatorially symmetric
cross-validation*, *volatility scaling*, *ablation*, *tail risk*, *insider
trading*. Those are small additions to the manuscript and they are the reason
the index can carry a headword for each concept rather than only for the
words the writer happened to use.

The script is idempotent: it strips the marks it finds before it places the
new ones, so the table of terms can be changed and the index rebuilt without
the marks accumulating.

## The number rule

No headline number is transcribed. It is computed by `scripts/run_system.py`,
rendered into the manuscript at the value the generator produces, and printed in full
in the appendices. Where a value is rounded, it is rounded in the direction the
generator gives. `build/check_numbers.py` is the mechanical guard: it re-derives
every one of the 134 macros from `build/numbers.json`, formats each the way the
book writes it, compares it with the macro file, and then looks for the value in
the rendered PDF.

## The data

One vendor, one endpoint, cached and checksummed: `data/cache/manifest.json` holds
a SHA-256 for every cached series, and gate 17 checks them. Returns are rebuilt from
the price and the dividend components rather than taken from the adjusted close,
and the adjusted close is kept only as a cross-check; the mean discrepancy is about
0.004 per cent a month.

**Months before a fund existed are masked, not zero-filled.** A zero return makes a
new fund look as though it had no volatility and no beta, which is the most
attractive possible appearance for a fund to have. The panel masks the pre-inception
months and the features are masked by the presence of a price.

**No revised macroeconomic series.** The panel is built from prices and dividends
only, so the data does not change under the book when a national accounts office
republishes a decade.

## Repairs made during the build

These are recorded because each was a class of error worth not repeating. Four of
them are the reason this file exists.

**The number macros were never typeset.** `source/generated_numbers.tex` was written
by the generator and never `\input` by the preamble. The book typeset cleanly, the
gates passed, and every number the book states about its own results was missing
from the page. The gate that should have caught it checked the source text, where
the macros were present, rather than the rendered PDF. Gate 7 now looks for the
value of every macro the text uses in the finished document.

**A LaTeX control word may not contain a digit.** `\numSpyW12` is tokenised as
`\numSpyW` followed by the characters `12`, so a macro with a digit in its name can
never be called, and its definition puts the digits into the document as stray
text. Twenty-four macros were affected, and the pages that quoted them lost the
numbers without any error being raised. The macros were renamed to spell the number
out, the generator now refuses to write a name with a digit in it, and gate 5 checks
the names as well as the definitions.

**The normal quantile function had its branches swapped.** `norm_ppf` evaluated
Acklam's central polynomial where the tail polynomial belongs, and then refined the
result against a CDF that was itself only accurate to 1e-8, which made the refinement
worse than the approximation it started from. `norm_ppf(0.975)` returned 1.562
instead of 1.960. Both the error function and the quantile function were rewritten
from their definitions — the Maclaurin series below the crossover and the
incomplete gamma continued fraction above it — and are now accurate to about 1e-16.
The tests in `tests/test_stats.py` check them against values that can be quoted.

**The wash-sale rule did not bite when it should.** A loss sale followed by a
repurchase within thirty days disallows the loss and adds it to the basis of the
replacement shares. The implementation only disallowed the loss when the
replacement shares were bought *below* the sale price, which is the one case where
the deduction is not lost. The rule turns on the timing, not on the price. The
reverse direction had a second fault: the queue of recent purchases was scanned
front to back and stopped at the first purchase outside the window, which meant
that the purchases inside the window, being newer, were never reached. Both are
fixed, and the tax model now carries the loss per share so that the basis
adjustment is right.

Three further faults were found and fixed in the measurement code itself and are
recorded in the book's own appendix: the horizon units were mismatched when the
ensemble combined its members, the pre-inception months were zero-filled, and the
ensemble score was pointwise where the documentation said it was cumulative.

## The tests

```sh
python3 -m unittest discover -s tests
```

The tests are written with the standard library's `unittest`, because the package's
dependency set is deliberately short: every estimator in `patientquant/` is written
in numpy from its objective function, including the normal and Student-t
distributions, the deflated Sharpe ratio, the CSCV probability of backtest
overfitting, the Hansen SPA test and the White reality check. That is a house rule
of the book and not an accident of the environment: the code that ships has to be
the code whose mathematics the book prints.

The tests that need the data cache skip themselves when the cache is not present.

## The covers

The covers are separate documents, because they are typeset at the trim size with
no page furniture. `build/make_cover_source_art.py` draws the source artwork from
the book's own measured record — the two wealth paths on a log axis, with the
index's worst drawdown washed in as the hole it fell into, over a field of the
sixty-one funds placed on their own risk and return. `build/make_cover_art.py`
crops and scrims it into the two plates, and `source/cover_front.tex` and
`source/cover_back.tex` set the type over them. The finished plates are committed,
so a fresh clone can build the book without running the cover script, and
`COVERS=1` rebuilds them.
