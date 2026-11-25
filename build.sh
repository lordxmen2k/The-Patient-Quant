#!/bin/sh
# Build "The Patient Quant" to PDF.
#
# Usage: ./build.sh
#
# The order matters.  The numbers are computed first, so that every figure,
# every macro and every appendix table comes from the same canonical
# source.  The covers are built before the typesetting, because the master
# file puts them in as full-bleed leaves.  The verification gates run after
# the typesetting, and the build fails if any of them fails.
#
# Environment:
#   SKIP_RUN=1   reuse build/numbers.json instead of re-running the system
#   COVERS=1     rebuild the cover plates even if they are already there
set +e
cd "$(dirname "$0")"
SRC=source
OUT=output
mkdir -p "$OUT"

# 0. make the book fonts available to fontconfig (Outfit, DejaVu Sans Mono)
mkdir -p "$HOME/.fonts"
cp fonts/Outfit-*.ttf fonts/DejaVuSansMono-*.ttf "$HOME/.fonts/" 2>/dev/null || true
fc-cache -f >/dev/null 2>&1 || true

# 1. word count from the book's own source text
python3 - <<'PY'
import re, pathlib
src = pathlib.Path("source")
files = sorted(src.glob("chapters/*.tex")) + [src/"frontmatter.tex", src/"backmatter.tex"]
words = 0
for f in files:
    t = f.read_text(encoding="utf-8")
    t = re.sub(r"\\idx\{[^}]*\}", " ", t)
    t = re.sub(r"\\(begin|end)\{[^}]*(\[[^\]]*\])?\}", " ", t)
    t = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}$^_&~\\]", " ", t)
    t = re.sub(r"\s+", " ", t)
    words += len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])
pathlib.Path("source/.wordcount").write_text(str(words), encoding="utf-8")
print("body word count:", words)
PY

# 2. the numbers.  One generator, run before everything that quotes them.
if [ -z "$SKIP_RUN" ]; then
  python3 scripts/run_system.py >/dev/null || echo "WARNING: run_system.py failed"
else
  echo "skipping the system run (SKIP_RUN set); using build/numbers.json"
fi
python3 build/make_numbers_tex.py >/dev/null || echo "WARNING: make_numbers_tex.py failed"
python3 build/make_appendix_tables.py >/dev/null || echo "WARNING: make_appendix_tables.py failed"

# 3. figures: eighteen computed charts.  One generator per artefact, so
#    that no stale copy can overwrite a good one.
python3 build/make_figures.py >/dev/null || echo "WARNING: make_figures.py failed"
python3 build/check_fig_overlaps.py >/dev/null 2>&1 || echo "WARNING: a figure has overlapping labels"

# 3a. the covers.  They are separate documents, so they are built by their
#     own script; the master file puts the finished plates in as full-bleed
#     leaves, so they have to exist before the typesetting.
if [ "$COVERS" = "1" ] || [ ! -f figures/cover_front.pdf ] || [ ! -f figures/cover_back.pdf ]; then
  ./build/make_covers.sh >/dev/null || echo "WARNING: the covers failed to build"
else
  echo "reusing the existing cover plates (COVERS=1 to rebuild them)"
fi

# 4. typeset
cd "$SRC"
xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || true
xelatex -interaction=nonstopmode main.tex >/dev/null
if [ -f main.idx ]; then makeindex -q main.idx 2>/dev/null || true; fi
xelatex -interaction=nonstopmode main.tex >/dev/null
xelatex -interaction=nonstopmode main.tex >/dev/null

# 5. page count, then refresh the colophon
cd ..
TOTAL=$(pdfinfo "$SRC/main.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
PAGES=$((TOTAL - 2))   # exclude the front and back cover leaves
WORDS=$(cat source/.wordcount)
python3 - "$PAGES" "$WORDS" <<'PY'
import sys, pathlib
pages, words = sys.argv[1], int(sys.argv[2])
formatted = f"{words:,}"
pathlib.Path("source/generated.tex").write_text(
    "\\newcommand{\\bookPages}{" + pages + "}\n"
    "\\newcommand{\\bookWords}{" + formatted + "}\n", encoding="utf-8")
PY
cd "$SRC"
xelatex -interaction=nonstopmode main.tex >/dev/null
xelatex -interaction=nonstopmode main.tex >/dev/null
cd ..
cp "$SRC/main.pdf" "$OUT/The-Patient-Quant.pdf"

# 6. the verification gates
echo
echo "=== verification gates ==="
python3 build/verify.py
STATUS=$?

echo
echo "=== headline numbers ==="
python3 build/check_numbers.py
[ $? -ne 0 ] && STATUS=1

echo
echo "PAGES=$PAGES (of $TOTAL including covers)  WORDS=$WORDS"
pdfinfo "$OUT/The-Patient-Quant.pdf" | head -12
exit $STATUS
