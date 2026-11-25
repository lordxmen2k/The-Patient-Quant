#!/usr/bin/env python3
"""
Report, for every index term, the chapters in which a placeable occurrence
actually exists.  Used to relocate terms that were assigned to a chapter they
do not appear in, and to show which terms the manuscript never names at all.

Run from the repository root:  python3 build/scan_index_terms.py
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from add_index_marks import CHAPTERS, MARKS  # noqa: E402
from index_marks import find_occurrence  # noqa: E402

STRIP = re.compile(r"\\idx\{[^{}]*\}")


def main() -> None:
    texts = {}
    for path in sorted(CHAPTERS.glob("ch*.tex")):
        # The marks already in the file are removed first, so that a term
        # that has been marked is not mistaken for one that is missing.
        texts[int(path.stem[2:])] = STRIP.sub("", path.read_text(encoding="utf-8"))

    absent, misplaced = [], []
    for num, term in MARKS:
        where = [c for c, t in texts.items() if find_occurrence(t, term)]
        if num in where:
            continue
        if where:
            misplaced.append((num, term, where))
        else:
            absent.append((num, term))

    print(f"terms: {len(MARKS)}  misplaced: {len(misplaced)}  "
          f"absent from the manuscript: {len(absent)}\n")

    print("--- MISPLACED (relocate to a chapter that has the term) ---")
    for num, term, where in misplaced:
        print(f"  ch{num:02d}: {term}  ->  "
              f"{', '.join('ch%02d' % c for c in where)}")

    print("\n--- ABSENT (the manuscript never names the term) ---")
    for num, term in absent:
        print(f"  ch{num:02d}: {term}")


if __name__ == "__main__":
    main()
