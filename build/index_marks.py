#!/usr/bin/env python3
"""
Index-mark placement engine.

The book uses two kinds of mark, defined in the preamble:

    \\idxw{Term}   prints Term and registers it in the index.  Use it where
                   the mark stands in for the word itself.
    \\idx{Term}    prints nothing and registers Term.  Use it where the
                   prose already writes the term, so that the mark only has
                   to sit beside the word.

This module finds a placeable occurrence of a term in a chapter and writes
the mark.  The canonical index entry is used, so that a term written in lower
case in one chapter and in title case in another produces one index entry
rather than two.

Three kinds of position are recognised, and they are handled differently:

  * plain prose.  The mark goes immediately after the word, which leaves the
    rendered prose byte for byte what it was before.
  * a heading argument.  \\chapter and \\section are moving arguments and
    LaTeX refuses an \\index inside them, so the mark is written just after
    the closing brace of the heading instead.
  * nothing.  The term is not in this chapter at all, and the caller is told
    so.

A candidate occurrence is skipped when it lies inside a formula, inside an
existing mark, or immediately before a full stop, a semicolon or a colon,
because a mark in those positions either breaks a formula or strands the
punctuation.

A term written with a hyphen also matches the LaTeX dashes --- so that
"bid-ask spread" is found where the manuscript writes "bid--ask spread".

Run from the repository root:  python3 build/add_index_marks.py
"""
import pathlib
import re

CHAPTERS = pathlib.Path("source/chapters")

# Commands whose argument is a heading.  \\chapter and \\section are moving
# arguments; \\index inside them is an error, so the mark is placed after
# the closing brace for all of them.
HEADING_CMDS = ("chapter", "section", "subsection", "subsubsection",
                "paragraph", "subparagraph")

# Commands whose argument is ordinary running text.  A mark may sit inside
# one of these: they are not moving arguments, so \index is safe there,
# and the text renders identically.
TEXT_CMDS = ("emph", "textit", "textbf", "textsc", "textsl", "textnormal",
             "mbox", "underline", "text")

# A dash, in any of the forms the manuscript uses.
DASH = r"[-‐‑–—―]"

# A separator between two words of the term: whitespace, or a LaTeX dash.
SPACE = r"(?:\s+|\s*" + DASH + r"{1,3}\s*)"


def term_pattern(term: str) -> re.Pattern:
    """Match the term as the manuscript may write it.

    Words are separated by whitespace or a dash; a hyphen inside a word
    matches any dash the manuscript uses in its place.
    """
    out = []
    for word in term.split():
        parts = re.split(r"[-‐‑–—―]+", word)
        out.append((DASH + "{1,3}").join(re.escape(p) for p in parts))
    return re.compile(r"(?<![\w\\])(" + SPACE.join(out) + r")(?![\w])",
                      re.IGNORECASE)


def _closing_brace(text: str, open_at: int) -> int:
    """Index of the brace that closes the one at open_at."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _open_argument(text: str, upto: int) -> tuple[str, int] | None:
    """The command name and the offset of its still-open argument brace.

    Returns None when there is no unbalanced brace before upto.
    """
    depth = 0
    i = upto - 1
    while i >= 0:
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth == 0:
                m = re.search(r"\\([A-Za-z]+)\s*$", text[:i])
                if not m:
                    return None
                return m.group(1), i
            depth -= 1
        i -= 1
    return None


def _in_math(text: str, upto: int) -> bool:
    """True if the text up to upto is inside a formula.

    An escaped dollar sign is a literal dollar sign, not a formula
    delimiter, so the count skips it.
    """
    count = 0
    i = 0
    while i < upto:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "$":
            count += 1
        i += 1
    return count % 2 == 1


def _mask_comments(text: str) -> str:
    """The text with its comments blanked out.

    A percent sign starts a comment that runs to the end of the line, and
    LaTeX throws the rest of the line away.  Every chapter opens with a
    comment naming the chapter, and several of those names contain the
    very word the index wants, so a search that does not skip comments
    puts the mark in a line that will never be typeset.

    The masked text keeps the original's length, so an offset found in
    the mask is an offset in the source.
    """
    out = list(text)
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "%":
            end = text.find("\n", index)
            if end == -1:
                end = length
            for position in range(index, end):
                out[position] = " "
            index = end
            continue
        index += 1
    return "".join(out)


def find_occurrence(text: str, term: str) -> tuple[int, int] | None:
    """The span of a placeable occurrence of term, or None."""
    masked = _mask_comments(text)
    pattern = term_pattern(term)
    for m in pattern.finditer(masked):
        start, end = m.span()
        before = masked[:start]
        after = masked[end:]
        if _in_math(masked, start):
            continue
        if "\\idxw{" in before[-12:] or "\\idx{" in before[-12:]:
            continue
        # already marked
        if after.startswith("\\idx{") or after.startswith("\\idxw{"):
            continue
        # a mark immediately before punctuation would strand it
        if after[:1] in (".", ";", ":"):
            continue
        opened = _open_argument(masked, start)
        if opened is None:
            return start, end
        cmd, brace_at = opened
        if cmd in HEADING_CMDS:
            close = _closing_brace(masked, brace_at)
            if close != -1:
                return close + 1, close + 1
        if cmd in TEXT_CMDS:
            # running text: the mark goes inside the argument
            return start, end
        # inside a formula, an existing mark, or another command's argument:
        # a mark here would break the prose, so the candidate is skipped
        continue
    return None


def add_mark(text: str, term: str) -> tuple[str, bool]:
    """Write the index mark for term into text, if it can be placed."""
    span = find_occurrence(text, term)
    if span is None:
        return text, False
    start, end = span
    if start == end:
        # a heading: the mark goes after the closing brace
        return text[:start] + "\\idx{" + term + "}" + text[start:], True
    return text[:end] + "\\idx{" + term + "}" + text[end:], True
