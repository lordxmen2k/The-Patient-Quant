"""patientquant — the investment system behind *The Patient Quant*.

This package builds, runs and evaluates one complete investment system for
a single investor who is managing their own money.  It is the instrument the
book is written around: every number the book states about its own results
comes from running this code, and the book's verification gates check the
rendered PDF against what this code produces.

The design rules are the same ones the book argues for:

  * the strategic allocation is a stated policy and the model never sets it;
    the model may only tilt inside a sleeve, and it may hold cash;
  * the machine's job is to estimate the posterior over the edge, and the
    sizing rule shrinks toward cash when that posterior cannot tell the edge
    from zero;
  * every estimator is written in numpy from its objective function, so that
    the mathematics the book prints is the mathematics the code runs;
  * a trial family is declared before the tests are run, and the selection
    tests are reported whatever they say.

The entry point is :func:`patientquant.system.run`; the panel it runs on is
built by :func:`patientquant.panel.monthly_panel`, and the selection tests
live in :mod:`patientquant.selection`.
"""

__version__ = "1.0.0"

__all__ = [
    "costs",
    "data",
    "evaluation",
    "features",
    "fundamentals",
    "models",
    "panel",
    "portfolio",
    "risk",
    "selection",
    "stats",
    "system",
    "universe",
]
