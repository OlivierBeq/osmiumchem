# -*- coding: utf-8 -*-

"""Command-line entry point.

The verbs live in :mod:`osmiumchem.cli_transcode`: ``encode``, ``decode``,
``roundtrip``, ``fit`` and ``codecs``. This is the thin wrapper that makes
``python -m osmiumchem`` and the ``osmiumchem`` script the same thing.

The benchmark that produced ``docs/RESULTS.md`` is not here. It reads corpora
from files, which this package deliberately knows nothing about, so it lives
in ``scripts/benchmark.py`` in the repository.
"""

from __future__ import annotations

import sys

from .cli_transcode import run

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    :param argv: Argument list, defaulting to :data:`sys.argv`.
    :returns: Process exit status.
    """
    return run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
