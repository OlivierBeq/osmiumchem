# -*- coding: utf-8 -*-

"""Make the repository's scripts importable.

The benchmark harness lives in ``scripts/`` rather than in the package,
because it reads corpora from files and the package knows nothing about where
corpora live. It is still tested, so the directory goes on the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
