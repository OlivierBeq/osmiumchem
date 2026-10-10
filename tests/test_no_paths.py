# -*- coding: utf-8 -*-

"""Guard against a developer's own filesystem creeping back into the project.

The corpora live outside this repository and are nobody else's to find, so a
hardcoded path is both useless to a user and a leak of where the author keeps
their files. The package once carried five of them.

A line that needs a path-shaped string to illustrate something says so with a
trailing ``# illustrative path`` comment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SELF = Path(__file__).resolve()
ROOT = SELF.parent.parent

#: Directories whose contents must name no absolute path.
SEARCHED = ("src", "scripts", "tests")

#: Opt-out for a line that shows what a path looks like rather than using one.
ALLOW = "# illustrative path"

#: What a path on someone's machine looks like. ``/tmp`` and ``/usr`` are left
#: out: they are the same everywhere and appear in runnable examples.
PATTERNS = (
    r"/mnt/[a-z]/",
    r"/home/[a-z0-9._-]+/",
    r"/Users/[a-z0-9._-]+/",
    r"[A-Z]:\\",
    r"Dropbox|OneDrive|iCloud",
)
COMPILED = tuple(re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS)


def _sources() -> list[Path]:
    """Return every Python file that ships or runs from this repository."""
    found: list[Path] = []
    for directory in SEARCHED:
        found.extend(
            sorted(
                path for path in (ROOT / directory).rglob("*.py") if "__pycache__" not in path.parts and path != SELF
            ),
        )
    return found


@pytest.mark.parametrize("path", _sources(), ids=lambda path: str(path.relative_to(ROOT)))
def test_no_absolute_paths_in_sources(path):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if ALLOW in line:
            continue
        for pattern in COMPILED:
            found = pattern.search(line)
            assert found is None, f"{path.relative_to(ROOT)}:{number} names a local path: {found.group(0)!r}"


def test_the_guard_would_catch_one():
    # A test that cannot fail is not a guard, so prove each pattern fires.
    offenders = (
        "root = '/mnt/c/corpora'",
        "root = '/home/someone/corpora'",
        "root = '/Users/someone/corpora'",
        r"root = 'D:\corpora'",
        "root = 'My Dropbox'",
    )
    for offender, pattern in zip(offenders, COMPILED, strict=True):
        assert pattern.search(offender), offender
