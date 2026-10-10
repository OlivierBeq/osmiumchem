# -*- coding: utf-8 -*-

"""Rendering benchmark results as text, Markdown and CSV.

Columns run in reading order: did the molecule come back unchanged, how sure
are we, what did it cost in bits, does that cost fit the budget. Every function
returns a string instead of printing, so one result set can go to a terminal, a
README or a spreadsheet.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only; benchmark imports this module
    from benchmark import BenchmarkReport, CodecResult

__all__ = ["render_csv", "render_failures", "render_markdown", "render_table"]

_COLUMNS = (
    ("codec", 22),
    ("family", 11),
    ("accuracy", 10),
    ("95% low", 9),
    ("ceiling", 8),
    ("med bits", 9),
    ("p99", 6),
    ("max", 6),
    ("fits", 6),
    ("native", 8),
)


def _accuracy_cell(result: CodecResult) -> str:
    """Format an accuracy with enough digits to see four nines."""
    if result.molecules == 0:
        return "-"
    return f"{100.0 * result.accuracy:.4f}%"


def _fits_cell(fraction: float | None) -> str:
    """Format a budget-compliance fraction without rounding up to 100%.

    A codec that overflows the budget once in ten thousand molecules is not
    compliant, and the OK flag treats it that way. Rounding 99.999% up to
    "100.0%" would contradict that column, so anything short of exact
    compliance is floored and shown with enough digits to read as below it.
    """
    if fraction is None:
        return "-"
    if fraction >= 1.0:
        return "100%"
    floored = math.floor(fraction * 10000.0) / 100.0
    return f"{floored:.2f}%"


def _optional(value: float | int | None, suffix: str = "") -> str:
    """Format a value that may be missing."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{100.0 * value:.2f}%{suffix}" if suffix == "%" else f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def render_table(report: BenchmarkReport, target: float = 0.9999) -> str:
    """Render the results as a fixed-width terminal table.

    :param report: The finished benchmark report.
    :param target: Accuracy target used to mark rows.
    """
    lines: list[str] = []
    lines.append(
        f"corpus {report.corpus}: {report.molecules} evaluation molecules, "
        f"{report.fit_molecules} fitted, budget {report.budget} bits",
    )
    lines.append("")
    header = "  ".join(name.rjust(width) if i else name.ljust(width) for i, (name, width) in enumerate(_COLUMNS))
    lines.append("    " + header)
    lines.append("    " + "-" * len(header))
    for result in report.sorted_by_bits():
        fits = result.within_budget(report.budget)
        mark = "OK " if result.supports_target(target) and (fits or 0.0) >= 1.0 else "   "
        if result.error:
            lines.append(f"{mark} {result.name.ljust(22)}  {result.family.rjust(11)}  ERROR {result.error[:70]}")
            continue
        cells = [
            result.name.ljust(_COLUMNS[0][1]),
            result.family.rjust(_COLUMNS[1][1]),
            _accuracy_cell(result).rjust(_COLUMNS[2][1]),
            f"{100.0 * result.interval[0]:.4f}%".rjust(_COLUMNS[3][1]),
            ("-" if result.ceiling is None else f"{100.0 * result.ceiling:.2f}%").rjust(_COLUMNS[4][1]),
            str(result.median_bits if result.median_bits is not None else "-").rjust(_COLUMNS[5][1]),
            str(result.percentile(0.99) if result.bits else "-").rjust(_COLUMNS[6][1]),
            str(result.max_bits if result.bits else "-").rjust(_COLUMNS[7][1]),
            _fits_cell(fits).rjust(_COLUMNS[8][1]),
            str(result.native_bits if result.native_bits else "-").rjust(_COLUMNS[9][1]),
        ]
        lines.append(mark + " " + "  ".join(cells))
    lines.append("")
    lines.append(
        "    'OK' marks codecs whose 95% lower bound clears the accuracy target "
        "and whose every payload fits the budget.",
    )
    lines.append(
        "    'ceiling' is the share of molecules with a unique payload: an upper bound on any decoder.",
    )
    lines.append("    'native' is the unquantised float cost, where one applies.")
    return "\n".join(lines)


def render_failures(report: BenchmarkReport, limit: int = 3) -> str:
    """Render a breakdown of how each codec failed.

    :param report: The finished benchmark report.
    :param limit: Failing examples shown per codec.
    """
    lines: list[str] = ["failure modes", ""]
    for result in report.results:
        if not result.failures and not result.error:
            continue
        total = sum(result.failures.values())
        detail = ", ".join(f"{kind}={count}" for kind, count in sorted(result.failures.items()))
        lines.append(f"  {result.name}: {total} failures ({detail or result.error})")
        for reference, decoded, kind in result.examples[:limit]:
            lines.append(f"      [{kind}] {reference[:68]}")
            lines.append(f"              -> {str(decoded)[:68]}")
    if len(lines) == 2:
        lines.append("  none")
    return "\n".join(lines)


def render_markdown(report: BenchmarkReport, target: float = 0.9999) -> str:
    """Render the results as a Markdown table."""
    lines = [
        f"### `{report.corpus}` — {report.molecules} molecules, {report.budget}-bit budget",
        "",
        "| codec | family | accuracy | 95% low | ceiling | median bits | p99 | max | fits budget | native bits |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in report.sorted_by_bits():
        if result.error:
            lines.append(f"| `{result.name}` | {result.family} | error | | | | | | | {result.error[:40]} |")
            continue
        fits = result.within_budget(report.budget)
        flag = " **OK**" if result.supports_target(target) and (fits or 0.0) >= 1.0 else ""
        lines.append(
            f"| `{result.name}`{flag} | {result.family} | {_accuracy_cell(result)} | "
            f"{100.0 * result.interval[0]:.4f}% | "
            f"{'-' if result.ceiling is None else f'{100.0 * result.ceiling:.2f}%'} | "
            f"{result.median_bits if result.median_bits is not None else '-'} | "
            f"{result.percentile(0.99) if result.bits else '-'} | "
            f"{result.max_bits if result.bits else '-'} | "
            f"{_fits_cell(fits)} | "
            f"{result.native_bits or '-'} |",
        )
    return "\n".join(lines)


def render_csv(reports: Sequence[BenchmarkReport]) -> str:
    """Render one or more reports as CSV, one row per codec and corpus."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "corpus",
            "codec",
            "family",
            "molecules",
            "successes",
            "accuracy",
            "wilson_low",
            "wilson_high",
            "injectivity_ceiling",
            "median_bits",
            "p99_bits",
            "max_bits",
            "mean_bits",
            "fraction_within_budget",
            "native_bits",
            "budget",
            "seconds",
            "lossless_by_design",
            "failures",
            "error",
            "notes",
        ],
    )
    for report in reports:
        for result in report.results:
            low, high = result.interval
            writer.writerow(
                [
                    report.corpus,
                    result.name,
                    result.family,
                    result.molecules,
                    result.successes,
                    f"{result.accuracy:.8f}",
                    f"{low:.8f}",
                    f"{high:.8f}",
                    "" if result.ceiling is None else f"{result.ceiling:.8f}",
                    result.median_bits if result.median_bits is not None else "",
                    result.percentile(0.99) if result.bits else "",
                    result.max_bits if result.bits else "",
                    "" if result.bits_per_molecule is None else f"{result.bits_per_molecule:.2f}",
                    "" if result.within_budget(report.budget) is None else f"{result.within_budget(report.budget):.6f}",
                    result.native_bits or "",
                    report.budget,
                    f"{result.seconds:.2f}",
                    result.lossless_by_design,
                    ";".join(f"{k}={v}" for k, v in sorted(result.failures.items())),
                    result.error or "",
                    result.notes,
                ],
            )
    return buffer.getvalue()
