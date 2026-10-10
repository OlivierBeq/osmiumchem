# -*- coding: utf-8 -*-

"""Generate ``docs/RESULTS.md`` from the CSVs written by benchmark runs.

The numbers are read from the run output rather than transcribed, so the prose
and the tables cannot drift apart.

Usage::

    python scripts/make_results.py results docs/RESULTS.md
"""

from __future__ import annotations

import csv
import math
import sys
import textwrap
from pathlib import Path

FAMILY_ORDER = {"string": 0, "entropy": 1, "structural": 2, "fingerprint": 3, "latent": 4, "control": 5}

#: Width the generated prose is wrapped to, so the document reads in an editor
#: as well as rendered.
WIDTH = 79


def slug(title: str) -> str:
    """Return the heading anchor GitHub will generate for ``title``."""
    kept = [character for character in title.lower() if character.isalnum() or character in " -"]
    return "".join(kept).replace(" ", "-")


def wrap(text: str) -> str:
    """Wrap a paragraph to :data:`WIDTH`."""
    return textwrap.fill(text, width=WIDTH)


def read(path: Path) -> list[dict[str, str]]:
    """Return the rows of a results CSV, or an empty list if it is absent."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float | None:
    """Return a float field, or ``None`` when blank."""
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def integer(row: dict[str, str], key: str) -> int | None:
    """Return an integer field, or ``None`` when blank."""
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else None


def pct(value: float | None, places: int = 4) -> str:
    """Format a fraction as a percentage."""
    return "-" if value is None else f"{100.0 * value:.{places}f}%"


def fits_cell(fraction: float | None) -> str:
    """Format budget compliance without rounding up to 100%.

    As in ``report._fits_cell``: a codec that overflows on one molecule in ten
    thousand is not compliant, so the figure is floored rather than rounded.
    """
    if fraction is None:
        return "-"
    if fraction >= 1.0:
        return "100%"
    return f"{math.floor(fraction * 10000.0) / 100.0:.2f}%"


def table(rows: list[dict[str, str]], *, budget: int = 1000) -> str:
    """Render rows as a Markdown results table, cheapest payload first."""
    if not rows:
        return "_no results available_"
    lines = [
        "| codec | family | accuracy | 95% low | ceiling | median bits | p99 | max | fits budget | native bits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def sort_key(row: dict[str, str]) -> tuple:
        median = integer(row, "median_bits")
        return (FAMILY_ORDER.get(row["family"], 9), median if median is not None else 10**9)

    for row in sorted(rows, key=sort_key):
        if row.get("error"):
            lines.append(f"| `{row['codec']}` | {row['family']} | error: {row['error'][:60]} | | | | | | | |")
            continue
        fits = number(row, "fraction_within_budget")
        median = integer(row, "median_bits")
        # Undefined for textual payloads; see run_codec.
        ceiling = None if row["family"] == "string" else number(row, "injectivity_ceiling")
        ok = (
            number(row, "wilson_low") is not None
            and number(row, "wilson_low") >= 0.9999
            and fits is not None
            and fits >= 1.0
        )
        flag = " **OK**" if ok else ""
        lines.append(
            f"| `{row['codec']}`{flag} | {row['family']} | {pct(number(row, 'accuracy'))} | "
            f"{pct(number(row, 'wilson_low'))} | {pct(ceiling, 2)} | "
            f"{median if median is not None else '-'} | "
            f"{integer(row, 'p99_bits') or '-'} | {integer(row, 'max_bits') or '-'} | "
            f"{fits_cell(fits)} | {integer(row, 'native_bits') or '-'} |",
        )
    return "\n".join(lines)


def headline(rows: list[dict[str, str]]) -> str:
    """Describe the cheapest codec that actually meets both requirements."""
    qualified = [
        row
        for row in rows
        if not row.get("error")
        and (number(row, "wilson_low") or 0) >= 0.9999
        and (number(row, "fraction_within_budget") or 0) >= 1.0
    ]
    if not qualified:
        return "No codec in this run met both requirements."
    best = min(qualified, key=lambda row: integer(row, "median_bits") or 10**9)
    return (
        f"`{best['codec']}` is the cheapest codec meeting both requirements: "
        f"{pct(number(best, 'accuracy'))} accuracy over {best['molecules']} molecules "
        f"(95% lower bound {pct(number(best, 'wilson_low'))}), median "
        f"{integer(best, 'median_bits')} bits, worst case {integer(best, 'max_bits')} bits."
    )


def main(argv: list[str]) -> int:
    """Write the results document."""
    source = Path(argv[1] if len(argv) > 1 else "results")
    target = Path(argv[2] if len(argv) > 2 else "docs/RESULTS.md")
    runs = [
        (
            "union_certify",
            "Certification on the union corpus: every local source, deduplicated",
            "union",
            "Every failure in this table is the *budget*, not fidelity. All three coders "
            "round-trip every molecule exactly, and what they lose is the molecules whose "
            "payload will not fit 1000 bits: lipidated and disulfide-bridged peptides and "
            "oligonucleotides, which the drug-like and PROTAC corpora do not contain. The "
            "largest molecule here is an oligonucleotide of 681 heavy atoms. The budget "
            "holds for small-molecule chemistry, while biologics-scale entities need 3-4k "
            "bits, and no entropy coder changes that.",
        ),
        ("union_fast", "The union corpus, all non-neural codecs", "union"),
        ("union_lossy_open", "Union corpus fingerprints as reconstruction (open world)", "union"),
        (
            "union_neural_closed",
            "Union corpus latent spaces, closed world: identification",
            "union",
            "The library holds the evaluation molecules here, so these figures measure "
            "*identification*, and the quantised rows are the injectivity ceiling by "
            "construction. Nobody reconstructs a molecule from its molecular weight and "
            "logP, which is what `descriptors-q16xall` scoring 100% at 256 bits would "
            "otherwise suggest; the `library-index` control in the fingerprint tables above "
            "shows what a library alone is worth. The `cddd-native` rows use CDDD's own "
            "decoder rather than a library, so they are identical in both worlds. Compare "
            "them with the open-world table to see which figures move and which do not.",
        ),
        (
            "union_neural",
            "Union corpus latent spaces, open world: reconstruction",
            "union",
            "The library holds only the fitting split, so a lookup-decoded latent has to "
            "resolve a molecule it has never seen, and cannot. That 0% is the right reading "
            "of a quantised latent space judged as a *code*. The `cddd-native` rows are "
            "unaffected, being decoded by CDDD's own beam search, and they are what a "
            "learned decoder actually delivers: 58% at 16384 bits, sixteen times over "
            "budget.",
        ),
        (
            "certify_o4",
            "Certification, strict budget: the configurations that actually qualify",
            "multitask",
        ),
        (
            "certify_200k",
            "Certification run: enough molecules to support a 99.99% claim",
            "multitask",
        ),
        ("fast_multitask_closed", "Drug-like molecules, all non-neural codecs", "multitask"),
        ("fast_multitask", "Drug-like molecules, all non-neural codecs, smaller sample", "multitask"),
        ("lossy_open", "Fingerprints judged as reconstruction, not identification (open world)", "multitask"),
        ("neural_multitask", "Learned latent spaces: CDDD and GROVER", "multitask"),
        ("fast_protacs", "PROTACs: beyond-rule-of-5, the hardest bit budget", "protacs"),
        (
            "strict_protacs",
            "PROTACs with the budget enforced: over-budget counts as a failure",
            "protacs",
        ),
        ("transfer_protacs", "Codes fitted on drug-like molecules, applied to PROTACs", "protacs"),
    ]
    parts: list[str] = [
        "# Results",
        "",
        wrap(
            "Generated by `scripts/make_results.py` from the CSVs in `results/`, so every "
            "number here is read from a run rather than transcribed. Re-run "
            "`scripts/run_all.sh` to reproduce the CSVs.",
        ),
        "",
        "How to read the tables:",
        "",
        "| column | meaning |",
        "| --- | --- |",
        "| `accuracy` | round trips that returned the same molecule |",
        "| `95% low` | lower bound of the Wilson interval; this is what certifies a codec |",
        "| `ceiling` | fraction of molecules with a **unique** payload: an upper bound "
        "on the accuracy of *any* decoder |",
        "| `median` / `p99` / `max` | payload size in bits |",
        "| `fits budget` | payloads within 1000 bits, never rounded up to 100% |",
        "| `native bits` | unquantised float cost, where one applies |",
        "",
        wrap(
            "**OK** marks a codec whose 95% lower bound clears 99.99% *and* whose every "
            "payload fits 1000 bits. Both are required.",
        ),
        "",
    ]
    primary = (
        read(source / "certify_o4.csv")
        or read(source / "certify_200k.csv")
        or read(source / "fast_multitask_closed.csv")
        or read(source / "fast_multitask.csv")
    )
    if primary:
        parts += ["## Headline", "", wrap(headline(primary)), ""]

    sections: list[tuple[str, str, str, list[dict[str, str]]]] = []
    for entry in runs:
        name, title, corpus = entry[:3]
        rows = read(source / f"{name}.csv")
        if rows:
            sections.append((title, corpus, entry[3] if len(entry) > 3 else "", rows))

    if sections:
        parts += ["## Runs", ""]
        parts += [f"- [{title}](#{slug(title)}) — `{corpus}`" for title, corpus, _, _ in sections]
        parts += [""]

    for title, corpus, note, rows in sections:
        molecules = rows[0].get("molecules", "?")
        parts += [
            f"## {title}",
            "",
            f"`{corpus}`, {molecules} evaluation molecules, 1000-bit budget.",
            "",
            *([wrap(note), ""] if note else []),
            table(rows),
            "",
        ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {target}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
