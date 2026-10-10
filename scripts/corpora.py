# -*- coding: utf-8 -*-

"""Read molecular corpora from files you name, and split them for fitting.

Nothing here knows where any corpus lives. Every source arrives as a
``--source path[:column]`` argument, so the corpora stay outside the
repository and the same script works on anyone's files.

Three properties matter for every corpus read here. Sampling is unbiased, by
seeded reservoir, because source files are often sorted (alphabetically by
SMILES, or grouped by assay) and the first ``n`` rows would be a chemically
skewed slice. The fit split is disjoint, carved by :func:`build_corpus` before
any codec sees the data, so a codec that learns a frequency table is not
scored on what it learned from. Inputs are canonical isomeric SMILES from the
start, so no codec is penalised for a spelling difference it inherited from a
source file.

Several sources aggregate into one corpus, deduplicated on canonical SMILES.
That matters when they overlap: activity tables repeat a compound once per
measurement, and the same molecule reached from two files must not land in the
fitting split *and* the evaluation split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from osmiumchem.identity import canonicalise_all

__all__ = [
    "Corpus",
    "SourceSpec",
    "add_source_arguments",
    "aggregate",
    "build_corpus",
    "canonicalise_all",
    "corpus_label",
    "parse_source",
    "read_sources",
]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One file to read SMILES from.

    :param path: The file.
    :param column: Column holding the SMILES, for tabular formats. ``None``
        picks the first column whose name contains "smiles".
    """

    path: Path
    column: str | None = None

    @property
    def reader(self) -> str:
        """Return the format, from the suffix."""
        suffixes = [part.lower() for part in self.path.suffixes]
        if ".parquet" in suffixes:
            return "parquet"
        if ".csv" in suffixes or ".tsv" in suffixes:
            return "csv"
        return "lines"

    @property
    def label(self) -> str:
        """Return a short name for logs and manifests."""
        return self.path.name

    def stream(self, *, distinct: bool = False) -> Iterator[str]:
        """Yield raw SMILES strings from the file.

        :param distinct: Narrow each Parquet batch to its distinct values
            first; see :func:`_iter_parquet_column`.
        """
        if not self.path.is_file():
            msg = f"no such source file: {self.path}"
            raise FileNotFoundError(msg)
        if self.reader == "parquet":
            yield from _iter_parquet_column(self.path, self.column, distinct=distinct)
        elif self.reader == "csv":
            yield from _iter_csv_column(self.path, self.column)
        else:
            yield from _iter_lines(self.path)


def parse_source(text: str) -> SourceSpec:
    """Parse a ``--source`` argument of the form ``path`` or ``path:column``.

    Split on the last colon, so a path may itself contain one. A trailing
    colon with nothing after it means "detect the column".

    :param text: The argument as typed.
    :raises argparse.ArgumentTypeError: If the path part is empty.
    """
    path, separator, column = text.rpartition(":")
    # A Windows drive letter is not a column separator.  # illustrative path
    if not separator or (len(path) == 1 and path.isalpha()):
        return SourceSpec(Path(text).expanduser(), None)
    if not path:
        msg = f"--source needs a file path, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return SourceSpec(Path(path).expanduser(), column or None)


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that name the corpora to read."""
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        metavar="FILE[:COLUMN]",
        default=[],
        help=(
            "corpus to read, repeatable. The column is optional and detected "
            "from the header when omitted. Several sources are aggregated and "
            "deduplicated on canonical SMILES"
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="name recorded for this corpus; defaults to the first source's filename",
    )


def _iter_csv_column(path: Path, column: str | None) -> Iterator[str]:
    """Yield a column from a CSV file, tolerating very large fields."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8", errors="replace") as handle:  # type: ignore[operator]
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        name = _pick_column(reader.fieldnames, column, path)
        for row in reader:
            value = row.get(name)
            if value:
                yield value.strip()


def _iter_parquet_column(path: Path, column: str | None, *, distinct: bool = False) -> Iterator[str]:
    """Yield a column from a Parquet file in batches.

    Reads one column, so a multi-gigabyte many-column file costs only the
    bytes of the SMILES.

    :param distinct: Deduplicate within each batch before crossing into
        Python. Activity tables repeat a compound once per measurement, and an
        ExCAPE-DB-sized table has hundreds of millions of rows over about a
        million compounds; Arrow's dictionary-aware ``unique`` saves minutes
        of building throwaway strings. Duplicates still cross batch
        boundaries, so this narrows the stream rather than replacing the
        deduplication downstream.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "reading Parquet corpora needs pyarrow; install it or convert the file to CSV"
        raise ImportError(msg) from exc
    handle = pq.ParquetFile(path)
    name = _pick_column(handle.schema_arrow.names, column, path)
    for batch in handle.iter_batches(batch_size=65536, columns=[name]):
        values = batch.column(name)
        if distinct:
            import pyarrow.compute as pc

            values = pc.unique(values)
        for value in values.to_pylist():
            if value:
                yield str(value).strip()


def _iter_lines(path: Path) -> Iterator[str]:
    """Yield SMILES from a one-per-line file, gzipped or plain.

    Takes the first whitespace-separated field, so a ``.smi`` file with a name
    column after the SMILES reads correctly.
    """
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[operator]
        for line in handle:
            value = line.strip()
            if value:
                yield value.split()[0]


def _pick_column(names: Sequence[str], column: str | None, path: Path) -> str:
    """Return the column to read, detecting one when none was given."""
    if column is not None and column in names:
        return column
    match = next((name for name in names if "smiles" in name.lower()), None)
    if match is None:
        wanted = f"no column {column!r} and " if column is not None else ""
        msg = f"{path.name}: {wanted}no SMILES-like column among {', '.join(names)}"
        raise KeyError(msg)
    return match


def _reservoir(stream: Iterable[str], limit: int | None, seed: int) -> list[str]:
    """Uniformly sample at most ``limit`` items from ``stream`` in one pass.

    Reservoir rather than a head slice because the source files are often
    sorted, and rather than a full shuffle because some are far too large to
    hold.

    :param stream: Iterable of candidate strings; empty entries are skipped.
    :param limit: Maximum number of items to keep; ``None`` keeps everything.
    :param seed: Seed making the draw reproducible.
    """
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    if limit is None:
        return [item for item in stream if item]
    reservoir: list[str] = []
    seen = 0
    for item in stream:
        if not item:
            continue
        if len(reservoir) < limit:
            reservoir.append(item)
        else:
            index = rng.randrange(seen + 1)
            if index < limit:
                reservoir[index] = item
        seen += 1
    return reservoir


def read_sources(sources: Sequence[SourceSpec], limit: int | None = None, seed: int = 0) -> list[str]:
    """Read raw SMILES from every source, sampled to at most ``limit``.

    Sampling spans the sources together rather than per file, so a corpus
    assembled from a large table and a small one is not half small table.

    :param sources: Files to read.
    :param limit: Maximum strings to keep; ``None`` keeps everything.
    :param seed: Sampling seed.
    :raises ValueError: If no source was given.
    """
    if not sources:
        msg = "no corpus given; pass at least one --source FILE[:COLUMN]"
        raise ValueError(msg)

    def chained() -> Iterator[str]:
        seen_raw: set[str] = set()
        for source in sources:
            # Raw-string deduplication is cheap and removes most of the volume
            # before canonicalisation, which is the expensive part.
            for value in source.stream(distinct=source.reader == "parquet"):
                if value not in seen_raw:
                    seen_raw.add(value)
                    yield value

    return _reservoir(chained(), limit, seed)


@dataclass(frozen=True, slots=True)
class Corpus:
    """A canonicalised, split corpus ready for fitting or benchmarking.

    :param name: Label recorded for the corpus.
    :param fit: Molecules codecs may learn from.
    :param evaluate: Molecules codecs are scored on; disjoint from ``fit``.
    :param rejected: Source strings RDKit could not parse, kept for the
        report.
    """

    name: str
    fit: list[str] = field(default_factory=list)
    evaluate: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of evaluation molecules."""
        return len(self.evaluate)

    @property
    def summary(self) -> str:
        """Return a one-line description of the split."""
        return (
            f"{self.name}: {len(self.evaluate)} evaluation molecules, "
            f"{len(self.fit)} fitting molecules, {len(self.rejected)} unparseable"
        )


def corpus_label(sources: Sequence[SourceSpec], label: str | None = None) -> str:
    """Return the name to record for a corpus read from ``sources``."""
    if label:
        return label
    if not sources:
        return "(none)"
    if len(sources) == 1:
        return sources[0].path.stem
    return f"{sources[0].path.stem}+{len(sources) - 1}"


def build_corpus(
    sources: Sequence[SourceSpec],
    *,
    evaluate: int | None = 20000,
    fit: int = 20000,
    seed: int = 0,
    workers: int = 1,
    label: str | None = None,
    fit_sources: Sequence[SourceSpec] | None = None,
) -> Corpus:
    """Read, canonicalise and split a corpus.

    Fitting and evaluation molecules are disjoint, so a codec that learns a
    frequency table is scored out of sample.

    :param sources: Files to evaluate on.
    :param evaluate: Number of evaluation molecules; ``None`` uses all
        available after the fitting split is removed.
    :param fit: Number of molecules reserved for fitting shared state.
    :param seed: Seed for sampling and splitting.
    :param workers: Processes used for canonicalisation.
    :param label: Name recorded for the corpus.
    :param fit_sources: Different files to fit on, for measuring how well a
        learned code transfers across chemical space.
    :returns: The prepared :class:`Corpus`.
    """
    name = corpus_label(sources, label)
    draw = None if evaluate is None else evaluate + fit
    pool = read_sources(sources, limit=draw, seed=seed)
    canonical, rejected = canonicalise_all(pool, workers=workers)
    rng = random.Random(seed)  # noqa: S311 - reproducible shuffling, not cryptography
    rng.shuffle(canonical)

    if fit_sources:
        foreign = read_sources(fit_sources, limit=fit, seed=seed + 1)
        fit_set, _ = canonicalise_all(foreign, workers=workers)
        evaluation = canonical if evaluate is None else canonical[:evaluate]
        return Corpus(name=name, fit=fit_set, evaluate=evaluation, rejected=rejected)

    fit_count = min(fit, max(0, len(canonical) - 1))
    fit_set = canonical[:fit_count]
    remaining = canonical[fit_count:]
    evaluation = remaining if evaluate is None else remaining[:evaluate]
    return Corpus(name=name, fit=fit_set, evaluate=evaluation, rejected=rejected)


def _append_new(
    raw: Sequence[str],
    handle: TextIO,
    seen: set[str],
    workers: int,
) -> tuple[int, int]:
    """Canonicalise ``raw`` and append the molecules not already written.

    :returns: ``(molecules written, strings rejected)``.
    """
    canonical, rejected = canonicalise_all(raw, workers=workers)
    written = 0
    for molecule in canonical:
        if molecule in seen:
            continue
        seen.add(molecule)
        # SMILES hold no commas, so one molecule per line is valid CSV.
        handle.write(f"{molecule}\n")
        written += 1
    return written, len(rejected)


def _source_rows(source: SourceSpec) -> int | None:
    """Return the number of rows in a source, when it is cheap to know.

    Parquet keeps the count in its footer. Finding it for a CSV means reading
    the file twice, which is not worth it, so the manifest records ``None``.
    """
    if source.reader != "parquet" or not source.path.is_file():
        return None
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(source.path).metadata.num_rows)


def aggregate(
    sources: Sequence[SourceSpec],
    target: Path | str,
    *,
    workers: int = 8,
    chunk: int = 200_000,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Aggregate several corpora into one file holding each molecule once.

    Sources usually overlap and repeat compounds per measurement, so
    deduplication happens twice: on the raw string as it streams, which is
    cheap and removes most of the volume, then on the canonical SMILES, which
    is what molecular identity means here. A molecule in four sources becomes
    one row, and that row decides which split it falls in, so no codec can be
    fitted on a molecule it is later scored on.

    Deduplicating on the raw string alone would not be enough, because sources
    disagree about aromatic notation and atom order, so the same molecule
    appears written several ways.

    Written to a ``.partial`` file and renamed, with a sibling
    ``.manifest.json`` recording what each source contributed.

    :param sources: Corpora to aggregate.
    :param target: Destination file.
    :param workers: Processes used for canonicalisation.
    :param chunk: Molecules canonicalised per batch. Larger batches use the
        process pool better, smaller ones keep memory flatter.
    :param log: Called with one progress line per source.
    :returns: The manifest. ``values_read`` counts the strings that reached
        Python, fewer than the rows for a Parquet source since each batch is
        narrowed to its distinct values first. ``source_rows`` holds the true
        row count where the format records one.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    announce = log if log is not None else (lambda message: None)

    seen_raw: set[str] = set()
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    partial = target.with_suffix(target.suffix + ".partial")
    started = time.perf_counter()
    with partial.open("w", encoding="utf-8") as handle:
        handle.write("smiles\n")
        for source in sources:
            clock = time.perf_counter()
            rows = fresh = written = rejected = 0
            pending: list[str] = []
            # Narrowing each Parquet batch to its distinct values is what makes
            # a hundred-million-row activity table affordable to stream.
            for raw in source.stream(distinct=source.reader == "parquet"):
                rows += 1
                if raw in seen_raw:
                    continue
                seen_raw.add(raw)
                fresh += 1
                pending.append(raw)
                if len(pending) >= chunk:
                    added, lost = _append_new(pending, handle, seen, workers)
                    written += added
                    rejected += lost
                    pending.clear()
            if pending:
                added, lost = _append_new(pending, handle, seen, workers)
                written += added
                rejected += lost
            records.append(
                {
                    "source": source.label,
                    "column": source.column,
                    "source_rows": _source_rows(source),
                    "values_read": rows,
                    "new_strings": fresh,
                    "molecules_added": written,
                    "rejected": rejected,
                    "seconds": round(time.perf_counter() - clock, 1),
                },
            )
            announce(
                f"{source.label:40s} {rows:>12,} values -> {written:>10,} new molecules "
                f"({rejected:,} unparseable, {time.perf_counter() - clock:.0f}s)",
            )
    partial.replace(target)

    manifest = {
        "molecules": len(seen),
        "rejected": sum(record["rejected"] for record in records),
        "sources": records,
        "seconds": round(time.perf_counter() - started, 1),
        "created": time.time(),
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    announce(f"{len(seen):,} unique molecules -> {target}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate the named sources into one deduplicated file.

    Worth doing once when the same corpus feeds several runs: the aggregate
    reads back as a single ``--source`` with no canonicalisation to redo.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/corpora.py",
        description="Aggregate corpora into one file holding each molecule once.",
    )
    add_source_arguments(parser)
    parser.add_argument("--out", type=Path, required=True, help="destination file")
    parser.add_argument("--workers", type=int, default=8, help="processes used to canonicalise")
    args = parser.parse_args(argv)
    if not args.source:
        parser.error("at least one --source is required")
    aggregate(args.source, args.out, workers=args.workers, log=lambda line: print(line, file=sys.stderr))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
