# -*- coding: utf-8 -*-

r"""The benchmark runner: fidelity, bit cost and why a codec failed.

This is the harness that produced ``docs/RESULTS.md``. It lives in
``scripts/`` rather than in the package because it reads corpora from files,
and the package deliberately knows nothing about where corpora live. Name them
yourself::

    python scripts/benchmark.py --source ~/corpora/protacs.csv:canonical_smiles \\
        --suite fast --evaluate 0 --fit 3000

Round-trip accuracy is the headline, but a bare percentage hides most of what
matters, so a run reports four things.

Accuracy comes with a Wilson score interval, and
:meth:`CodecResult.supports_target` answers yes only when the *lower* bound
clears the target. That bar is strict. A run with no failures at all reaches a
95% lower bound of 99.96% at ten thousand molecules and 99.98% at twenty
thousand, so certifying 99.99% takes around forty thousand clean molecules.
Anything smaller can refute a codec but not confirm one, hence the large
default evaluation split.

Bit cost is reported as a distribution, since a hard ceiling makes the tail
matter more than the median: results carry the median, the 99th percentile, the
maximum and the fraction of molecules inside the budget.

Failures are classified, because a decode that returns nothing, one that
returns different connectivity and one that returns the right skeleton with a
flipped stereocentre are three different problems. :func:`classify_failure`
tells them apart.

For lossy codecs the fraction of molecules with a unique payload bounds the
accuracy of *any* decoder, so a poor score can be attributed to the code itself
rather than to the decoder used to read it.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corpora import Corpus, add_source_arguments, build_corpus, parse_source
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from report import render_csv, render_failures, render_markdown, render_table

from osmiumchem.codecs import SUITES, build_suite, codec_names
from osmiumchem.codecs.base import BaseCodec
from osmiumchem.codecs.vectors import BIT_BUDGET, injectivity
from osmiumchem.identity import canonical_smiles, same_molecule, silence_rdkit
from osmiumchem.types import Bits

__all__ = [
    "BenchmarkReport",
    "CodecResult",
    "classify_failure",
    "main",
    "run_benchmark",
    "run_codec",
    "wilson_interval",
]


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a success rate.

    Wilson rather than the textbook normal approximation, because it stays
    sensible at the edge of the range, which is where this benchmark lives. A
    codec with zero failures still has a meaningful upper bound on its error
    rate, where the normal approximation collapses to zero width.

    :param successes: Number of successful round trips.
    :param total: Number of molecules attempted.
    :param z: Standard-normal quantile; the default is two-sided 95%.
    :returns: ``(low, high)`` bounds on the true success rate.
    """
    if total <= 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4 * total * total),
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def classify_failure(reference: str, decoded: str | None) -> str:
    """Describe how a decoded molecule differs from the original.

    :param reference: Canonical SMILES that was encoded.
    :param decoded: Whatever the decoder returned.
    :returns: One of ``"no-output"`` (nothing came back), ``"unparseable"``
        (a string came back but is not a molecule), ``"stereo-only"`` (right
        skeleton, wrong stereochemistry), ``"formula-kept"`` (same atoms,
        different connectivity, the signature of a tautomer or rearrangement)
        or ``"formula-changed"`` (atoms gained or lost).
    """
    if decoded is None:
        return "no-output"
    candidate = Chem.MolFromSmiles(decoded)
    if candidate is None:
        return "unparseable"
    original = Chem.MolFromSmiles(reference)
    if original is None:  # pragma: no cover - references are pre-canonicalised
        return "unparseable"
    flat_reference = Chem.MolToSmiles(original, isomericSmiles=False)
    flat_candidate = Chem.MolToSmiles(candidate, isomericSmiles=False)
    if flat_reference == flat_candidate:
        return "stereo-only"
    if rdMolDescriptors.CalcMolFormula(original) == rdMolDescriptors.CalcMolFormula(candidate):
        return "formula-kept"
    return "formula-changed"


@dataclass(slots=True)
class CodecResult:
    """Everything measured about one codec on one corpus."""

    name: str
    family: str
    molecules: int = 0
    successes: int = 0
    #: Payload size for *every* molecule the codec encoded, round trip or not.
    #: Size belongs to the encoding rather than to the decoder's luck, and
    #: recording only successes would empty the size columns for the lossy
    #: codecs whose size matters most, leaving no way to show that a 2048-bit
    #: fingerprint overflows a 1000-bit budget.
    bits: list[int] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    examples: list[tuple[str, str | None, str]] = field(default_factory=list)
    ceiling: float | None = None
    native_bits: int | None = None
    seconds: float = 0.0
    notes: str = ""
    lossless_by_design: bool = True
    error: str | None = None

    @property
    def accuracy(self) -> float:
        """Return the measured round-trip accuracy."""
        return self.successes / self.molecules if self.molecules else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        """Return the 95% Wilson interval on the accuracy."""
        return wilson_interval(self.successes, self.molecules)

    def supports_target(self, target: float = 0.9999) -> bool:
        """Return whether the evidence supports accuracy of at least ``target``.

        The *lower* confidence bound has to clear the target, so a clean but
        small run cannot certify a high accuracy on its own.
        """
        return self.interval[0] >= target

    def percentile(self, fraction: float) -> int | None:
        """Return a percentile of the payload size, in bits."""
        if not self.bits:
            return None
        ordered = sorted(self.bits)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    @property
    def median_bits(self) -> int | None:
        """Return the median payload size in bits."""
        return self.percentile(0.5)

    @property
    def max_bits(self) -> int | None:
        """Return the largest payload size in bits."""
        return max(self.bits) if self.bits else None

    def within_budget(self, budget: int = BIT_BUDGET) -> float | None:
        """Return the fraction of payloads that fit ``budget``."""
        if not self.bits:
            return None
        return sum(1 for value in self.bits if value <= budget) / len(self.bits)

    @property
    def bits_per_molecule(self) -> float | None:
        """Return the mean payload size in bits."""
        return sum(self.bits) / len(self.bits) if self.bits else None


def _matches(reference: str, candidate: str) -> bool:
    """Return whether ``candidate`` is the same molecule as ``reference``.

    The cheap string comparisons come first, since the benchmark feeds
    canonical references and they carry nearly every call. The full re-parse is
    reached only on an apparent failure, so it costs nothing in the common case
    and still answers correctly for a caller whose references were never
    canonicalised.
    """
    if candidate == reference or canonical_smiles(candidate) == reference:
        return True
    return same_molecule(reference, candidate)


def _decode_batch(codec: BaseCodec, payloads: Sequence[Any]) -> list[str | None]:
    """Decode a batch, using the codec's batch path when it has one."""
    batch = getattr(codec, "decode_many", None)
    if callable(batch):
        return list(batch(payloads))
    return [codec.decode(payload) for payload in payloads]


def _encode_batch(codec: BaseCodec, smiles: Sequence[str]) -> list[Any]:
    """Encode a batch, using the codec's batch path when it has one."""
    batch = getattr(codec, "encode_many", None)
    if callable(batch):
        return list(batch(smiles))
    return [codec.encode(item) for item in smiles]


def run_codec(
    codec: BaseCodec,
    corpus: Corpus,
    *,
    budget: int = BIT_BUDGET,
    batch_size: int = 512,
    max_examples: int = 5,
    collect_ceiling: bool = True,
    strict_budget: bool = False,
) -> CodecResult:
    """Fit, run and score one codec against one corpus.

    :param codec: The codec to evaluate; it is fitted on ``corpus.fit``.
    :param corpus: The prepared corpus.
    :param budget: Bit budget used for the compliance figure.
    :param batch_size: Molecules encoded and decoded per batch.
    :param max_examples: Failing examples retained for the report.
    :param collect_ceiling: Whether to compute the injectivity ceiling, which
        is only meaningful for bitstring payloads.
    :param strict_budget: Treat the budget as a constraint rather than a
        statistic. An oversized payload is then scored as an ``over-budget``
        failure even though it decoded correctly, so the accuracy column reads
        as accuracy *subject to* the budget. Use it when the budget is a
        requirement, and leave it off to keep accuracy and size independent.
    :returns: The populated :class:`CodecResult`.
    """
    silence_rdkit()
    result = CodecResult(
        name=codec.name,
        family=getattr(codec, "family", "other"),
        notes=getattr(codec, "notes", ""),
        lossless_by_design=getattr(codec, "lossless_by_design", True),
    )
    started = time.perf_counter()
    try:
        codec.fit(corpus.fit)
        # A lookup-decoded codec needs its library before it can decode. The
        # closed world includes the evaluation molecules, which makes its score
        # an identification rate rather than a reconstruction rate.
        library = getattr(codec, "populate_library", None)
        if callable(library):
            world = getattr(getattr(codec, "library", None), "world", "closed")
            library(list(corpus.evaluate) if world == "closed" else list(corpus.fit))
        prepare = getattr(codec, "prepare_library", None)
        if callable(prepare):
            prepare(list(corpus.evaluate))
    except Exception as exc:  # noqa: BLE001 - a codec that cannot be fitted is reported
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - started
        return result

    # The injectivity ceiling is defined only for bit-sequence payloads. A
    # string codec would contribute nothing but its encode failures, giving a
    # 0% ceiling that reads as "every molecule collides" when it means "this
    # metric does not apply".
    collect = collect_ceiling and getattr(codec, "kind", "") == "bitstring"
    payloads_for_ceiling: list[Bits | None] = []
    for start in range(0, len(corpus.evaluate), batch_size):
        chunk = list(corpus.evaluate[start : start + batch_size])
        try:
            payloads = _encode_batch(codec, chunk)
            decoded = _decode_batch(codec, payloads)
        except Exception as exc:  # noqa: BLE001 - surfaced rather than aborting the run
            result.error = f"{type(exc).__name__}: {exc}"
            break
        for reference, payload, candidate in zip(chunk, payloads, decoded, strict=True):
            result.molecules += 1
            if collect:
                payloads_for_ceiling.append(payload if isinstance(payload, Bits) else None)
            if payload is None:
                result.failures["no-encoding"] = result.failures.get("no-encoding", 0) + 1
                if len(result.examples) < max_examples:
                    result.examples.append((reference, None, "no-encoding"))
                continue
            size = codec.payload_bits(payload)
            result.bits.append(size)
            if strict_budget and size > budget:
                # Decoding may have succeeded, but an oversized payload is not
                # a solution to the problem as posed.
                result.failures["over-budget"] = result.failures.get("over-budget", 0) + 1
                if len(result.examples) < max_examples:
                    result.examples.append((reference, candidate, "over-budget"))
                continue
            if candidate is not None and _matches(reference, candidate):
                result.successes += 1
                continue
            kind = classify_failure(reference, candidate)
            result.failures[kind] = result.failures.get(kind, 0) + 1
            if len(result.examples) < max_examples:
                result.examples.append((reference, candidate, kind))

    if collect and payloads_for_ceiling:
        result.ceiling = float(injectivity(payloads_for_ceiling)["ceiling"])
    native = getattr(codec, "native_bits", None)
    if isinstance(native, int):
        result.native_bits = native
    result.seconds = time.perf_counter() - started
    return result


@dataclass(slots=True)
class BenchmarkReport:
    """The results of a whole benchmark run."""

    corpus: str
    molecules: int
    fit_molecules: int
    budget: int
    results: list[CodecResult] = field(default_factory=list)

    def passing(self, target: float = 0.9999) -> list[CodecResult]:
        """Return codecs whose evidence supports ``target`` within the budget."""
        return [
            item
            for item in self.results
            if item.supports_target(target) and (item.within_budget(self.budget) or 0.0) >= 1.0
        ]

    def sorted_by_bits(self) -> list[CodecResult]:
        """Return results ordered by median payload size, cheapest first."""
        return sorted(self.results, key=lambda item: (item.median_bits is None, item.median_bits or 0))


def run_benchmark(
    codecs: Sequence[BaseCodec],
    corpus: Corpus,
    *,
    budget: int = BIT_BUDGET,
    batch_size: int = 512,
    progress: bool = False,
    strict_budget: bool = False,
    on_result=None,  # noqa: ANN001 - optional callback
) -> BenchmarkReport:
    """Run every codec against ``corpus`` and collect the results.

    A codec that raises is recorded with its error and the run goes on, so one
    missing optional dependency does not cost the whole table.

    :param codecs: Codecs to evaluate.
    :param corpus: The prepared corpus.
    :param budget: Bit budget for compliance reporting.
    :param batch_size: Molecules per batch.
    :param progress: Reserved for callers that want progress output.
    :param strict_budget: Score over-budget payloads as failures; see
        :func:`run_codec`.
    :param on_result: Optional callable invoked with each finished result.
    :returns: The assembled :class:`BenchmarkReport`.
    """
    report = BenchmarkReport(
        corpus=corpus.name,
        molecules=len(corpus.evaluate),
        fit_molecules=len(corpus.fit),
        budget=budget,
    )
    for codec in codecs:
        result = run_codec(
            codec,
            corpus,
            budget=budget,
            batch_size=batch_size,
            strict_budget=strict_budget,
        )
        report.results.append(result)
        if on_result is not None:
            on_result(result)
    return report


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/benchmark.py",
        description="Benchmark molecular encodings on round-trip fidelity under a bit budget.",
    )
    add_source_arguments(parser)
    parser.add_argument(
        "--fit-source",
        action="append",
        dest="fit_source",
        type=parse_source,
        metavar="FILE[:COLUMN]",
        default=[],
        help="fit shared state on different files, to test transfer across chemical space",
    )
    parser.add_argument(
        "--suite",
        default="fast",
        choices=SUITES,
        help="which codecs to run (default: %(default)s)",
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=50000,
        help=(
            "evaluation molecules; 0 means every available one. The default is large because "
            "certifying 99.99%% at 95%% confidence takes roughly 40000 clean round trips "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument("--fit", type=int, default=50000, help="molecules used to fit shared state")
    parser.add_argument("--budget", type=int, default=1000, help="bit budget (default: %(default)s)")
    parser.add_argument(
        "--target",
        type=float,
        default=0.9999,
        help="accuracy target a codec must clear (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--workers", type=int, default=4, help="processes used to canonicalise input")
    parser.add_argument("--batch-size", type=int, default=512, help="molecules per encode/decode batch")
    parser.add_argument(
        "--world",
        default="open",
        choices=("closed", "open"),
        help=(
            "library world for lookup-decoded codecs. 'open' gives the decoder only the fitting "
            "split, so it has to reconstruct a molecule it has never seen; 'closed' hands it the "
            "evaluation molecules, which measures identification and is reported anyway as the "
            "'ceiling' column (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--strict-budget",
        action="store_true",
        help=(
            "treat the bit budget as a requirement: a payload larger than --budget counts as a "
            "failure even if it decoded correctly, so accuracy reads as 'accuracy subject to the budget'"
        ),
    )
    parser.add_argument("--only", default=None, help="comma-separated codec names to keep")
    parser.add_argument("--csv", type=Path, default=None, help="write per-codec results to this CSV file")
    parser.add_argument("--markdown", type=Path, default=None, help="write a Markdown table here")
    parser.add_argument("--list-codecs", action="store_true", help="list the codecs in the chosen suite")
    parser.add_argument("--quiet", action="store_true", help="print only the final table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a benchmark sweep over the named corpora.

    :param argv: Argument list, defaulting to :data:`sys.argv`.
    :returns: Process exit status.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    silence_rdkit()

    codecs = build_suite(args.suite, world=args.world)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        codecs = [codec for codec in codecs if codec.name in wanted]
        if not codecs:
            print(f"no codec matched {sorted(wanted)}", file=sys.stderr)  # noqa: T201
            return 2

    if args.list_codecs:
        for codec in codecs:
            print(f"{codec.name:24s} {codec.family:12s} {codec.notes}")  # noqa: T201
        return 0

    if not args.source:
        parser.error("at least one --source FILE[:COLUMN] is required")

    if not args.quiet:
        names = ", ".join(source.label for source in args.source)
        print(f"building corpus from {names} ...", file=sys.stderr)  # noqa: T201
    corpus = build_corpus(
        args.source,
        evaluate=args.evaluate or None,
        fit=args.fit,
        seed=args.seed,
        workers=args.workers,
        label=args.label,
        fit_sources=args.fit_source,
    )
    if not corpus.evaluate:
        print(f"{corpus.name}: no usable molecules left to evaluate on", file=sys.stderr)  # noqa: T201
        return 1
    if not args.quiet:
        print(corpus.summary, file=sys.stderr)  # noqa: T201
        print(f"running {len(codecs)} codecs: {', '.join(codec_names(codecs))}", file=sys.stderr)  # noqa: T201

    def announce(result: CodecResult) -> None:
        if args.quiet:
            return
        status = result.error or f"{100.0 * result.accuracy:.4f}%  median {result.median_bits} bits"
        print(f"  {result.name:24s} {status}  [{result.seconds:.1f}s]", file=sys.stderr)  # noqa: T201

    report = run_benchmark(
        codecs,
        corpus,
        budget=args.budget,
        batch_size=args.batch_size,
        strict_budget=args.strict_budget,
        on_result=announce,
    )

    print(render_table(report, target=args.target))  # noqa: T201
    print()  # noqa: T201
    print(render_failures(report))  # noqa: T201

    if args.csv:
        args.csv.write_text(render_csv([report]), encoding="utf-8")
        if not args.quiet:
            print(f"wrote {args.csv}", file=sys.stderr)  # noqa: T201
    if args.markdown:
        args.markdown.write_text(render_markdown(report, target=args.target), encoding="utf-8")
        if not args.quiet:
            print(f"wrote {args.markdown}", file=sys.stderr)  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
