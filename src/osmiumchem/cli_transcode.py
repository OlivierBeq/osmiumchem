# -*- coding: utf-8 -*-

"""The ``encode``, ``decode``, ``roundtrip``, ``fit`` and ``codecs`` commands.

The commands share the codebook argument group, since the thing to know about
them is that a fitted codec is a shared artefact and ``encode`` and ``decode``
have to hold the same one. By default both take the codebook that shipped with
the package for ``--corpus``, so they cannot disagree; ``--fit-on`` fits a new
one from your own molecules and ``--codebook`` names a file explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .api import DEFAULT_CORPUS, fit_codec, list_codebooks, load_codec
from .codebook import Codebook, auto_path, save
from .codecs import catalogue
from .codecs.vectors import BIT_BUDGET
from .formats import INPUT_FORMATS, OUTPUT_FORMATS, PAYLOAD_FORMATS
from .identity import silence_rdkit
from .transcode import decode, encode, round_trip

__all__ = ["COMMANDS", "add_subcommands", "run"]

#: The verbs this module handles. :mod:`osmiumchem.__main__` checks the
#: first argument against this set to decide where to route.
COMMANDS = ("encode", "decode", "roundtrip", "fit", "codecs")


def _add_codebook_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that select and build the fitted codec."""
    group = parser.add_argument_group(
        "codebook",
        "Which fitted codec to use. Entropy codecs are frozen frequency tables, so a payload "
        "means something only to the table that produced it: encode and decode need the same "
        "codebook. These defaults produce the same one every time, and it is cached.",
    )
    group.add_argument("--codec", required=True, help="codec name, e.g. smiles+arith-o4")
    group.add_argument(
        "--codebook",
        type=Path,
        default=None,
        help="codebook file to read, such as one written by `osmiumchem fit --out`",
    )
    group.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help="which shipped codebook to use; 'osmiumchem codecs --corpora' lists them (default: %(default)s)",
    )
    group.add_argument(
        "--fit-on",
        dest="fit_on",
        type=Path,
        default=None,
        help=(
            "file of SMILES, one per line, to fit the codec on instead of using a shipped "
            "codebook. The fitted table is the code, so matching it to your own series costs "
            "fewer bits per molecule"
        ),
    )
    group.add_argument(
        "--world",
        default="closed",
        choices=("closed", "open"),
        help="library world for lookup-decoded codecs (default: %(default)s)",
    )
    group.add_argument("--workers", type=int, default=4, help="processes used to canonicalise --fit-on")
    group.add_argument(
        "--library",
        type=Path,
        default=None,
        help=(
            "file of SMILES, one per line, to add to the lookup library of codecs that decode by "
            "lookup (fingerprints, latents). Forces a refit, as it changes what they can decode"
        ),
    )


def add_subcommands(subparsers: Any) -> None:  # noqa: ANN401 - argparse's _SubParsersAction
    """Register the five commands on a subparser action."""
    molecules_help = (
        "molecules to encode. Accepts SMILES, SELFIES, InChI or DeepSMILES; use '-' to read "
        "them from standard input, one per line"
    )

    codecs = subparsers.add_parser(
        "codecs",
        help="list codec names that encode and decode accepts",
        description="List every codec name, with its family and what it does.",
    )
    codecs.add_argument("--family", default=None, help="show only this family")
    codecs.add_argument(
        "--corpora",
        action="store_true",
        help="list the corpora a codebook ships for instead of the codecs",
    )
    codecs.add_argument("--json", action="store_true", help="emit JSON")

    fit = subparsers.add_parser(
        "fit",
        help="fit a codec on your own molecules and save its codebook",
        description=(
            "Fit a codec's shared state on the molecules in --fit-on and write it to a codebook, "
            "so later encode and decode calls are immediate and provably use identical state. "
            "Pass that file back with --codebook."
        ),
    )
    _add_codebook_arguments(fit)
    fit.add_argument("--out", type=Path, default=None, help="write the codebook here (default: the cache)")
    fit.add_argument("--json", action="store_true", help="emit JSON")

    for verb, summary, description in (
        (
            "encode",
            "encode a molecule into a payload",
            "Encode one or more molecules and report the payload and its exact bit cost.",
        ),
        (
            "decode",
            "decode a payload back into a molecule",
            "Decode payloads produced by `encode` with the same codec and codebook.",
        ),
        (
            "roundtrip",
            "encode then decode, and check the molecule survived",
            (
                "Encode and decode in one step, and report whether the same molecule came back. "
                "The identity test is the one the benchmark scores with, so a molecule that "
                "passes here would have counted there. The recovered SMILES is the canonical "
                "form, often not the string you typed, and that still counts as a success."
            ),
        ),
    ):
        parser = subparsers.add_parser(verb, help=summary, description=description)
        _add_codebook_arguments(parser)
        if verb == "decode":
            parser.add_argument(
                "payloads",
                nargs="*",
                help="payloads as printed by encode; use '-' to read them from standard input",
            )
        else:
            parser.add_argument("molecules", nargs="*", help=molecules_help)
            parser.add_argument(
                "--in",
                dest="input_format",
                default="auto",
                choices=INPUT_FORMATS,
                help="how to read the input (default: %(default)s)",
            )
            parser.add_argument(
                "--payload-format",
                default="auto",
                choices=PAYLOAD_FORMATS,
                help="textual form for the payload (default: %(default)s)",
            )
        if verb != "encode":
            parser.add_argument(
                "--out",
                dest="output_format",
                default="smiles",
                choices=OUTPUT_FORMATS,
                help="representation for the recovered molecule (default: %(default)s)",
            )
        parser.add_argument(
            "--budget",
            type=int,
            default=BIT_BUDGET,
            help="bit budget the payload is checked against (default: %(default)s)",
        )
        parser.add_argument("--json", action="store_true", help="emit JSON, one object per molecule")
        parser.add_argument("--quiet", action="store_true", help="print only the result lines")


def _items(given: list[str]) -> list[str]:
    """Return the positional arguments, expanding ``-`` to lines of stdin."""
    collected: list[str] = []
    for item in given:
        if item == "-":
            collected.extend(line.strip() for line in sys.stdin if line.strip())
        else:
            collected.append(item)
    return collected


def _molecules_in(path: Path | None) -> list[str] | None:
    """Read molecules from a file, one per line."""
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _open(args: argparse.Namespace) -> tuple[Codebook, Path | None, bool]:
    """Load or fit the codebook the arguments describe.

    Three ways in, checked in this order: an explicit ``--codebook`` file,
    molecules to fit on from ``--fit-on``, or the codebook that shipped for
    ``--corpus``. Delegated to :mod:`osmiumchem.api`, so the command line and
    the Python interface cannot disagree about what a given set of settings
    means.
    """
    library = _molecules_in(args.library)
    fitting = _molecules_in(args.fit_on)
    if args.codebook is not None and fitting is not None:
        msg = "--codebook and --fit-on both say where the codec comes from; pass one"
        raise ValueError(msg)
    if library is not None and fitting is None:
        # Extra library molecules are part of the fitted state, so they can only
        # go in while it is being fitted. A codebook that already exists has its
        # library baked in, and silently dropping them would change what decodes.
        msg = "--library needs --fit-on: an existing codebook already has its library"
        raise ValueError(msg)
    if args.codebook is not None:
        transcoder = load_codec(args.codebook)
        return transcoder.codebook, transcoder.path, False
    if fitting is not None:
        transcoder = fit_codec(
            args.codec,
            fitting,
            label=args.fit_on.stem,
            world=args.world,
            workers=args.workers,
            library=library,
        )
        return transcoder.codebook, transcoder.path, True
    transcoder = load_codec(args.codec, corpus=args.corpus)
    return transcoder.codebook, transcoder.path, False


def _report_codebook(book: Codebook, path: Path | None, fitted: bool, *, quiet: bool) -> None:
    """Tell the user which codebook is in use, on stderr."""
    if quiet:
        return
    verb = "fitted" if fitted else "loaded"
    where = f" -> {path}" if path is not None else " (not cached)"
    print(f"{verb} codebook: {book.provenance.summary()}{where}", file=sys.stderr)  # noqa: T201


def _emit(payload: dict[str, Any], *, as_json: bool, line: str) -> None:
    """Print either a JSON object or a human-readable line."""
    print(json.dumps(payload, default=str) if as_json else line)  # noqa: T201


def _codecs(args: argparse.Namespace) -> int:
    """Run the ``codecs`` command."""
    if args.corpora:
        corpora = list_codebooks()
        if args.json:
            print(json.dumps(corpora, indent=2))  # noqa: T201
            return 0
        if not corpora:
            print("no codebooks ship with this install", file=sys.stderr)  # noqa: T201
            return 2
        for label in corpora:
            print(label)  # noqa: T201
        return 0
    codecs = [
        codec for codec in catalogue().values() if args.family is None or getattr(codec, "family", "") == args.family
    ]
    if not codecs:
        print(f"no codec in family {args.family!r}", file=sys.stderr)  # noqa: T201
        return 2
    if args.json:
        print(json.dumps([codec.describe() for codec in codecs], indent=2))  # noqa: T201
        return 0
    width = max(len(codec.name) for codec in codecs)
    for codec in sorted(codecs, key=lambda item: (item.family, item.name)):
        print(f"{codec.name:{width}s}  {codec.family:11s}  {codec.notes}")  # noqa: T201
    return 0


def _fit(args: argparse.Namespace) -> int:
    """Run the ``fit`` command."""
    book, path, fitted = _open(args)
    if args.out is not None:
        path = save(book, args.out)
    elif fitted:
        # Fitted in memory and nowhere to put it, so the cache is the default:
        # a codebook nobody can read again cannot decode these payloads.
        path = save(book, auto_path(book.provenance))
    if args.json:
        print(json.dumps({"codebook": str(path), "fitted_now": fitted, **_provenance(book)}, indent=2))  # noqa: T201
    else:
        _report_codebook(book, path, fitted, quiet=False)
        print(path if path is not None else "(not cached)")  # noqa: T201
    return 0


def _provenance(book: Codebook) -> dict[str, Any]:
    """Return the provenance fields worth putting in JSON output."""
    provenance = book.provenance
    return {
        "codec": provenance.codec,
        "corpus": provenance.corpus,
        "fit_requested": provenance.molecules,
        "fit_actual": provenance.fitted,
        "seed": provenance.seed,
        "world": provenance.world,
        "library": provenance.library,
    }


def _encode(args: argparse.Namespace) -> int:
    """Run the ``encode`` command."""
    book, path, fitted = _open(args)
    _report_codebook(book, path, fitted, quiet=args.quiet or args.json)
    molecules = _items(args.molecules)
    if not molecules:
        print("no molecules given; pass them as arguments or use '-' for stdin", file=sys.stderr)  # noqa: T201
        return 2
    failures = 0
    for molecule in molecules:
        result = encode(
            book.codec,
            molecule,
            input_format=args.input_format,
            payload_format=args.payload_format,
            budget=args.budget,
        )
        if not result.ok:
            failures += 1
            _emit(
                {"ok": False, "input": molecule, "reason": result.reason},
                as_json=args.json,
                line=f"! {molecule}  {result.reason}",
            )
            continue
        fits = "" if result.within_budget else f"  OVER BUDGET (>{args.budget})"
        _emit(
            {
                "ok": True,
                "input": molecule,
                "input_format": result.input_format,
                "canonical": result.canonical,
                "bits": result.bits,
                "within_budget": result.within_budget,
                "form": result.form,
                "payload": result.text,
            },
            as_json=args.json,
            line=f"{result.text}\t{result.bits} bits{fits}",
        )
    return 1 if failures else 0


def _decode(args: argparse.Namespace) -> int:
    """Run the ``decode`` command."""
    book, path, fitted = _open(args)
    _report_codebook(book, path, fitted, quiet=args.quiet or args.json)
    payloads = _items(args.payloads)
    if not payloads:
        print("no payloads given; pass them as arguments or use '-' for stdin", file=sys.stderr)  # noqa: T201
        return 2
    failures = 0
    for text in payloads:
        result = decode(book.codec, text, output_format=args.output_format)
        if not result.ok:
            failures += 1
            _emit(
                {"ok": False, "payload": text, "reason": result.reason},
                as_json=args.json,
                line=f"! {result.reason}",
            )
            continue
        _emit(
            {
                "ok": True,
                "payload": text,
                "bits": result.bits,
                "smiles": result.smiles,
                "output_format": result.output_format,
                "molecule": result.rendered,
            },
            as_json=args.json,
            line=f"{result.rendered}\t{result.bits} bits",
        )
    return 1 if failures else 0


def _roundtrip(args: argparse.Namespace) -> int:
    """Run the ``roundtrip`` command."""
    book, path, fitted = _open(args)
    _report_codebook(book, path, fitted, quiet=args.quiet or args.json)
    molecules = _items(args.molecules)
    if not molecules:
        print("no molecules given; pass them as arguments or use '-' for stdin", file=sys.stderr)  # noqa: T201
        return 2
    failures = 0
    for molecule in molecules:
        result = round_trip(
            book.codec,
            molecule,
            input_format=args.input_format,
            output_format=args.output_format,
            payload_format=args.payload_format,
            budget=args.budget,
        )
        recovered = result.decoded.rendered if result.decoded is not None else None
        mark = {"ok": "OK  ", "over-budget": "OVER", "failed": "FAIL"}[result.verdict]
        note = f"  {result.reason}" if result.reason else ""
        _emit(
            {
                "ok": result.ok,
                "verdict": result.verdict,
                "input": molecule,
                "input_format": result.encoded.input_format,
                "canonical": result.encoded.canonical,
                "bits": result.bits,
                "within_budget": result.within_budget,
                "payload": result.encoded.text,
                "recovered": recovered,
                "identical": result.identical,
                "reason": result.reason,
            },
            as_json=args.json,
            line=f"{mark}  {result.bits if result.bits is not None else '-'} bits  {recovered or '-'}{note}",
        )
        if result.verdict != "ok":
            failures += 1
    return 1 if failures else 0


def run(argv: list[str]) -> int:
    """Parse and run one of :data:`COMMANDS`.

    :param argv: Argument list starting with the verb.
    :returns: Process exit status. ``1`` means at least one molecule or payload
        failed, ``2`` a usage error.
    """
    parser = argparse.ArgumentParser(
        prog="osmiumchem",
        description="Encode and decode individual molecules with a named codec.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subcommands(subparsers)
    args = parser.parse_args(argv)
    silence_rdkit()

    handler = {
        "codecs": _codecs,
        "fit": _fit,
        "encode": _encode,
        "decode": _decode,
        "roundtrip": _roundtrip,
    }[args.command]
    try:
        return handler(args)
    except KeyError as exc:
        # An unknown --codec or --corpus. Both carry a message naming the
        # valid spellings, which beats a traceback.
        print(exc.args[0] if exc.args else str(exc), file=sys.stderr)  # noqa: T201
        return 2
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
        return 2
