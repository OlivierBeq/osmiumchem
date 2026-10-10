# -*- coding: utf-8 -*-

r"""Fit the codebooks that ship inside the package.

An order-4 arithmetic coder *is* its frequency table, so a fresh install can
encode nothing until one is fitted. Fitting needs a corpus, and the package
carries none, so the fitted tables themselves are what ships.

One run covers one corpus. Point it at the files that corpus is made of and
give the label to record::

    python scripts/make_codebooks.py --label protacs \
        --source ~/corpora/protacs_chembl37.csv:canonical_smiles

    python scripts/make_codebooks.py --label union --fit 100000 \
        --source ~/corpora/union_molecules.csv:smiles

Only the ``string`` and ``entropy`` codecs are shipped. A fingerprint or latent
codec has no inverse and decodes by looking a payload up in a *library of
molecules*, so its fitted state is a corpus; shipping that would ship the
data. Those are fitted at use instead, from ``--fit-on`` or ``fit_codec``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from corpora import add_source_arguments, build_corpus

from osmiumchem.api import describe_codecs
from osmiumchem.codebook import bundled_path, bundled_root, fit_codebook_on, save

#: Families whose fitted state is counts and alphabets rather than molecules.
SHIPPED_FAMILIES = ("string", "entropy")

#: Fitting molecules to draw by default. The benchmark's own largest fit, and
#: well past the point where more molecules change the payload: the measured
#: difference between a 20k and a 100k fit is about a bit.
DEFAULT_FIT = 100000


def shippable() -> list[str]:
    """Return the codecs a codebook is shipped for."""
    return [record["name"] for record in describe_codecs() if record["family"] in SHIPPED_FAMILIES]


def main(argv: list[str] | None = None) -> int:
    """Fit and write every shipped codebook for one corpus."""
    parser = argparse.ArgumentParser(
        prog="python scripts/make_codebooks.py",
        description="Fit the codebooks that ship inside the package, for one corpus.",
    )
    add_source_arguments(parser)
    parser.add_argument("--fit", type=int, default=DEFAULT_FIT, help="molecules to fit on (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=8, help="processes used to canonicalise")
    parser.add_argument("--only", default=None, help="comma-separated codec names, instead of every shipped one")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write into (default: the package's own codebooks directory)",
    )
    args = parser.parse_args(argv)
    if not args.source:
        parser.error("at least one --source FILE[:COLUMN] is required")
    if not args.label:
        parser.error("--label is required: it is the corpus name load_codec() will ask for")

    names = shippable()
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        unknown = wanted - set(names)
        if unknown:
            parser.error(f"not shippable: {', '.join(sorted(unknown))}")
        names = [name for name in names if name in wanted]

    print(f"building corpus from {', '.join(s.label for s in args.source)} ...", file=sys.stderr)  # noqa: T201
    corpus = build_corpus(
        args.source,
        evaluate=0,
        fit=args.fit,
        seed=args.seed,
        workers=args.workers,
        label=args.label,
    )
    if not corpus.fit:
        print(f"{corpus.name}: no usable molecules to fit on", file=sys.stderr)  # noqa: T201
        return 1
    print(f"{len(corpus.fit):,} fitting molecules, {len(corpus.rejected):,} unparseable", file=sys.stderr)  # noqa: T201

    total = 0
    for name in names:
        book = fit_codebook_on(
            name,
            corpus.fit,
            corpus=args.label,
            requested=args.fit,
            seed=args.seed,
            canonical=True,
        )
        _forget_cache(book.codec)
        target = bundled_path(name, args.label) if args.out is None else args.out / bundled_path(name, args.label).name
        written = save(book, target)
        size = written.stat().st_size
        total += size
        print(f"{name:30s} {size / 1024:8.0f} KiB  {written.name}")  # noqa: T201

    root = bundled_root() if args.out is None else args.out
    shipped = sum(path.stat().st_size for path in root.glob("*.oc.xz"))
    print(  # noqa: T201
        f"\n{len(names)} codebooks, {total / 1024:,.0f} KiB written; {shipped / 1024 / 1024:.1f} MiB shipped in total",
    )
    return 0


def _forget_cache(codec: object) -> None:
    """Drop any memoised state before the codec is pickled.

    :class:`~osmiumchem.coding.model.OrderKModel` memoises cumulative
    frequencies per context as it codes, so a model that has been used pickles
    larger than one straight out of the fit. Nothing here has coded a molecule
    yet, but the fit is cheap to protect against a future one.
    """
    for attribute in ("model", "_model"):
        model = getattr(codec, attribute, None)
        cache = getattr(model, "_cache", None)
        if isinstance(cache, dict):
            cache.clear()


if __name__ == "__main__":
    sys.exit(main())
