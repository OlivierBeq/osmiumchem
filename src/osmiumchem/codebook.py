# -*- coding: utf-8 -*-

"""Fitted codec state, saved so that encode and decode can be separate calls.

Most of the interesting codecs are fitted: an order-4 arithmetic coder is a
frozen frequency table, and its payloads mean nothing to any other table. The
benchmark never notices, since one process fits the codec and round-trips
every molecule. At a shell prompt it is the whole problem, because ``encode``
and ``decode`` are two commands in two processes that have to agree bit for
bit.

A codebook is the shared artefact they agree on: the fitted codec plus enough
provenance to tell whether two of them are the same one. This is what
:mod:`osmiumchem.coding.model` means by state shipped alongside the decoder.
Only the per-molecule payload counts against the bit budget, the same
accounting a trained autoencoder's weights get.

Codebooks come from two places. A set fitted on each benchmark corpus ships
inside the package, under :func:`bundled_root`, which is what makes encoding
work on a fresh install with no corpus on disk. Anything you fit yourself goes
to :func:`cache_root`, keyed by provenance, or wherever
:func:`~osmiumchem.api.save_codec` puts it.

.. warning::

   Codebooks are Python pickles, so loading one runs the code inside it. The
   shipped ones are part of the package, so they are already as trusted as the
   code that reads them; anything else is not. Load only codebooks you wrote
   yourself or that came with an install you trust. :func:`load` checks a
   magic number and a container version first, which catches an unrelated or
   stale file, but no such check makes an untrusted pickle safe.
"""

from __future__ import annotations

import hashlib
import lzma
import os
import pickle
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .codecs import resolve_codec
from .identity import canonicalise_all
from .version import __version__

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .codecs.base import BaseCodec

__all__ = [
    "Codebook",
    "Provenance",
    "SUFFIX",
    "auto_path",
    "bundled_corpora",
    "bundled_path",
    "bundled_root",
    "cache_root",
    "fit_codebook_on",
    "load",
    "save",
]

#: File magic, so an unrelated file is rejected before anything is unpickled.
MAGIC = "osmiumchem/codebook"

#: Extension every codebook is written with. ``.xz`` because the container is
#: lzma, ``.oc`` so a stray codebook is recognisable.
SUFFIX = ".oc.xz"

#: Container version. Bumped whenever the stored layout or the fitted codec
#: classes change in a way that makes an older file decode differently. This is
#: the check that matters: a codebook pickles its own parameters, so the
#: package version it was written by is recorded but not enforced, which is
#: what lets the shipped set survive a release.
FORMAT = 2


@dataclass(frozen=True, slots=True)
class Provenance:
    """What a codebook was fitted on, and by what.

    Two codebooks with equal provenance hold the same fitted state, which is
    what makes the cache sound. Corpus sampling and shuffling are both seeded,
    so ``(corpus, molecules, seed)`` reproduces the fitting split from the same
    source files on any machine.

    :param codec: Codec name.
    :param corpus: Label of the corpus the fitting split came from. A name
        only: no path, since the corpora live outside the package.
    :param molecules: Number of molecules requested for fitting.
    :param seed: Sampling seed.
    :param world: Library world for lookup-decoded codecs.
    :param fitted: Molecules the codec was fitted on, which can fall short of
        ``molecules`` if the corpus is smaller or some inputs did not parse.
    :param library: Molecules placed in the lookup library, where one applies.
    :param library_digest: Fingerprint of the *extra* library molecules, if
        any. Part of :attr:`key`, because extra molecules change what a
        lookup-decoded codec can decode. Without it, a codebook built with
        ``--library`` would be reused by a later call that never asked for
        those molecules.
    :param version: Version of this package that wrote the codebook.
    :param created: Unix timestamp.
    """

    codec: str
    corpus: str
    molecules: int
    seed: int
    world: str = "closed"
    fitted: int = 0
    library: int = 0
    library_digest: str = ""
    version: str = __version__
    created: float = 0.0

    @property
    def key(self) -> tuple[str, str, int, int, str, str]:
        """Return the fields that determine the fitted state."""
        return (
            self.codec,
            self.corpus,
            self.molecules,
            self.seed,
            self.world,
            self.library_digest,
        )

    def summary(self) -> str:
        """Return a one-line human-readable description."""
        extra = f", library {self.library}" if self.library else ""
        return (
            f"{self.codec} fitted on {self.fitted} molecules from {self.corpus} "
            f"(requested {self.molecules}, seed {self.seed}{extra})"
        )


@dataclass(frozen=True, slots=True)
class Codebook:
    """A fitted codec together with its provenance."""

    codec: BaseCodec
    provenance: Provenance

    @property
    def name(self) -> str:
        """Return the codec's name."""
        return self.codec.name


def cache_root() -> Path:
    """Return the directory holding cached codebooks.

    ``OSMIUMCHEM_CACHE`` redirects it. Codebooks sit in their own
    subdirectory, so clearing one cache leaves anything else alone.
    """
    override = os.environ.get("OSMIUMCHEM_CACHE")
    base = Path(override) if override else Path.home() / ".cache" / "osmiumchem"
    return base / "codebooks"


def bundled_root() -> Path:
    """Return the package directory holding the codebooks that ship with it."""
    return Path(__file__).resolve().parent / "codebooks"


def _safe(stem: str) -> str:
    """Return ``stem`` with anything awkward in a filename replaced."""
    return "".join(character if character.isalnum() or character in "-_+." else "_" for character in stem)


def bundled_path(codec: str, corpus: str) -> Path:
    """Return the path of the shipped codebook for ``codec`` on ``corpus``.

    Shipped files are named by codec and corpus alone, unlike the cache, which
    also keys on the fit size and seed. One codebook per pair is all that
    ships, and its own provenance records how it was fitted.
    """
    return bundled_root() / f"{_safe(f'{codec}__{corpus}')}{SUFFIX}"


def bundled_corpora(codec: str | None = None) -> list[str]:
    """Return the corpus labels a shipped codebook exists for.

    :param codec: Restrict to one codec; ``None`` lists every label seen.
    """
    root = bundled_root()
    if not root.is_dir():
        return []
    labels: set[str] = set()
    for path in root.glob(f"*{SUFFIX}"):
        stem = path.name.removesuffix(SUFFIX)
        name, separator, corpus = stem.rpartition("__")
        if not separator:
            continue
        if codec is None or _safe(codec) == name:
            labels.add(corpus)
    return sorted(labels)


def library_digest(molecules: Sequence[str] | None) -> str:
    """Return a short, order-insensitive fingerprint of extra library molecules.

    Empty when there are no extras, which keeps the usual cache filename
    readable.
    """
    if not molecules:
        return ""
    joined = "\n".join(sorted(set(molecules))).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:12]


def auto_path(provenance: Provenance) -> Path:
    """Return the cache path a codebook with this provenance is stored at."""
    codec, corpus, molecules, seed, world, digest = provenance.key
    stem = f"{codec}__{corpus}__n{molecules}__s{seed}__{world}"
    if digest:
        stem += f"__lib{digest}"
    return cache_root() / f"{_safe(stem)}{SUFFIX}"


def fit_codebook_on(
    name: str,
    molecules: Sequence[str],
    *,
    corpus: str = "(given)",
    requested: int | None = None,
    seed: int = 0,
    world: str = "closed",
    workers: int = 1,
    library: Sequence[str] | None = None,
    canonical: bool = False,
) -> Codebook:
    """Fit a codec by name on molecules you already hold.

    The only way to fit one: the package takes molecules, never a corpus to go
    and find. A codec fitted on the chemistry it will encode produces shorter
    payloads than one fitted on a generic split, so this is usually what you
    want anyway.

    :param name: Codec name.
    :param molecules: The fitting corpus.
    :param corpus: Label recorded in the provenance and part of the cache key.
        Different corpora need distinguishable labels, or the cache cannot tell
        two codebooks apart.
    :param requested: Molecules asked for, when that differs from the number
        usable. Defaults to the length of ``molecules``.
    :param seed: Sampling seed, recorded in the provenance.
    :param world: Library world for lookup-decoded codecs.
    :param workers: Processes used for canonicalisation.
    :param library: Extra molecules for the lookup library.
    :param canonical: Set when ``molecules`` are already canonical SMILES, to
        skip a second pass through RDKit.
    :raises ValueError: If no molecule in ``molecules`` is usable.
    """
    codec = resolve_codec(name, world=world)
    fitting = list(molecules) if canonical else canonicalise_all(molecules, workers=workers)[0]
    if not fitting:
        msg = f"no molecules to fit {name!r} on"
        raise ValueError(msg)
    codec.fit(fitting)

    # A lookup-decoded codec decodes nothing until its library is populated.
    # Filling it from the fitting split gives the open world: the decoder holds
    # the molecules it learned from and nothing else, so a payload for an unseen
    # molecule has nothing to resolve to. ``library`` widens that explicitly.
    #
    # The extras are canonicalised first.
    # :meth:`~osmiumchem.codecs.vectors.LibraryDecoder.add` tells entries apart
    # by their SMILES strings, so a non-canonical entry would land on the same
    # payload as its canonical twin, count as a different molecule, and mark
    # that payload ambiguous. Adding a molecule to the library would then be
    # what stopped it decoding.
    extra, _ = canonicalise_all(library or [], workers=workers)
    shelved = fitting + extra
    populate = getattr(codec, "populate_library", None)
    if callable(populate):
        populate(shelved)
    prepare = getattr(codec, "prepare_library", None)
    if callable(prepare):
        prepare(shelved)
    held = len(shelved) if (callable(populate) or callable(prepare)) else 0

    provenance = Provenance(
        codec=codec.name,
        corpus=corpus,
        molecules=len(fitting) if requested is None else requested,
        seed=seed,
        world=world,
        fitted=len(fitting),
        library=held,
        library_digest=library_digest(library),
        created=time.time(),
    )
    return Codebook(codec=codec, provenance=provenance)


def save(book: Codebook, path: Path) -> Path:
    """Write a codebook to ``path``, creating parent directories.

    The header is a separate pickle ahead of the codec, so :func:`load` can
    check what it holds before unpickling the heavy part.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {
        "magic": MAGIC,
        "format": FORMAT,
        "provenance": asdict(book.provenance),
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    # preset 9 costs a tenth of a second on the largest table and takes about
    # a fifth off what gzip manages; decompression is the cheaper direction.
    with lzma.open(temporary, "wb", preset=9) as handle:
        pickle.dump(header, handle, protocol=5)
        pickle.dump(book.codec, handle, protocol=5)
    temporary.replace(path)
    return path


def load(path: Path) -> Codebook:
    """Read a codebook written by :func:`save`.

    The package version that wrote the file is recorded but not enforced,
    since a codebook pickles its own parameters and a release does not change
    them. :data:`FORMAT` is the gate, and it is bumped when they do change.

    :param path: Codebook file.
    :raises ValueError: If the file is not a codebook, or uses a container
        format this build cannot read.
    """
    path = Path(path)
    with lzma.open(path, "rb") as handle:
        try:
            header = pickle.load(handle)  # noqa: S301 - see the module warning
        except Exception as exc:  # noqa: BLE001 - any failure means "not a codebook"
            msg = f"{path} is not an osmiumchem codebook ({type(exc).__name__})"
            raise ValueError(msg) from exc
        if not isinstance(header, dict) or header.get("magic") != MAGIC:
            msg = f"{path} is not an osmiumchem codebook"
            raise ValueError(msg)
        if header.get("format") != FORMAT:
            msg = f"{path} uses codebook format {header.get('format')!r}, this build reads {FORMAT}; refit it"
            raise ValueError(msg)
        stored = header.get("provenance") or {}
        codec = pickle.load(handle)  # noqa: S301 - see the module warning
    return Codebook(codec=codec, provenance=Provenance(**stored))
