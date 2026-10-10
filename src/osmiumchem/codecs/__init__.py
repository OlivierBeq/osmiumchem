# -*- coding: utf-8 -*-

"""The codec registry.

Codecs are grouped into suites, so a run can be scoped to whatever is being
investigated: ``lossless`` for the codecs that can reach a 99.99% target,
``lossy`` for fingerprints and latent spaces, ``neural`` for the ones that need
model weights, ``all`` for everything.

:func:`resolve_codec` handles one codec at a time, turning a name such as
``"smiles+arith-o4"`` into an instance. It builds the names from the same
factories the suites use rather than parsing them, so the names that work here
are the names in the results tables and there is no second grammar to drift
out of step.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence

from .base import BaseCodec, StringCodec
from .entropy import entropy_codecs
from .graph import graph_codecs
from .latents import CdddNativeCodec, LibraryIndexCodec, latent_codecs
from .strings import string_codecs
from .vectors import FingerprintCodec, LatentCodec, fingerprint_codecs

__all__ = [
    "BaseCodec",
    "CdddNativeCodec",
    "FingerprintCodec",
    "LatentCodec",
    "LibraryIndexCodec",
    "StringCodec",
    "SUITES",
    "build_suite",
    "catalogue",
    "codec_names",
    "resolve_codec",
    "entropy_codecs",
    "fingerprint_codecs",
    "graph_codecs",
    "latent_codecs",
    "string_codecs",
]

#: Suite names accepted by :func:`build_suite`.
SUITES = ("lossless", "lossy", "neural", "fast", "all")


def build_suite(name: str = "fast", *, world: str = "closed") -> list[BaseCodec]:
    """Return the codecs belonging to a suite.

    :param name: One of :data:`SUITES`.
    :param world: Library world for lookup-decoded codecs; see
        :class:`~osmiumchem.codecs.vectors.LibraryDecoder`.
    :returns: Freshly constructed codec instances.
    """
    if name == "lossless":
        return [*string_codecs(), *entropy_codecs(), *graph_codecs()]
    if name == "lossy":
        return [*fingerprint_codecs(world=world), LibraryIndexCodec()]
    if name == "neural":
        return list(latent_codecs(world=world))
    if name == "fast":
        # Neither model weights nor a long run.
        return [
            *string_codecs(),
            *entropy_codecs(orders=(0, 2, 3)),
            *graph_codecs(),
            *fingerprint_codecs(world=world),
            LibraryIndexCodec(),
        ]
    if name == "all":
        return [
            *string_codecs(),
            *entropy_codecs(),
            *graph_codecs(),
            *fingerprint_codecs(world=world),
            LibraryIndexCodec(),
            *latent_codecs(world=world),
        ]
    known = ", ".join(SUITES)
    msg = f"unknown suite {name!r}; choose one of {known}"
    raise ValueError(msg)


def codec_names(codecs: Sequence[BaseCodec]) -> list[str]:
    """Return the names of ``codecs``, for logging and selection."""
    return [codec.name for codec in codecs]


def catalogue(*, world: str = "closed") -> dict[str, BaseCodec]:
    """Return every codec the benchmark knows, keyed by name.

    Construction is cheap in every family. The neural encoders import JAX and
    fetch their weights on first :meth:`fit` rather than in ``__init__``, so the
    full catalogue costs nothing to build even without those packages
    installed.

    :param world: Library world for lookup-decoded codecs; see
        :class:`~osmiumchem.codecs.vectors.LibraryDecoder`.
    """
    return {codec.name: codec for codec in build_suite("all", world=world)}


def resolve_codec(name: str, *, world: str = "closed") -> BaseCodec:
    """Return a fresh, unfitted codec by name.

    :param name: A codec name as it appears in the results tables, for example
        ``"smiles+arith-o4"``, ``"selfies-hypervalent+huffman"``,
        ``"atompair-512-chiral"`` or ``"graph-bits"``.
    :param world: Library world for lookup-decoded codecs.
    :raises KeyError: If no codec has that name. The names are terse and easy
        to mistype, so the message lists the closest spellings.
    """
    codecs = catalogue(world=world)
    codec = codecs.get(name)
    if codec is not None:
        return codec
    close = difflib.get_close_matches(name, sorted(codecs), n=5, cutoff=0.4)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    msg = f"unknown codec {name!r}.{hint} All {len(codecs)} names: osmiumchem.list_codecs(), or `osmiumchem codecs`."
    raise KeyError(msg)
