# -*- coding: utf-8 -*-

"""Lossless molecular encodings under a hard bit budget.

Fit a codec, encode a molecule, decode it back. Codebooks fitted on each
benchmark corpus ship with the package, so the first call needs nothing on
disk::

    import osmiumchem as oc

    codec = oc.load_codec("smiles+arith-o4")
    result = codec.round_trip("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    result.verdict, result.bits, result.decoded.smiles
    # ('ok', 55, 'Cn1c(=O)c2c(ncn2C)n(C)c1=O')

:func:`list_codecs` names every codec, :func:`list_codebooks` the corpora one
ships for, and :mod:`osmiumchem.api` documents the rest. Reach for
:func:`fit_codec` when you have your own chemistry, which pays off: the fitted
table *is* the code, so matching it to your series costs fewer bits per
molecule. :func:`save_codec` keeps the result.

The benchmark that produced ``docs/RESULTS.md`` is not here. It reads corpora
from files, which the package deliberately knows nothing about, so it lives in
``scripts/benchmark.py`` in the repository.

Names are bound lazily, by :pep:`562`. RDKit, NumPy and the codecs take about
two thirds of a second to import, wasted on a program that wants one symbol
from this package, so the cost falls on first use of a name that needs it.
``import osmiumchem`` itself stays immediate.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .version import __version__

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .api import (
        Transcoder,
        describe_codecs,
        fit_codec,
        list_codebooks,
        list_codecs,
        load_codec,
        save_codec,
    )
    from .codebook import Codebook, Provenance, fit_codebook_on
    from .codecs import catalogue, resolve_codec
    from .formats import (
        INPUT_FORMATS,
        OUTPUT_FORMATS,
        PAYLOAD_FORMATS,
        detect_format,
        parse_molecule,
        render_molecule,
    )
    from .identity import canonical_smiles, canonicalise_all, same_molecule
    from .transcode import (
        Decoded,
        Encoded,
        RoundTripped,
        decode,
        decode_many,
        encode,
        encode_many,
        round_trip,
        round_trip_many,
    )
    from .types import Bits, Codec, CodecKind, RoundTrip

#: Where each public name lives, for :func:`__getattr__`.
_EXPORTS: dict[str, str] = {
    # A codec, fitted or shipped.
    "Transcoder": "api",
    "describe_codecs": "api",
    "fit_codec": "api",
    "list_codebooks": "api",
    "list_codecs": "api",
    "load_codec": "api",
    "save_codec": "api",
    # The steps underneath, taking a codec rather than a name.
    "Decoded": "transcode",
    "Encoded": "transcode",
    "RoundTripped": "transcode",
    "decode": "transcode",
    "decode_many": "transcode",
    "encode": "transcode",
    "encode_many": "transcode",
    "round_trip": "transcode",
    "round_trip_many": "transcode",
    # Fitted state, saved and reloaded.
    "Codebook": "codebook",
    "Provenance": "codebook",
    "fit_codebook_on": "codebook",
    # The registry.
    "catalogue": "codecs",
    "resolve_codec": "codecs",
    # Representations in and out.
    "INPUT_FORMATS": "formats",
    "OUTPUT_FORMATS": "formats",
    "PAYLOAD_FORMATS": "formats",
    "detect_format": "formats",
    "parse_molecule": "formats",
    "render_molecule": "formats",
    # Molecular identity, which is what a round trip is judged on.
    "canonical_smiles": "identity",
    "canonicalise_all": "identity",
    "same_molecule": "identity",
    # Shared types.
    "Bits": "types",
    "Codec": "types",
    "CodecKind": "types",
    "RoundTrip": "types",
}

#: Submodules, resolved on attribute access, so ``osmiumchem.codebook`` works
#: without a second import statement.
_SUBMODULES = (
    "api",
    "bits",
    "codebook",
    "codecs",
    "coding",
    "formats",
    "identity",
    "transcode",
    "types",
)

#: Spelled out rather than derived from :data:`_EXPORTS`, so static analysis
#: and editors can see the names. A test keeps the two in step.
__all__ = [
    "Bits",
    "Codebook",
    "Codec",
    "CodecKind",
    "Decoded",
    "Encoded",
    "INPUT_FORMATS",
    "OUTPUT_FORMATS",
    "PAYLOAD_FORMATS",
    "Provenance",
    "RoundTrip",
    "RoundTripped",
    "Transcoder",
    "__version__",
    "canonical_smiles",
    "canonicalise_all",
    "catalogue",
    "decode",
    "decode_many",
    "describe_codecs",
    "detect_format",
    "encode",
    "encode_many",
    "fit_codebook_on",
    "fit_codec",
    "list_codebooks",
    "list_codecs",
    "load_codec",
    "parse_molecule",
    "render_molecule",
    "resolve_codec",
    "round_trip",
    "round_trip_many",
    "same_molecule",
    "save_codec",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - the exports are of many types
    """Import a public name, or a submodule, on first use."""
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    where = _EXPORTS.get(name)
    if where is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(f".{where}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public names, so tab completion finds the lazy ones."""
    return sorted({*__all__, *_SUBMODULES})
