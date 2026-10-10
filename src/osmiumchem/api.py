# -*- coding: utf-8 -*-

"""The Python interface: fit a codec, encode molecules, decode them back.

Three verbs and nothing else. :func:`load_codec` takes a codec that came
fitted with the package, :func:`fit_codec` fits one on molecules you pass, and
:func:`save_codec` keeps that for later::

    import osmiumchem as oc

    codec = oc.load_codec("smiles+arith-o4")
    codec.round_trip("CN1C=NC2=C1C(=O)N(C)C(=O)N2C").bits   # 55

    mine = oc.fit_codec("smiles+arith-o4", my_series)
    oc.save_codec(mine, "series.oc.xz")
    again = oc.load_codec("series.oc.xz")

:class:`Transcoder` holds the fitted codec (the *codebook*) and the defaults
that would otherwise be repeated at every call: the representation molecules
come in as, the one they go out as, and the bit budget. Holding on to one
matters because a fitted codec is shared state. An order-4 arithmetic coder is
a frozen frequency table, and its payloads mean nothing to any other table, so
whatever one transcoder encodes, that same transcoder decodes.

Nothing here reads a corpus from disk. The codebooks shipped with the package
were fitted on the benchmark corpora and cover every string and entropy codec,
so :func:`load_codec` needs no data; fitting on your own chemistry, which
produces shorter payloads, needs only the molecules themselves.

A molecule that cannot be handled comes back as a result with a ``reason``
rather than an exception, so a loop over a few thousand molecules needs no try
block. Exceptions are kept for misuse, such as an unknown codec name or an
unsupported format.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .codebook import (
    SUFFIX,
    Codebook,
    auto_path,
    bundled_corpora,
    bundled_path,
    fit_codebook_on,
    library_digest,
    load,
    save,
)
from .codecs import catalogue, resolve_codec
from .codecs.vectors import BIT_BUDGET
from .formats import INPUT_FORMATS, OUTPUT_FORMATS, PAYLOAD_FORMATS
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .codecs.base import BaseCodec

__all__ = [
    "Transcoder",
    "describe_codecs",
    "fit_codec",
    "list_codebooks",
    "list_codecs",
    "load_codec",
    "save_codec",
]

#: Corpus a shipped codebook is taken from when none is named. The broadest
#: one, so a payload from it is the least specialised to any chemistry.
DEFAULT_CORPUS = "union"


def _check_formats(input_format: str, output_format: str, payload_format: str) -> None:
    """Reject unsupported format names.

    Checked before any fitting, so a typo costs nothing rather than a corpus
    pass followed by an error.
    """
    for value, allowed, label in (
        (input_format, INPUT_FORMATS, "input"),
        (output_format, OUTPUT_FORMATS, "output"),
        (payload_format, PAYLOAD_FORMATS, "payload"),
    ):
        if value not in allowed:
            known = ", ".join(allowed)
            msg = f"unsupported {label} format {value!r}; choose one of {known}"
            raise ValueError(msg)


def list_codecs(*, family: str | None = None, world: str = "closed") -> list[str]:
    """Return every codec name :func:`fit_codec` accepts.

    These are the names used in ``docs/RESULTS.md``. They come from the same
    factories the benchmark suites are built from, so a name that works in one
    place works in the other.

    :param family: Restrict to one family, as shown by :func:`describe_codecs`.
    :param world: Library world for lookup-decoded codecs.
    """
    return [name for name, codec in sorted(catalogue(world=world).items()) if family is None or codec.family == family]


def describe_codecs(*, family: str | None = None, world: str = "closed") -> list[dict[str, Any]]:
    """Return a metadata record for every codec, sorted by family then name.

    Each record carries ``name``, ``family``, ``kind``, ``size_model``,
    ``lossless_by_design`` and ``notes``. Check ``kind`` before choosing a
    payload format and ``lossless_by_design`` before expecting a round trip: a
    fingerprint codec cannot invert its own hash and decodes only by looking
    the payload up in a library.

    :param family: Restrict to one family.
    :param world: Library world for lookup-decoded codecs.
    """
    codecs = [codec for codec in catalogue(world=world).values() if family is None or codec.family == family]
    return [codec.describe() for codec in sorted(codecs, key=lambda item: (item.family, item.name))]


@dataclass(slots=True)
class Transcoder:
    """A fitted codec with the defaults to use it, by name.

    Build one with :func:`load_codec` or :func:`fit_codec`; the constructor is
    for a codebook already in hand.

    :param codebook: The fitted codec and its provenance.
    :param path: Where the codebook was read from or written to, if anywhere.
    :param fitted_now: Whether this call fitted the codec or loaded a cached
        codebook.
    :param budget: Bit budget payloads are judged against, 1000 bits by
        default.
    :param input_format: How incoming molecules are read; ``auto`` detects each
        one and reports what it decided.
    :param output_format: Representation recovered molecules come back in.
    :param payload_format: Textual form for payloads; ``auto`` means ``bits``
        for binary codecs and ``text`` for string ones.
    """

    codebook: Codebook
    path: Path | None = None
    fitted_now: bool = False
    budget: int = BIT_BUDGET
    input_format: str = "auto"
    output_format: str = "smiles"
    payload_format: str = "auto"

    def __post_init__(self) -> None:
        """Validate the stored defaults."""
        _check_formats(self.input_format, self.output_format, self.payload_format)

    @classmethod
    def load(cls, path: Path | str, **defaults: Any) -> Transcoder:
        """Load a codebook written earlier and wrap it.

        .. warning::

           A codebook is a Python pickle, so loading one runs the code inside
           it. Load only codebooks you wrote yourself.

        :param path: Codebook file.
        :param defaults: Any of the keyword defaults above.
        """
        target = Path(path)
        return cls(codebook=load(target), path=target, fitted_now=False, **defaults)

    @property
    def codec(self) -> BaseCodec:
        """Return the fitted codec itself, for the lower-level functions."""
        return self.codebook.codec

    @property
    def name(self) -> str:
        """Return the codec's name."""
        return self.codebook.name

    @property
    def provenance(self) -> Any:  # noqa: ANN401 - avoids re-exporting the dataclass
        """Return what the codec was fitted on."""
        return self.codebook.provenance

    def save(self, path: Path | str | None = None) -> Path:
        """Write the codebook, so another process can decode these payloads.

        :param path: Destination, defaulting to where this codebook came from
            or to its place in the cache.
        :returns: The path written.
        """
        target = Path(path) if path is not None else (self.path or auto_path(self.provenance))
        written = save(self.codebook, target)
        self.path = written
        return written

    def encode(
        self,
        molecule: str,
        *,
        input_format: str | None = None,
        payload_format: str | None = None,
        budget: int | None = None,
    ) -> Encoded:
        """Encode one molecule. See :func:`osmiumchem.transcode.encode`."""
        return encode(
            self.codec,
            molecule,
            input_format=input_format or self.input_format,
            payload_format=payload_format or self.payload_format,
            budget=self.budget if budget is None else budget,
        )

    def decode(self, payload: str | Any, *, output_format: str | None = None) -> Decoded:
        """Decode one payload. See :func:`osmiumchem.transcode.decode`."""
        return decode(self.codec, payload, output_format=output_format or self.output_format)

    def round_trip(
        self,
        molecule: str,
        *,
        input_format: str | None = None,
        output_format: str | None = None,
        payload_format: str | None = None,
        budget: int | None = None,
    ) -> RoundTripped:
        """Encode then decode one molecule, and judge whether it survived.

        ``result.verdict`` is ``"ok"``, ``"over-budget"`` or ``"failed"``.
        Keeping a molecule in too many bits is not the same outcome as losing
        it, and the budget results in ``docs/RESULTS.md`` turn on that.
        """
        return round_trip(
            self.codec,
            molecule,
            input_format=input_format or self.input_format,
            output_format=output_format or self.output_format,
            payload_format=payload_format or self.payload_format,
            budget=self.budget if budget is None else budget,
        )

    def encode_many(
        self,
        molecules: Iterable[str],
        *,
        input_format: str | None = None,
        payload_format: str | None = None,
        budget: int | None = None,
    ) -> list[Encoded]:
        """Encode several molecules, one result each, in order."""
        return encode_many(
            self.codec,
            molecules,
            input_format=input_format or self.input_format,
            payload_format=payload_format or self.payload_format,
            budget=self.budget if budget is None else budget,
        )

    def decode_many(
        self,
        payloads: Iterable[str | Any],
        *,
        output_format: str | None = None,
    ) -> list[Decoded]:
        """Decode several payloads, one result each, in order."""
        return decode_many(
            self.codec,
            payloads,
            output_format=output_format or self.output_format,
        )

    def round_trip_many(
        self,
        molecules: Iterable[str],
        *,
        input_format: str | None = None,
        output_format: str | None = None,
        payload_format: str | None = None,
        budget: int | None = None,
    ) -> list[RoundTripped]:
        """Round-trip several molecules, using both batch paths where they exist."""
        return round_trip_many(
            self.codec,
            molecules,
            input_format=input_format or self.input_format,
            output_format=output_format or self.output_format,
            payload_format=payload_format or self.payload_format,
            budget=self.budget if budget is None else budget,
        )

    def summary(self) -> str:
        """Return a one-line description of what is fitted and where it lives."""
        where = f" at {self.path}" if self.path is not None else " (not cached)"
        return f"{self.provenance.summary()}{where}"

    def __repr__(self) -> str:
        """Return a representation naming the codec and its fitting corpus."""
        provenance = self.provenance
        return (
            f"Transcoder({self.name!r}, corpus={provenance.corpus!r}, fitted={provenance.fitted}, budget={self.budget})"
        )


def _is_path(source: Path | str) -> bool:
    """Return whether ``source`` names a file rather than a codec.

    A codec name never ends in ``.oc.xz`` and never exists on disk, so the two
    cannot collide in practice.
    """
    if isinstance(source, Path):
        return True
    return source.endswith(SUFFIX) or Path(source).is_file()


def list_codebooks(codec: str | None = None) -> list[str]:
    """Return the corpus labels that :func:`load_codec` can load a codec from.

    :param codec: Restrict to one codec name; ``None`` lists every label any
        shipped codebook was fitted on.
    """
    return bundled_corpora(codec)


def load_codec(
    source: Path | str,
    *,
    corpus: str | None = None,
    budget: int = BIT_BUDGET,
    input_format: str = "auto",
    output_format: str = "smiles",
    payload_format: str = "auto",
) -> Transcoder:
    """Return a fitted codec, either shipped with the package or from a file.

    ``source`` is read as a codec name, or as a path when it ends in
    ``.oc.xz`` or names an existing file::

        oc.load_codec("smiles+arith-o4")                     # shipped
        oc.load_codec("smiles+arith-o4", corpus="protacs")   # shipped, other corpus
        oc.load_codec("series.oc.xz")                        # one you fitted

    The ``encode``, ``decode`` and ``roundtrip`` commands call this with the
    same defaults, so a payload produced at a shell prompt decodes here and the
    other way round.

    :param source: Codec name, or a path to a codebook file.
    :param corpus: Which shipped codebook to take, defaulting to
        :data:`DEFAULT_CORPUS`. :func:`list_codebooks` has the labels. Not
        allowed together with a path, since the file decides its own corpus.
    :param budget: Bit budget payloads are judged against.
    :param input_format: Default input representation.
    :param output_format: Default output representation.
    :param payload_format: Default payload form.
    :returns: A :class:`Transcoder` ready to encode and decode.
    :raises ValueError: If a path and a ``corpus`` are both given, or a format
        name is unsupported.
    :raises FileNotFoundError: If a named codebook file does not exist.
    :raises KeyError: If nothing ships for this codec and corpus.
    """
    _check_formats(input_format, output_format, payload_format)
    defaults = {
        "budget": budget,
        "input_format": input_format,
        "output_format": output_format,
        "payload_format": payload_format,
    }
    if _is_path(source):
        if corpus is not None:
            msg = f"{source} carries its own provenance; drop corpus={corpus!r}"
            raise ValueError(msg)
        target = Path(source)
        if not target.is_file():
            msg = f"no such codebook: {target}"
            raise FileNotFoundError(msg)
        return Transcoder(codebook=load(target), path=target, fitted_now=False, **defaults)

    # Resolved first, so a mistyped name costs a list of near spellings rather
    # than a report that nothing ships for it.
    name = resolve_codec(str(source)).name
    label = DEFAULT_CORPUS if corpus is None else corpus
    target = bundled_path(name, label)
    if not target.is_file():
        available = bundled_corpora(name)
        if not available:
            known = ", ".join(bundled_corpora()) or "none"
            msg = (
                f"no codebook ships for {name!r}: fingerprint and latent codecs "
                f"decode by looking a payload up in a library of molecules, so "
                f"their fitted state is a corpus. Fit one with fit_codec. "
                f"Corpora with shipped codebooks: {known}"
            )
            raise KeyError(msg)
        msg = f"no codebook ships for {name!r} on corpus {label!r}; available: {', '.join(available)}"
        raise KeyError(msg)
    return Transcoder(codebook=load(target), path=target, fitted_now=False, **defaults)


def save_codec(codec: Transcoder, path: Path | str) -> Path:
    """Write a fitted codec to ``path``, so another process can decode it.

    The format is the one the shipped codebooks use, so
    :func:`load_codec` reads the result back with no special case. Fitting is
    not cached automatically — a list of molecules has no reproducible name the
    way a corpus label does — so this is how a custom codec survives the
    process that made it.

    :param codec: A fitted transcoder, from :func:`fit_codec`.
    :param path: Destination; ``.oc.xz`` by convention.
    :returns: The path written.
    """
    return codec.save(path)


def fit_codec(
    name: str,
    molecules: Sequence[str],
    *,
    label: str | None = None,
    world: str = "closed",
    workers: int = 1,
    library: Sequence[str] | None = None,
    budget: int = BIT_BUDGET,
    input_format: str = "auto",
    output_format: str = "smiles",
    payload_format: str = "auto",
) -> Transcoder:
    """Fit a codec on molecules you already have.

    Payloads are charged against the fitted table, so fitting on the series you
    intend to encode costs fewer bits per molecule than using a codebook fitted
    on a generic corpus. The molecules are canonicalised here.

    The result is not cached: a list of molecules has no reproducible name the
    way a corpus label does. Pass it to :func:`save_codec` to keep it, which
    another process will need in order to decode these payloads.

    :param name: Codec name.
    :param molecules: The fitting corpus, as SMILES. Entries that do not parse
        are dropped, as the benchmark drops them.
    :param label: Name recorded in the provenance, by default a digest of the
        molecules so that two corpora cannot be confused.
    :param world: Library world for lookup-decoded codecs.
    :param workers: Processes used to canonicalise the corpus.
    :param library: Extra molecules for the lookup library.
    :param budget: Bit budget payloads are judged against.
    :param input_format: Default input representation.
    :param output_format: Default output representation.
    :param payload_format: Default payload form.
    :raises ValueError: If no molecule in ``molecules`` is usable.
    """
    _check_formats(input_format, output_format, payload_format)
    molecules_list = list(molecules)
    book = fit_codebook_on(
        name,
        molecules_list,
        corpus=label or f"custom:{library_digest(molecules_list)}",
        world=world,
        workers=workers,
        library=library,
    )
    return Transcoder(
        codebook=book,
        path=None,
        fitted_now=True,
        budget=budget,
        input_format=input_format,
        output_format=output_format,
        payload_format=payload_format,
    )
