# -*- coding: utf-8 -*-

"""Core types shared by every codec in the benchmark.

Two things are measured at once: whether ``decode(encode(mol))`` returns the
same molecule, and how many bits the payload takes. The second needs one
unambiguous notion of payload size, which is what :class:`Bits` provides. A
codec may emit a ``str`` (SMILES, SELFIES) or a packed bit sequence
(fingerprints, entropy coders, quantised latents); either way it reports an
exact bit count, and text is charged through a documented alphabet model
rather than through Python's internal representation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "Bits",
    "Codec",
    "CodecKind",
    "RoundTrip",
    "SizeModel",
]

#: How a codec represents an encoded molecule.
#:
#: ``"string"``
#:     A textual representation (SMILES, SELFIES, InChI, ...). Its bit cost is
#:     *derived* from an explicit alphabet model, see :class:`SizeModel`.
#: ``"bitstring"``
#:     A genuine bit sequence; its bit cost is simply its length.
CodecKind = Literal["string", "bitstring"]

#: How the bit cost of a *string* payload is accounted for.
#:
#: ``"utf8"``
#:     8 bits per UTF-8 byte: what you pay if you store the string verbatim.
#: ``"fixed_alphabet"``
#:     ``ceil(log2(|alphabet|))`` bits per character, the cost of a flat code
#:     over the alphabet actually observed in the corpus.
#: ``"exact"``
#:     The payload is already a bit sequence; its length is the cost.
SizeModel = Literal["utf8", "fixed_alphabet", "exact"]


@dataclass(frozen=True, slots=True)
class Bits:
    """An immutable, exactly-sized sequence of bits.

    The length is stored rather than read off the backing integer, because
    leading zeros count: ``0b0101`` and ``0b101`` are a 4-bit and a 3-bit
    payload.

    :param value: Non-negative integer holding the bits, most significant
        first, so bit ``i`` of the sequence is bit ``length - 1 - i`` of
        ``value``.
    :param length: Number of significant bits.
    """

    value: int
    length: int

    def __post_init__(self) -> None:
        """Check that ``value`` fits in ``length`` bits."""
        if self.length < 0:
            msg = f"bit length must be non-negative, got {self.length}"
            raise ValueError(msg)
        if self.value < 0:
            msg = f"bit value must be non-negative, got {self.value}"
            raise ValueError(msg)
        if self.value.bit_length() > self.length:
            msg = f"value needs {self.value.bit_length()} bits, which overflows the declared length {self.length}"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of bits."""
        return self.length

    def __iter__(self) -> Iterator[int]:
        """Iterate over the bits, most significant first."""
        for i in range(self.length - 1, -1, -1):
            yield (self.value >> i) & 1

    def __str__(self) -> str:
        """Render the payload as a string of ``0`` and ``1``."""
        return format(self.value, f"0{self.length}b") if self.length else ""

    @classmethod
    def from_iterable(cls, bits: Sequence[int] | Iterator[int]) -> Bits:
        """Build a payload from an iterable of ``0``/``1`` values."""
        value = 0
        length = 0
        for bit in bits:
            if bit not in (0, 1):
                msg = f"expected only 0 or 1, got {bit!r}"
                raise ValueError(msg)
            value = (value << 1) | bit
            length += 1
        return cls(value, length)

    @classmethod
    def from_string(cls, text: str) -> Bits:
        """Build a payload from a string of ``0`` and ``1`` characters."""
        if not text:
            return cls(0, 0)
        return cls(int(text, 2), len(text))

    @classmethod
    def from_bytes(cls, raw: bytes) -> Bits:
        """Build a payload from raw bytes, charging 8 bits per byte."""
        return cls(int.from_bytes(raw, "big") if raw else 0, 8 * len(raw))

    def to_bytes(self) -> bytes:
        """Pad to a whole number of bytes (on the right) and return the bytes."""
        padded = (self.length + 7) // 8
        if padded == 0:
            return b""
        return (self.value << (8 * padded - self.length)).to_bytes(padded, "big")


@dataclass(slots=True)
class RoundTrip:
    """The outcome of encoding then decoding a single molecule.

    :param ok: Whether the decoded molecule is the input molecule, as judged
        by :func:`osmiumchem.identity.same_molecule`.
    :param bits: Payload size in bits, or ``None`` if encoding failed.
    :param encoded: The payload, kept for failure reports.
    :param decoded: Canonical form of whatever came back, if anything.
    :param error: Exception text when a stage raised instead of returning.
    :param stage: Which stage failed (``"encode"``, ``"decode"``, ``"compare"``),
        or ``None`` on success.
    """

    ok: bool
    bits: int | None = None
    encoded: Any = None
    decoded: str | None = None
    error: str | None = None
    stage: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Codec(Protocol):
    """A reversible (or attempted-reversible) molecular representation.

    Codecs hold no per-call state. Anything learned from a corpus (character
    frequencies, a Huffman table, a quantisation scale, a nearest-neighbour
    index) is installed by :meth:`fit` first, so the decoder is a fixed shared
    artefact and only the per-molecule payload is charged against the bit
    budget. Same arrangement as a trained autoencoder: the weights ship once,
    the latent code travels per molecule.
    """

    #: Short, stable, human-readable identifier used in reports.
    name: str
    #: Whether payloads are strings or bit sequences.
    kind: CodecKind
    #: How a string payload's bit cost is computed.
    size_model: SizeModel

    def fit(self, smiles: Sequence[str]) -> None:
        """Learn any shared state from a training corpus of SMILES.

        A no-op for codecs that have no shared state. Implementations must not
        look at the evaluation split.
        """
        ...

    def encode(self, smiles: str) -> Any:
        """Encode a canonical SMILES into this codec's payload."""
        ...

    def decode(self, payload: Any) -> str | None:
        """Decode a payload back into a SMILES, or ``None`` on failure."""
        ...

    def payload_bits(self, payload: Any) -> int:
        """Return the exact bit cost of ``payload``."""
        ...


def flat_code_width(alphabet_size: int) -> int:
    """Return the width of a flat binary code over ``alphabet_size`` symbols.

    :param alphabet_size: Number of distinct symbols; must be at least 1.
    :returns: ``ceil(log2(alphabet_size))``, or ``0`` for a one-symbol
        alphabet, which carries no information.
    """
    if alphabet_size < 1:
        msg = f"alphabet size must be at least 1, got {alphabet_size}"
        raise ValueError(msg)
    if alphabet_size == 1:
        return 0
    return math.ceil(math.log2(alphabet_size))
