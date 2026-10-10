# -*- coding: utf-8 -*-

"""Base classes shared by the codec families.

Every codec answers four questions: what shared state it needs
(:meth:`~BaseCodec.fit`), what a molecule becomes
(:meth:`~BaseCodec.encode`), what that turns back into
(:meth:`~BaseCodec.decode`), and what it cost
(:meth:`~BaseCodec.payload_bits`).

Cost is the subtle one, because a SMILES string has no intrinsic bit length;
it depends on how you agree to store it. ``len(smiles) * 8`` charges a full
byte for an alphabet of about forty symbols, overstating the cost by a third
and flattering every compressor measured against it. :class:`StringCodec`
charges a flat code over the alphabet observed while fitting,
``ceil(log2(|alphabet|))`` bits per token, and records the UTF-8 figure
separately. That is the fairest zero-effort baseline: no modelling, no
compression, just no bits spent on symbols that never occur.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..types import Bits, CodecKind, SizeModel, flat_code_width

__all__ = ["BaseCodec", "StringCodec"]


class BaseCodec:
    """Common scaffolding for codecs.

    :cvar family: Grouping used to organise the results table.
    :cvar lossless_by_design: Whether the technique preserves information in
        principle. ``False`` for codecs expected to lose molecules, so the
        report can say why they failed.
    """

    name: str = "base"
    kind: CodecKind = "bitstring"
    size_model: SizeModel = "exact"
    family: str = "other"
    notes: str = ""
    lossless_by_design: bool = True

    def fit(self, smiles: Sequence[str]) -> None:
        """Learn shared state from a fitting corpus. Default is a no-op."""

    def encode(self, smiles: str) -> Any:
        """Encode a canonical SMILES. Subclasses must override."""
        raise NotImplementedError

    def decode(self, payload: Any) -> str | None:
        """Decode a payload back to SMILES. Subclasses must override."""
        raise NotImplementedError

    def payload_bits(self, payload: Any) -> int:
        """Return the exact bit cost of ``payload``."""
        if isinstance(payload, Bits):
            return payload.length
        if isinstance(payload, (bytes, bytearray)):
            return 8 * len(payload)
        msg = f"{type(self).__name__} cannot size a payload of type {type(payload).__name__}"
        raise TypeError(msg)

    def describe(self) -> dict[str, Any]:
        """Return metadata used by the report."""
        return {
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "size_model": self.size_model,
            "lossless_by_design": self.lossless_by_design,
            "notes": self.notes,
        }


class StringCodec(BaseCodec):
    """A codec whose payload is text, charged under a flat alphabet code.

    Subclasses implement :meth:`to_text` and :meth:`from_text`. Tokenisation,
    alphabet discovery and bit accounting happen here.
    """

    kind: CodecKind = "string"
    size_model: SizeModel = "fixed_alphabet"

    def __init__(self) -> None:
        """Create a codec with an empty alphabet."""
        self._alphabet: dict[str, int] = {}
        self._code_width: int = 8

    @property
    def alphabet_size(self) -> int:
        """Return the number of distinct tokens seen while fitting."""
        return len(self._alphabet)

    @property
    def code_width(self) -> int:
        """Return the per-token width of the flat code, in bits."""
        return self._code_width

    def tokenize(self, text: str) -> list[str]:
        """Split a payload into the tokens the flat code is defined over.

        Characters by default; token-based representations override it.
        """
        return list(text)

    def to_text(self, smiles: str) -> str | None:
        """Convert a SMILES into this codec's textual form."""
        raise NotImplementedError

    def from_text(self, text: str) -> str | None:
        """Convert this codec's textual form back into a SMILES."""
        raise NotImplementedError

    def fit(self, smiles: Sequence[str]) -> None:
        """Discover the token alphabet from the fitting corpus."""
        tokens: set[str] = set()
        for item in smiles:
            text = self.to_text(item)
            if text is not None:
                tokens.update(self.tokenize(text))
        self._alphabet = {token: index for index, token in enumerate(sorted(tokens))}
        self._code_width = flat_code_width(max(1, len(self._alphabet)))

    def encode(self, smiles: str) -> str | None:
        """Return the textual payload for ``smiles``.

        The tokenisation is checked to partition the payload exactly, so a
        tokeniser bug surfaces as an encode failure rather than a corrupted
        molecule.
        """
        text = self.to_text(smiles)
        if text is None:
            return None
        if "".join(self.tokenize(text)) != text:
            return None
        return text

    def decode(self, payload: Any) -> str | None:
        """Return the SMILES recovered from a textual payload."""
        if not isinstance(payload, str):
            return None
        return self.from_text(payload)

    def payload_bits(self, payload: Any) -> int:
        """Return the flat-code cost of a textual payload.

        A token missing from the fitted alphabet is charged at the width the
        alphabet would need to hold it, so an unseen symbol cannot make a
        payload look cheaper than it is.
        """
        if not isinstance(payload, str):
            return super().payload_bits(payload)
        tokens = self.tokenize(payload)
        unseen = {token for token in tokens if token not in self._alphabet}
        width = self._code_width
        if unseen:
            width = flat_code_width(max(1, len(self._alphabet) + len(unseen)))
        return width * len(tokens)

    def utf8_bits(self, payload: str) -> int:
        """Return the cost of storing ``payload`` verbatim as UTF-8."""
        return 8 * len(payload.encode("utf-8"))

    def entropy_floor_bits(self, payload: str, counts: dict[str, int]) -> float:
        """Return the order-0 entropy cost of ``payload`` under ``counts``.

        Reported next to the flat-code figure, to show how much of a
        compressor's gain comes from symbol skew alone.
        """
        total = sum(counts.values())
        if total == 0:
            return math.inf
        bits = 0.0
        for token in self.tokenize(payload):
            probability = counts.get(token, 0) / total
            if probability <= 0:
                return math.inf
            bits -= math.log2(probability)
        return bits
