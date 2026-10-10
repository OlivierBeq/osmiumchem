# -*- coding: utf-8 -*-

"""Lossless bit-level codecs over a textual molecular representation.

Both requirements are met here at once. These codecs are entropy coders over a
string that already represents the molecule faithfully, so round-trip accuracy
is whatever the string representation achieves, complete for canonical SMILES
and SELFIES. The coders then shrink the payload well inside a 1000-bit budget.

The family forms a ladder, so a saving can be attributed to a cause:

``flat``
    ``ceil(log2(|alphabet|))`` bits per token. No modelling.
``huffman``
    An optimal per-symbol code. The gain over ``flat`` is symbol skew.
``arithmetic``, order 0
    Fractional bits per symbol. The gain over ``huffman`` is Huffman's
    inability to spend a non-integer number of bits.
``arithmetic``, order k
    Conditioned on the preceding ``k`` tokens. The gain over order 0 is
    context, where most of the remaining redundancy in SMILES sits: a ``(`` is
    usually followed by an atom, a ``[`` by an element symbol.

The general-purpose compressors (``zlib``, ``bz2``, ``lzma``) are here for
contrast, and they do badly. Each writes a multi-byte container header, which
on a 50-character string can cost more than the string. Worth measuring, since
reaching for ``zlib`` is the first instinct.

Shared state (frequency tables, Huffman lengths, compression dictionaries) is
fitted on a split disjoint from the evaluation set and treated as a shipped
artefact rather than charged per molecule, as a trained decoder's weights would
be.
"""

from __future__ import annotations

import bz2
import lzma
import zlib
from collections import Counter
from collections.abc import Sequence

from ..coding.alphabet import EscapingAlphabet
from ..coding.arithmetic import ArithmeticCodec
from ..coding.huffman import HuffmanCode
from ..coding.model import OrderKModel
from ..types import Bits
from .base import BaseCodec
from .strings import SelfiesCodec, SmilesCodec, StringCodec

__all__ = [
    "ArithmeticStringCodec",
    "CompressorCodec",
    "FlatCodeCodec",
    "HuffmanStringCodec",
    "entropy_codecs",
]


def _text_codec(representation: str) -> StringCodec:
    """Return the textual codec named by ``representation``.

    :param representation: ``"smiles"``, ``"selfies"``, or ``"selfies-<preset>"``
        to select a SELFIES valence preset such as ``selfies-hypervalent``.
    """
    if representation == "smiles":
        return SmilesCodec()
    if representation == "selfies":
        return SelfiesCodec()
    if representation.startswith("selfies-"):
        return SelfiesCodec(representation.removeprefix("selfies-"))
    msg = f"unsupported representation {representation!r}; expected 'smiles', 'selfies' or 'selfies-<preset>'"
    raise ValueError(msg)


class _TextBackedCodec(BaseCodec):
    """Shared plumbing for codecs that compress a textual representation.

    :param representation: ``"smiles"`` or ``"selfies"``.
    """

    kind = "bitstring"
    size_model = "exact"
    family = "entropy"

    def __init__(self, representation: str = "smiles") -> None:
        """Wrap the chosen textual representation."""
        self.representation = representation
        self._text = _text_codec(representation)

    def _tokens(self, smiles: str) -> list[str] | None:
        """Return the token sequence for ``smiles``, or ``None`` on failure.

        The tokenisation is checked to partition the text exactly. A tokeniser
        that dropped a character would corrupt the molecule without saying so,
        the one failure this benchmark must never score as a success, so a
        mismatch becomes an encode failure.
        """
        text = self._text.to_text(smiles)
        if text is None:
            return None
        tokens = self._text.tokenize(text)
        if "".join(tokens) != text:
            return None
        return tokens

    def _from_tokens(self, tokens: Sequence[str]) -> str | None:
        """Rebuild a SMILES from tokens."""
        return self._text.from_text("".join(tokens))

    def _token_corpus(self, smiles: Sequence[str]) -> list[list[str]]:
        """Return the token sequences of every fittable molecule in ``smiles``."""
        return [tokens for item in smiles if (tokens := self._tokens(item)) is not None]

    def fit(self, smiles: Sequence[str]) -> None:
        """Fit the textual codec's alphabet, then any subclass state."""
        self._text.fit(smiles)


class FlatCodeCodec(_TextBackedCodec):
    """Fixed-width code over the fitted token alphabet.

    The zero-modelling baseline, where every token costs the same
    ``ceil(log2(|alphabet|))`` bits.
    """

    def __init__(self, representation: str = "smiles") -> None:
        """Create the codec for ``representation``."""
        super().__init__(representation)
        self.name = f"{representation}+flat"
        self.notes = "fixed-width token code; no modelling, the zero-effort baseline"
        self._alphabet = EscapingAlphabet()
        self._width = 8

    def fit(self, smiles: Sequence[str]) -> None:
        """Collect the token alphabet."""
        super().fit(smiles)
        self._alphabet.fit(self._token_corpus(smiles))
        self._width = max(1, (len(self._alphabet) - 1).bit_length())

    def encode(self, smiles: str) -> Bits | None:
        """Pack the escaped symbol sequence into fixed-width fields."""
        tokens = self._tokens(smiles)
        if tokens is None:
            return None
        ids = self._alphabet.to_ids(tokens)
        if ids is None:
            return None
        value = 0
        for index in [*ids, self._alphabet.stop_id]:
            value = (value << self._width) | index
        return Bits(value, self._width * (len(ids) + 1))

    def decode(self, payload: Bits) -> str | None:
        """Unpack fixed-width fields back into a SMILES."""
        if not isinstance(payload, Bits):
            return None
        stop = self._alphabet.stop_id
        count = payload.length // self._width
        ids: list[int] = []
        for position in range(count):
            shift = payload.length - self._width * (position + 1)
            index = (payload.value >> shift) & ((1 << self._width) - 1)
            if index == stop:
                tokens = self._alphabet.from_ids(ids)
                return None if tokens is None else self._from_tokens(tokens)
            ids.append(index)
        return None


class HuffmanStringCodec(_TextBackedCodec):
    """Canonical Huffman code over token frequencies."""

    def __init__(self, representation: str = "smiles") -> None:
        """Create the codec for ``representation``."""
        super().__init__(representation)
        self.name = f"{representation}+huffman"
        self.notes = "optimal per-symbol code; the gain over flat is symbol skew alone"
        self._alphabet = EscapingAlphabet()
        self._code = HuffmanCode()

    def fit(self, smiles: Sequence[str]) -> None:
        """Fit Huffman code lengths on the fitting corpus."""
        super().fit(smiles)
        corpus = self._token_corpus(smiles)
        self._alphabet.fit(corpus)
        self._code = HuffmanCode(stop_symbol=self._alphabet.stop)
        self._code.fit_counts(self._alphabet.expanded_counts(corpus))

    def encode(self, smiles: str) -> Bits | None:
        """Huffman-encode the escaped symbol sequence."""
        tokens = self._tokens(smiles)
        if tokens is None:
            return None
        expanded = self._alphabet.expand(tokens)
        if expanded is None:
            return None
        return self._code.encode(expanded)

    def decode(self, payload: Bits) -> str | None:
        """Huffman-decode back to a SMILES."""
        if not isinstance(payload, Bits):
            return None
        expanded = self._code.decode(payload)
        if expanded is None:
            return None
        tokens = self._alphabet.collapse(expanded)
        if tokens is None:
            return None
        return self._from_tokens(tokens)


class ArithmeticStringCodec(_TextBackedCodec):
    """Order-``k`` arithmetic coding over tokens.

    :param representation: ``"smiles"`` or ``"selfies"``.
    :param order: Context length; ``0`` is a plain unigram model.
    :param weight_base: How strongly longer contexts are favoured.
    :param max_total: Frequency-total cap, trading coding efficiency for speed.
    """

    def __init__(
        self,
        representation: str = "smiles",
        order: int = 3,
        weight_base: int = 24,
        max_total: int = 1 << 14,
    ) -> None:
        """Create the codec."""
        super().__init__(representation)
        self.order = order
        self.name = f"{representation}+arith-o{order}"
        self.notes = (
            "exact arithmetic coding, order 0; fractional bits per symbol"
            if order == 0
            else f"exact arithmetic coding conditioned on the previous {order} tokens"
        )
        self._alphabet = EscapingAlphabet()
        self._model = OrderKModel(order=order, weight_base=weight_base, max_total=max_total)
        self._codec: ArithmeticCodec | None = None

    def fit(self, smiles: Sequence[str]) -> None:
        """Fit the context model on the fitting corpus."""
        super().fit(smiles)
        corpus = self._token_corpus(smiles)
        if not corpus:
            msg = "cannot fit an arithmetic codec on an empty corpus"
            raise ValueError(msg)
        self._alphabet.fit(corpus)
        sequences = [ids for tokens in corpus if (ids := self._alphabet.to_ids(tokens)) is not None]
        self._model.fit(sequences, size=len(self._alphabet), stop=self._alphabet.stop_id)
        self._codec = ArithmeticCodec(self._model)

    def encode(self, smiles: str) -> Bits | None:
        """Arithmetic-encode the escaped symbol sequence."""
        if self._codec is None:
            msg = f"{self.name} must be fitted before use"
            raise RuntimeError(msg)
        tokens = self._tokens(smiles)
        if tokens is None:
            return None
        ids = self._alphabet.to_ids(tokens)
        if ids is None:
            # Reachable only for non-ASCII input, which cannot be escaped.
            return None
        return self._codec.encode(ids)

    def decode(self, payload: Bits) -> str | None:
        """Arithmetic-decode back to a SMILES."""
        if self._codec is None:
            msg = f"{self.name} must be fitted before use"
            raise RuntimeError(msg)
        if not isinstance(payload, Bits):
            return None
        try:
            ids = self._codec.decode(payload)
        except (ValueError, EOFError):
            return None
        tokens = self._alphabet.from_ids(ids)
        if tokens is None:
            return None
        return self._from_tokens(tokens)


class CompressorCodec(_TextBackedCodec):
    """A general-purpose byte compressor applied to the textual form.

    :param representation: ``"smiles"`` or ``"selfies"``.
    :param algorithm: ``"zlib"``, ``"bz2"`` or ``"lzma"``.
    """

    def __init__(self, representation: str = "smiles", algorithm: str = "zlib") -> None:
        """Create the codec."""
        super().__init__(representation)
        if algorithm not in {"zlib", "bz2", "lzma"}:
            msg = f"unsupported algorithm {algorithm!r}"
            raise ValueError(msg)
        self.algorithm = algorithm
        self.name = f"{representation}+{algorithm}"
        self.notes = "general-purpose compressor; container headers dominate at this message length"

    def encode(self, smiles: str) -> Bits | None:
        """Compress the textual form into a byte payload."""
        text = self._text.to_text(smiles)
        if text is None:
            return None
        raw = text.encode("utf-8")
        if self.algorithm == "zlib":
            blob = zlib.compress(raw, 9)
        elif self.algorithm == "bz2":
            blob = bz2.compress(raw, 9)
        else:
            blob = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
        return Bits.from_bytes(blob)

    def decode(self, payload: Bits) -> str | None:
        """Decompress a byte payload back to a SMILES."""
        if not isinstance(payload, Bits):
            return None
        blob = payload.to_bytes()
        try:
            if self.algorithm == "zlib":
                raw = zlib.decompress(blob)
            elif self.algorithm == "bz2":
                raw = bz2.decompress(blob)
            else:
                raw = lzma.decompress(blob)
        except (zlib.error, lzma.LZMAError, OSError, EOFError, ValueError):
            # ``lzma.LZMAError`` derives from ``Exception`` rather than
            # ``OSError``, so naming it is what keeps a corrupt payload from
            # escaping as a crash.
            return None
        return self._text.from_text(raw.decode("utf-8", errors="replace"))


def token_frequency_report(smiles: Sequence[str], representation: str = "smiles") -> Counter[str]:
    """Return token counts over a corpus, for inspection and documentation."""
    codec = _text_codec(representation)
    counts: Counter[str] = Counter()
    for item in smiles:
        text = codec.to_text(item)
        if text is not None:
            counts.update(codec.tokenize(text))
    return counts


def entropy_codecs(orders: Sequence[int] = (0, 1, 2, 3, 4)) -> list[BaseCodec]:
    """Return the full entropy-coding ladder for SMILES and SELFIES.

    The SELFIES rungs appear twice, under the default valence rules and under
    the ``hypervalent`` preset, because the default refuses a small but
    non-negligible share of real molecules, which on its own keeps it short of
    a 99.99% target.

    :param orders: Context orders to include for arithmetic coding.
    """
    codecs: list[BaseCodec] = []
    for representation in ("smiles", "selfies", "selfies-hypervalent"):
        codecs.append(FlatCodeCodec(representation))
        codecs.append(HuffmanStringCodec(representation))
        codecs.extend(ArithmeticStringCodec(representation, order=order) for order in orders)
    codecs.extend(CompressorCodec("smiles", algorithm) for algorithm in ("zlib", "bz2", "lzma"))
    return codecs
