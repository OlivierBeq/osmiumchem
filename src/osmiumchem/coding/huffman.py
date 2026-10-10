# -*- coding: utf-8 -*-

"""Canonical Huffman coding over a fixed symbol alphabet.

Huffman sits between a flat fixed-width code and arithmetic coding. It uses the
skew in SMILES characters, where ``C`` and ``c`` dominate and ``%`` and bracket
atoms are rare, but it cannot spend a fractional bit per symbol or condition on
context. Having it in the results separates the two gains: flat to Huffman is
symbol skew, Huffman to order-k arithmetic is context.

Codes are canonical, so the table is reproducible. Lengths come from the
Huffman tree, then codes are assigned by sorting on ``(length, symbol)``. Heap
ties during tree building are broken deterministically for the same reason.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable, Sequence

from ..bits import BitReader, BitWriter
from ..types import Bits

__all__ = ["HuffmanCode"]


class HuffmanCode:
    """A frozen canonical Huffman code fitted on token frequencies.

    :param stop_symbol: Sentinel appended to every message, so the decoder
        knows where to stop without a length prefix.
    """

    __slots__ = ("_decode_table", "_encode_table", "_lengths", "stop_symbol")

    def __init__(self, stop_symbol: str = "\x00stop") -> None:
        """Create an unfitted code."""
        self.stop_symbol = stop_symbol
        self._encode_table: dict[str, tuple[int, int]] = {}
        self._decode_table: dict[tuple[int, int], str] = {}
        self._lengths: dict[str, int] = {}

    @property
    def alphabet(self) -> list[str]:
        """Return the coded symbols, including the stop symbol."""
        return sorted(self._encode_table)

    def fit(self, sequences: Iterable[Sequence[str]]) -> HuffmanCode:
        """Fit the code on tokenised training sequences.

        :param sequences: Token lists, one per training molecule.
        :returns: ``self``, for chaining.
        """
        counts: Counter[str] = Counter()
        messages = 0
        for seq in sequences:
            counts.update(seq)
            messages += 1
        counts[self.stop_symbol] += max(1, messages)
        return self.fit_counts(counts)

    def fit_counts(self, counts: dict[str, int] | Counter[str]) -> HuffmanCode:
        """Fit the code directly from symbol counts.

        :param counts: Positive count per symbol; must be non-empty.
        :returns: ``self``, for chaining.
        """
        if not counts:
            msg = "cannot build a Huffman code over an empty alphabet"
            raise ValueError(msg)
        self._lengths = _code_lengths(counts)
        self._assign_canonical_codes()
        return self

    def _assign_canonical_codes(self) -> None:
        """Turn code lengths into canonical codes."""
        self._encode_table = {}
        self._decode_table = {}
        code = 0
        previous_length = 0
        for symbol, length in sorted(self._lengths.items(), key=lambda item: (item[1], item[0])):
            code <<= length - previous_length
            self._encode_table[symbol] = (code, length)
            self._decode_table[(length, code)] = symbol
            code += 1
            previous_length = length

    def cost_bits(self, tokens: Sequence[str]) -> int | None:
        """Return the payload size for ``tokens``, or ``None`` if uncodable."""
        try:
            total = sum(self._encode_table[token][1] for token in tokens)
        except KeyError:
            return None
        return total + self._encode_table[self.stop_symbol][1]

    def encode(self, tokens: Sequence[str]) -> Bits | None:
        """Encode tokens, returning ``None`` if any symbol is outside the code.

        An out-of-alphabet symbol is reported rather than substituted, because
        a substitution would change the molecule.
        """
        writer = BitWriter()
        for token in [*tokens, self.stop_symbol]:
            entry = self._encode_table.get(token)
            if entry is None:
                return None
            code, length = entry
            writer.uint(code, length)
        return writer.done()

    def decode(self, payload: Bits) -> list[str] | None:
        """Decode a payload into tokens, or ``None`` if it is malformed."""
        reader = BitReader(payload)
        tokens: list[str] = []
        code = 0
        length = 0
        while True:
            if reader.remaining == 0:
                # A well-formed payload ends at the stop symbol, so running
                # dry before that means it is corrupt.
                return None
            code = (code << 1) | reader.bit()
            length += 1
            symbol = self._decode_table.get((length, code))
            if symbol is None:
                if length > 64:  # pragma: no cover - corrupt input guard
                    return None
                continue
            if symbol == self.stop_symbol:
                return tokens
            tokens.append(symbol)
            code = 0
            length = 0


def _code_lengths(counts: dict[str, int] | Counter[str]) -> dict[str, int]:
    """Return Huffman code lengths per symbol.

    A one-symbol alphabet gets length 1 rather than 0, since a zero-length
    code cannot be written to or read from a payload.
    """
    if len(counts) == 1:
        return {next(iter(counts)): 1}
    # Heap entries are (weight, tiebreak, payload), with ``tiebreak`` a
    # monotone counter, so ordering never depends on dict iteration order.
    heap: list[tuple[int, int, object]] = []
    for order, (symbol, count) in enumerate(sorted(counts.items())):
        heapq.heappush(heap, (max(1, count), order, symbol))
    tiebreak = len(counts)
    while len(heap) > 1:
        left_weight, _, left = heapq.heappop(heap)
        right_weight, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_weight + right_weight, tiebreak, (left, right)))
        tiebreak += 1
    lengths: dict[str, int] = {}
    _walk(heap[0][2], 0, lengths)
    return lengths


def _walk(node: object, depth: int, out: dict[str, int]) -> None:
    """Record the depth of every leaf under ``node``."""
    if isinstance(node, tuple):
        left, right = node
        _walk(left, depth + 1, out)
        _walk(right, depth + 1, out)
    else:
        out[str(node)] = depth
