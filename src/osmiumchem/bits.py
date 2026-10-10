# -*- coding: utf-8 -*-

"""Sequential bit writing and reading, plus the integer codes codecs need.

Nothing here is molecule-aware. A byte-aligned container rounds every field up
to the next byte, up to seven wasted bits each time, which is worth avoiding
against a 1000-bit budget, so the structural codecs write through
:class:`BitWriter`.

Elias gamma and Elias delta cover the counts a structural encoding has to
write without knowing their size in advance: atom counts, ring sizes,
neighbour offsets. Gamma is shorter for small values, delta for large ones.
Both are prefix-free, so the reader needs no length field.
"""

from __future__ import annotations

from .types import Bits

__all__ = ["BitReader", "BitWriter", "elias_delta_cost", "elias_gamma_cost"]


class BitWriter:
    """Accumulate bits most-significant-first into a :class:`Bits` payload."""

    __slots__ = ("_length", "_value")

    def __init__(self) -> None:
        """Start an empty payload."""
        self._value = 0
        self._length = 0

    def __len__(self) -> int:
        """Return the number of bits written so far."""
        return self._length

    def bit(self, value: int) -> BitWriter:
        """Write a single bit.

        :param value: Taken as truthy/falsy, so ``0``/``1`` and
            ``False``/``True`` both work.
        """
        self._value = (self._value << 1) | (1 if value else 0)
        self._length += 1
        return self

    def uint(self, value: int, width: int) -> BitWriter:
        """Write ``value`` as a fixed-width unsigned integer.

        :param value: Non-negative integer that must fit in ``width`` bits.
        :param width: Number of bits to emit; may be ``0`` only for ``value=0``.
        :raises ValueError: If ``value`` is negative or does not fit.
        """
        if width < 0:
            msg = f"width must be non-negative, got {width}"
            raise ValueError(msg)
        if value < 0:
            msg = f"cannot write negative value {value} as unsigned"
            raise ValueError(msg)
        if value.bit_length() > width:
            msg = f"value {value} does not fit in {width} bits"
            raise ValueError(msg)
        if width:
            self._value = (self._value << width) | value
            self._length += width
        return self

    def elias_gamma(self, value: int) -> BitWriter:
        """Write a strictly positive integer in Elias gamma code.

        ``n`` zero bits followed by the ``n + 1`` bit binary expansion of
        ``value``, with ``n = floor(log2(value))``, so ``2n + 1`` bits in
        total.

        :param value: Integer greater than or equal to 1.
        """
        if value < 1:
            msg = f"Elias gamma encodes integers >= 1, got {value}"
            raise ValueError(msg)
        width = value.bit_length()
        self.uint(0, width - 1)
        return self.uint(value, width)

    def elias_delta(self, value: int) -> BitWriter:
        """Write a strictly positive integer in Elias delta code.

        :param value: Integer greater than or equal to 1.
        """
        if value < 1:
            msg = f"Elias delta encodes integers >= 1, got {value}"
            raise ValueError(msg)
        width = value.bit_length()
        self.elias_gamma(width)
        # ``width`` already implies the leading 1, so drop it.
        return self.uint(value & ((1 << (width - 1)) - 1), width - 1)

    def bits(self, payload: Bits) -> BitWriter:
        """Append an existing :class:`Bits` payload verbatim."""
        self._value = (self._value << payload.length) | payload.value
        self._length += payload.length
        return self

    def done(self) -> Bits:
        """Return the accumulated payload."""
        return Bits(self._value, self._length)


class BitReader:
    """Consume bits most-significant-first from a :class:`Bits` payload."""

    __slots__ = ("_payload", "_pos")

    def __init__(self, payload: Bits) -> None:
        """Read ``payload`` from its first bit."""
        self._payload = payload
        self._pos = 0

    @property
    def remaining(self) -> int:
        """Return the number of unread bits."""
        return self._payload.length - self._pos

    def bit(self) -> int:
        """Read a single bit.

        :raises EOFError: If the payload is exhausted.
        """
        if self._pos >= self._payload.length:
            msg = "ran past the end of the payload"
            raise EOFError(msg)
        shift = self._payload.length - 1 - self._pos
        self._pos += 1
        return (self._payload.value >> shift) & 1

    def uint(self, width: int) -> int:
        """Read a fixed-width unsigned integer.

        :param width: Number of bits to consume; ``0`` yields ``0``.
        :raises EOFError: If fewer than ``width`` bits remain.
        """
        if width < 0:
            msg = f"width must be non-negative, got {width}"
            raise ValueError(msg)
        if width == 0:
            return 0
        if self.remaining < width:
            msg = f"need {width} bits but only {self.remaining} remain"
            raise EOFError(msg)
        shift = self._payload.length - self._pos - width
        mask = (1 << width) - 1
        self._pos += width
        return (self._payload.value >> shift) & mask

    def elias_gamma(self) -> int:
        """Read an Elias gamma coded integer."""
        zeros = 0
        while self.bit() == 0:
            zeros += 1
            if zeros > self._payload.length:  # pragma: no cover - corrupt input guard
                msg = "malformed Elias gamma code"
                raise EOFError(msg)
        return (1 << zeros) | self.uint(zeros)

    def elias_delta(self) -> int:
        """Read an Elias delta coded integer."""
        width = self.elias_gamma()
        return (1 << (width - 1)) | self.uint(width - 1)


def elias_gamma_cost(value: int) -> int:
    """Return the number of bits Elias gamma needs for ``value``."""
    if value < 1:
        msg = f"Elias gamma encodes integers >= 1, got {value}"
        raise ValueError(msg)
    return 2 * value.bit_length() - 1


def elias_delta_cost(value: int) -> int:
    """Return the number of bits Elias delta needs for ``value``."""
    if value < 1:
        msg = f"Elias delta encodes integers >= 1, got {value}"
        raise ValueError(msg)
    width = value.bit_length()
    return elias_gamma_cost(width) + width - 1
