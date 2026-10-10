# -*- coding: utf-8 -*-

"""Exact arithmetic coding over arbitrary-precision integers.

Production coders keep the active interval in a fixed-width register and
renormalise as bits settle, which is fast and easy to get subtly wrong; carry
propagation and underflow straddling are where those implementations leak.
Exact reversibility is the whole premise here, so this one trades the other
way.

The interval is an exact rational ``[lo / den, (lo + size) / den)`` over
Python's unbounded integers. A symbol with frequency ``f`` out of total ``T``
and cumulative predecessor count ``c`` subdivides it as::

    den  <- den * T
    lo   <- lo * T + size * c
    size <- size * f

Nothing rounds, so there is no renormalisation and no carry to propagate. The
payload is the shortest binary fraction inside the final interval, and the
decoder walks the same subdivision back. The arithmetic being exact and
integer-only, the bitstring reproduces across platforms and Python builds.

The price is big-integer arithmetic growing with message length, quadratic
overall. For strings of tens to a few hundred symbols that is fast enough.

A stop symbol terminates the message instead of a length prefix, so the decoder
needs only the payload and the model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Bits

__all__ = ["ArithmeticCodec", "FrequencyModel"]


@runtime_checkable
class FrequencyModel(Protocol):
    """Supplies integer symbol frequencies conditioned on the emitted prefix.

    The model must be deterministic and depend only on symbols already coded.
    The decoder rebuilds the prefix one symbol at a time, and it has to see the
    distribution the encoder saw.
    """

    #: Number of symbols, including the stop symbol.
    size: int
    #: Index of the stop symbol within the alphabet.
    stop: int

    def cumulative(self, prefix: tuple[int, ...]) -> tuple[list[int], int]:
        """Return cumulative frequencies and their total for the next symbol.

        :param prefix: Symbol ids already coded, oldest first.
        :returns: ``(cum, total)``, where ``cum`` has ``size + 1`` entries with
            ``cum[0] == 0``, ``cum[-1] == total`` and a strictly positive
            frequency ``cum[s + 1] - cum[s]`` for every symbol. A zero
            frequency makes a symbol unencodable, so models must smooth.
        """
        ...


class ArithmeticCodec:
    """Encode and decode symbol sequences against a :class:`FrequencyModel`."""

    __slots__ = ("_model",)

    def __init__(self, model: FrequencyModel) -> None:
        """Bind the codec to ``model``."""
        self._model = model

    def encode(self, symbols: list[int]) -> Bits:
        """Encode a sequence of symbol ids into a bit payload.

        :param symbols: Symbol ids, excluding the stop symbol, which is
            appended automatically.
        :returns: The shortest bit payload identifying the coded interval.
        :raises ValueError: If a symbol id is out of range, or if the model
            hands back a zero frequency for a symbol that must be coded.
        """
        model = self._model
        lo = 0
        size = 1
        den = 1
        prefix: list[int] = []
        for symbol in [*symbols, model.stop]:
            if not 0 <= symbol < model.size:
                msg = f"symbol id {symbol} outside alphabet of size {model.size}"
                raise ValueError(msg)
            cum, total = model.cumulative(tuple(prefix))
            freq = cum[symbol + 1] - cum[symbol]
            if freq <= 0:
                msg = f"model assigned zero frequency to symbol {symbol}; models must smooth"
                raise ValueError(msg)
            lo = lo * total + size * cum[symbol]
            size = size * freq
            den = den * total
            prefix.append(symbol)
        return _shortest_fraction_in(lo, size, den)

    def decode(self, payload: Bits, *, max_symbols: int = 100_000) -> list[int]:
        """Decode a bit payload back into symbol ids.

        :param payload: The payload :meth:`encode` returned. Its declared bit
            length is part of the data, trailing zero bits included.
        :param max_symbols: Bound on the decode loop, so a corrupt payload
            cannot spin forever.
        :returns: The symbol ids, with the stop symbol stripped.
        :raises ValueError: If the payload does not terminate within
            ``max_symbols`` symbols.
        """
        model = self._model
        # The payload denotes the binary fraction ``point / 2 ** width``.
        point = payload.value
        width = payload.length
        scale = 1 << width

        lo = 0
        size = 1
        den = 1
        prefix: list[int] = []
        for _ in range(max_symbols):
            cum, total = model.cumulative(tuple(prefix))
            # Where the payload sits inside the current interval, scaled to
            # ``total``. Exact integer arithmetic throughout.
            target = ((point * den - lo * scale) * total) // (size * scale)
            symbol = _bisect_bucket(cum, target)
            if symbol == model.stop:
                return prefix
            lo = lo * total + size * cum[symbol]
            size = size * (cum[symbol + 1] - cum[symbol])
            den = den * total
            prefix.append(symbol)
        msg = f"payload did not terminate within {max_symbols} symbols"
        raise ValueError(msg)


def _bisect_bucket(cum: list[int], target: int) -> int:
    """Return the symbol whose cumulative bucket contains ``target``.

    :param cum: Cumulative frequencies, non-decreasing, ``cum[0] == 0``.
    :param target: A value in ``[0, cum[-1])``.
    """
    low = 0
    high = len(cum) - 1
    while low + 1 < high:
        mid = (low + high) // 2
        if cum[mid] <= target:
            low = mid
        else:
            high = mid
    return low


def _shortest_fraction_in(lo: int, size: int, den: int) -> Bits:
    """Return the fewest-bit binary fraction inside ``[lo/den, (lo+size)/den)``.

    Any point inside the final interval identifies the message, so the encoder
    picks the one with the shortest binary expansion. That expansion is the
    payload, as a fixed-length bit field whose length travels with it because
    leading zeros matter.

    :param lo: Numerator of the interval's lower bound.
    :param size: Numerator of the interval's width, strictly positive.
    :param den: Shared denominator, strictly positive.
    """
    if size <= 0 or den <= 0:  # pragma: no cover - guards an impossible state
        msg = f"degenerate interval: size={size}, den={den}"
        raise ValueError(msg)
    # An interval of width ``size/den`` holds a dyadic point as soon as the
    # resolution is finer than the interval, so the search starts just below
    # that bound rather than at zero.
    width = max(0, (den // size).bit_length() - 1)
    while True:
        scale = 1 << width
        # Smallest ``point`` with ``point / scale >= lo / den``.
        point = -((-lo * scale) // den)
        if point * den < (lo + size) * scale:
            return Bits(point, width)
        width += 1
