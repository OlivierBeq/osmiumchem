# -*- coding: utf-8 -*-

"""Token alphabets with an escape hatch for symbols never seen while fitting.

A static model is fitted on a finite corpus, so eventually it meets a token
that corpus did not contain. On a 4000-molecule fit split, about one evaluation
molecule in four hundred carries one: ``[SH]``, ``[/O]``, exotic isotopes, rare
elements. Giving up on those caps a codec near 99.7%, well short of the 99.99%
target, and more fitting data does not remove the tail, since the space of
legal bracket atoms is effectively unbounded.

So the alphabet always holds every printable ASCII character alongside the
tokens seen while fitting, and an unknown token goes out as an escape symbol,
its own characters, and a closing escape symbol::

    [SH]  ->  ESC  '['  'S'  'H'  ']'  ESC

Escapes are rare, so the average cost is negligible, and SMILES and SELFIES are
pure ASCII, so the fallback is always available. The codec stays lossless on
every input while spending close to the model's entropy on the common case. The
closing delimiter reuses the escape symbol, which is unambiguous because a
token's own characters never include it.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Iterable, Sequence

__all__ = ["EscapingAlphabet"]

#: Characters guaranteed to be codable, so any ASCII token can be escaped.
_PRINTABLE = tuple(sorted(set(string.printable) - set(string.whitespace) | {" "}))


class EscapingAlphabet:
    """A token alphabet that can always encode any ASCII token.

    :param escape: Sentinel marking the start and end of a literal run,
        outside the printable range so it cannot collide with a real token.
    :param stop: End-of-message sentinel, which saves a length prefix.
    """

    __slots__ = ("_index", "_observed", "_symbols", "escape", "stop")

    def __init__(self, escape: str = "\x00esc", stop: str = "\x00stop") -> None:
        """Create an unfitted alphabet."""
        self.escape = escape
        self.stop = stop
        self._symbols: list[str] = []
        self._index: dict[str, int] = {}
        self._observed: set[str] = set()

    def __len__(self) -> int:
        """Return the number of symbols, including both sentinels."""
        return len(self._symbols)

    @property
    def symbols(self) -> list[str]:
        """Return the symbol list, in id order."""
        return self._symbols

    @property
    def index(self) -> dict[str, int]:
        """Return the symbol-to-id mapping."""
        return self._index

    @property
    def stop_id(self) -> int:
        """Return the id of the stop sentinel."""
        return self._index[self.stop]

    @property
    def escape_id(self) -> int:
        """Return the id of the escape sentinel."""
        return self._index[self.escape]

    @property
    def observed(self) -> set[str]:
        """Return the tokens actually seen while fitting."""
        return self._observed

    def fit(self, sequences: Iterable[Sequence[str]]) -> EscapingAlphabet:
        """Build the alphabet from tokenised fitting sequences.

        :param sequences: Token lists, one per fitting molecule.
        :returns: ``self``, for chaining.
        """
        observed: set[str] = set()
        for sequence in sequences:
            observed.update(sequence)
        self._observed = observed
        # Printable ASCII is always present, so escaping always works.
        self._symbols = sorted(observed | set(_PRINTABLE))
        self._symbols.extend((self.escape, self.stop))
        self._index = {symbol: i for i, symbol in enumerate(self._symbols)}
        return self

    def expand(self, tokens: Sequence[str]) -> list[str] | None:
        """Rewrite tokens into codable symbols, escaping unknown ones.

        :param tokens: Tokens from a molecular string.
        :returns: The symbol sequence, or ``None`` for a token holding a
            non-ASCII character, which cannot be escaped.
        """
        out: list[str] = []
        for token in tokens:
            if token in self._index and token not in (self.escape, self.stop):
                out.append(token)
                continue
            out.append(self.escape)
            for character in token:
                if character not in self._index:
                    return None
                out.append(character)
            out.append(self.escape)
        return out

    def collapse(self, symbols: Sequence[str]) -> list[str] | None:
        """Invert :meth:`expand`, rejoining escaped literal runs.

        :param symbols: Symbol sequence as produced by :meth:`expand`.
        :returns: The original tokens, or ``None`` if the escape framing is
            malformed, which means the payload is corrupt.
        """
        out: list[str] = []
        position = 0
        count = len(symbols)
        while position < count:
            symbol = symbols[position]
            if symbol != self.escape:
                out.append(symbol)
                position += 1
                continue
            position += 1
            literal: list[str] = []
            while position < count and symbols[position] != self.escape:
                literal.append(symbols[position])
                position += 1
            if position >= count:
                return None  # unterminated escape run
            position += 1
            out.append("".join(literal))
        return out

    def expanded_counts(self, sequences: Iterable[Sequence[str]]) -> Counter[str]:
        """Return symbol counts over expanded fitting sequences.

        Every symbol gets a floor of one, so a code can be built over the whole
        alphabet including characters the corpus never used.
        """
        counts: Counter[str] = Counter()
        for sequence in sequences:
            expanded = self.expand(sequence)
            if expanded is not None:
                counts.update(expanded)
        for symbol in self._symbols:
            counts[symbol] = max(counts.get(symbol, 0), 1)
        counts[self.stop] = max(counts.get(self.stop, 1), 1)
        return counts

    def to_ids(self, tokens: Sequence[str]) -> list[int] | None:
        """Expand ``tokens`` and map them to symbol ids."""
        expanded = self.expand(tokens)
        if expanded is None:
            return None
        return [self._index[symbol] for symbol in expanded]

    def from_ids(self, ids: Sequence[int]) -> list[str] | None:
        """Map symbol ids back to tokens, undoing any escaping."""
        try:
            symbols = [self._symbols[i] for i in ids]
        except IndexError:
            return None
        return self.collapse(symbols)
