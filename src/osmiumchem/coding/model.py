# -*- coding: utf-8 -*-

"""Token patterns and the order-k context model behind the arithmetic coder.

The model is static: fitted once on a training split, then frozen, which is how
a trained decoder is deployed. Its weights are a shared artefact shipped with
the decoder, and only the payload is charged against a molecule's bit budget.
The fitting split is disjoint from the evaluation set. An adaptive model that
learned from the molecule it was coding would pay for its own learning curve on
every molecule, which is a bad trade on strings this short.

Frequencies are blended across orders rather than escaped between them. For a
context of ``k`` preceding symbols, a candidate symbol's count is a weighted sum
of its counts under every suffix length from ``0`` to ``k``, with longer
contexts weighted exponentially higher::

    raw[s] = 1 + sum_j weight_base**j * count_j(context[-j:], s)

Cruder than full PPM, which escapes from a long context down to a shorter one
through its own escape symbol, but simple, free of exclusion bookkeeping, and
integer-only and deterministic, so encoder and decoder cannot drift apart and
the bits reproduce across machines. The additive ``1`` keeps every symbol at a
non-zero frequency, which the arithmetic coder requires.

This class deals in probabilities over integer symbol ids and nothing else.
Which symbols exist, and what happens to one the fitting corpus never
contained, is :class:`~osmiumchem.coding.alphabet.EscapingAlphabet`'s business.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

__all__ = ["OrderKModel"]

#: Matches one SMILES token: a bracket atom, a two-letter element, or a single
#: character. Order matters, since the alternatives are tried left to right.
_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]"  # bracket atoms, e.g. [nH], [C@@H], [13C], [Na+]
    r"|Br|Cl"  # two-letter organic-subset elements
    r"|%\d\d"  # two-digit ring closures, e.g. %10
    r"|.)",  # everything else, one character at a time
)

#: Matches one SELFIES token: a bracketed group, or any single character.
#:
#: The trailing ``.`` alternative is load-bearing. SELFIES separates the
#: fragments of a salt or co-crystal with a literal ``.``, so a pattern matching
#: only bracketed groups would drop the separator and fuse ``[Na+1].[Cl-1]``
#: into one species. Every tokeniser here partitions its input exactly.
_SELFIES_TOKEN_RE = re.compile(r"(\[[^\]]+\]|.)")


class OrderKModel:
    """A frozen, integer-only, order-``k`` blended context model.

    Works on integer symbol ids alone, leaving which symbols exist and how an
    unseen one is escaped to
    :class:`~osmiumchem.coding.alphabet.EscapingAlphabet`.

    :param order: Longest context length to condition on; ``0`` gives a
        unigram model.
    :param weight_base: Multiplier favouring longer contexts, so counts under a
        context of length ``j`` are weighted ``weight_base ** j``.
    :param max_total: Ceiling on the frequency total handed to the arithmetic
        coder. Smaller totals keep its big integers small, and the codec fast,
        at a slight cost in coding efficiency.
    """

    __slots__ = ("_cache", "_counts", "max_total", "order", "size", "stop", "weight_base")

    def __init__(self, order: int = 3, weight_base: int = 24, max_total: int = 1 << 14) -> None:
        """Create an unfitted model."""
        if order < 0:
            msg = f"order must be non-negative, got {order}"
            raise ValueError(msg)
        if weight_base < 1:
            msg = f"weight_base must be at least 1, got {weight_base}"
            raise ValueError(msg)
        self.order = order
        self.weight_base = weight_base
        self.max_total = max_total
        self._counts: list[dict[tuple[int, ...], dict[int, int]]] = []
        self._cache: dict[tuple[int, ...], tuple[list[int], int]] = {}
        self.size = 0
        self.stop = 0

    def fit(self, sequences: Iterable[Sequence[int]], size: int, stop: int) -> OrderKModel:
        """Fit the model on id sequences.

        :param sequences: Symbol id lists, one per fitting molecule, *without*
            the stop symbol, which is appended internally.
        :param size: Alphabet size, including the stop symbol.
        :param stop: Id of the stop symbol.
        :returns: ``self``, for chaining.
        """
        if size < 1:
            msg = f"alphabet size must be positive, got {size}"
            raise ValueError(msg)
        if self.max_total < size:
            msg = f"max_total ({self.max_total}) must be at least the alphabet size ({size})"
            raise ValueError(msg)
        self.size = size
        self.stop = stop
        counts: list[dict[tuple[int, ...], dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(self.order + 1)
        ]
        for sequence in sequences:
            ids = [*sequence, stop]
            for position, symbol in enumerate(ids):
                for j in range(min(self.order, position) + 1):
                    context = tuple(ids[position - j : position])
                    counts[j][context][symbol] += 1
        self._counts = [{k: dict(v) for k, v in level.items()} for level in counts]
        self._cache.clear()
        return self

    def cumulative(self, prefix: tuple[int, ...]) -> tuple[list[int], int]:
        """Return cumulative frequencies for the symbol following ``prefix``.

        Only the last :attr:`order` symbols matter, so the prefix is truncated
        and the answer memoised. Encoder and decoder read the same table for
        the same context, which is what keeps them in step.
        """
        context = prefix[-self.order :] if self.order else ()
        cached = self._cache.get(context)
        if cached is not None:
            return cached

        raw = [1] * self.size
        weight = 1
        for j in range(min(self.order, len(context)) + 1):
            level = self._counts[j].get(context[len(context) - j :] if j else ())
            if level:
                for symbol, count in level.items():
                    raw[symbol] += weight * count
            weight *= self.weight_base

        total = sum(raw)
        if total > self.max_total:
            # Integer-only rescaling, then back up to the floor of 1 that the
            # arithmetic coder relies on.
            factor = -(-total // self.max_total)
            raw = [max(1, value // factor) for value in raw]

        cum = [0]
        for value in raw:
            cum.append(cum[-1] + value)
        result = (cum, cum[-1])
        self._cache[context] = result
        return result
