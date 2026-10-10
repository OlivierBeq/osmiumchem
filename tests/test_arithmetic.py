# -*- coding: utf-8 -*-

import hashlib
import math
import random

import pytest

from osmiumchem.coding.arithmetic import ArithmeticCodec


class SkewedModel:
    """Deterministic, prefix-dependent, deliberately lopsided frequencies."""

    def __init__(self, size, seed=0, max_total=1 << 16):
        self.size = size
        self.stop = size - 1
        self.seed = seed
        self.max_total = max_total

    def cumulative(self, prefix):
        digest = hashlib.blake2b(repr((self.seed, prefix)).encode(), digest_size=16).digest()
        rng = random.Random(digest)
        step = max(1, self.max_total // self.size)
        freqs = [1 + (rng.randrange(step) if rng.random() < 0.5 else 0) for _ in range(self.size)]
        cumulative = [0]
        for freq in freqs:
            cumulative.append(cumulative[-1] + freq)
        return cumulative, cumulative[-1]


def test_round_trip_fuzz():
    rng = random.Random(4242)
    for trial in range(400):
        size = rng.randint(2, 48)
        codec = ArithmeticCodec(SkewedModel(size, seed=trial))
        symbols = [rng.randrange(size - 1) for _ in range(rng.randint(0, 60))]
        assert codec.decode(codec.encode(symbols)) == symbols


def test_empty_message():
    codec = ArithmeticCodec(SkewedModel(4, seed=1))
    assert codec.decode(codec.encode([])) == []


def test_reaches_entropy_within_two_bits():
    class Biased:
        size = 3
        stop = 2

        def cumulative(self, prefix):
            return ([0, 900, 999, 1000], 1000)

    codec = ArithmeticCodec(Biased())
    sequence = [0] * 900 + [1] * 99
    payload = codec.encode(sequence)
    ideal = -(900 * math.log2(0.9) + 99 * math.log2(0.099) + math.log2(0.001))
    # Arithmetic coding should land within a bit or two of the Shannon bound.
    assert payload.length <= ideal + 2
    assert codec.decode(payload) == sequence


def test_near_certain_message_is_almost_free():
    class Certain:
        size = 2
        stop = 1

        def cumulative(self, prefix):
            if len(prefix) < 8:
                return ([0, 999_999, 1_000_000], 1_000_000)
            return ([0, 1, 1_000_000], 1_000_000)

    codec = ArithmeticCodec(Certain())
    payload = codec.encode([0] * 8)
    assert payload.length <= 4
    assert codec.decode(payload) == [0] * 8


def test_rejects_out_of_range_symbol():
    codec = ArithmeticCodec(SkewedModel(4, seed=2))
    with pytest.raises(ValueError):
        codec.encode([99])


def test_decode_terminates_on_corrupt_payload():
    codec = ArithmeticCodec(SkewedModel(6, seed=3))
    payload = codec.encode([1, 2, 3])
    with pytest.raises(ValueError):
        codec.decode(payload, max_symbols=1)
