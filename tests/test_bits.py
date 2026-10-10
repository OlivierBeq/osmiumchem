# -*- coding: utf-8 -*-

import random

import pytest

from osmiumchem.bits import BitReader, BitWriter, elias_delta_cost, elias_gamma_cost
from osmiumchem.types import Bits, flat_code_width


def test_bits_tracks_leading_zeros():
    # Leading zeros are data: these two payloads must stay distinguishable.
    assert Bits.from_string("0101") != Bits.from_string("101")
    assert len(Bits.from_string("0101")) == 4
    assert str(Bits.from_string("0001")) == "0001"


def test_bits_rejects_overflow():
    with pytest.raises(ValueError):
        Bits(0b1111, 2)
    with pytest.raises(ValueError):
        Bits(-1, 4)
    with pytest.raises(ValueError):
        Bits(0, -1)


def test_bits_bytes_round_trip():
    payload = Bits.from_bytes(b"molecule")
    assert payload.length == 64
    assert payload.to_bytes() == b"molecule"


def test_bits_iteration_is_msb_first():
    assert list(Bits.from_string("1001")) == [1, 0, 0, 1]
    assert Bits.from_iterable([1, 0, 0, 1]) == Bits.from_string("1001")


def test_empty_payload():
    empty = Bits(0, 0)
    assert len(empty) == 0
    assert str(empty) == ""
    assert empty.to_bytes() == b""


@pytest.mark.parametrize("size,expected", [(1, 0), (2, 1), (3, 2), (45, 6), (1024, 10)])
def test_flat_code_width(size, expected):
    assert flat_code_width(size) == expected


def test_elias_costs_match_written_width():
    for value in (1, 2, 3, 17, 255, 10**6):
        writer = BitWriter()
        writer.elias_gamma(value)
        assert len(writer) == elias_gamma_cost(value)
        writer = BitWriter()
        writer.elias_delta(value)
        assert len(writer) == elias_delta_cost(value)


def test_mixed_field_round_trip_fuzz():
    rng = random.Random(20260409)
    for _ in range(400):
        operations = []
        writer = BitWriter()
        for _ in range(rng.randint(1, 12)):
            kind = rng.choice(["bit", "uint", "gamma", "delta"])
            if kind == "bit":
                value = rng.randint(0, 1)
                writer.bit(value)
                operations.append((kind, value, None))
            elif kind == "uint":
                width = rng.randint(0, 24)
                value = rng.randint(0, (1 << width) - 1) if width else 0
                writer.uint(value, width)
                operations.append((kind, value, width))
            else:
                value = rng.randint(1, 10**6)
                getattr(writer, f"elias_{kind}")(value)
                operations.append((kind, value, None))
        reader = BitReader(writer.done())
        for kind, value, width in operations:
            if kind == "bit":
                assert reader.bit() == value
            elif kind == "uint":
                assert reader.uint(width) == value
            else:
                assert getattr(reader, f"elias_{kind}")() == value
        assert reader.remaining == 0


def test_reader_raises_past_end():
    reader = BitReader(Bits.from_string("1"))
    assert reader.bit() == 1
    with pytest.raises(EOFError):
        reader.bit()


def test_writer_rejects_bad_uint():
    with pytest.raises(ValueError):
        BitWriter().uint(5, 2)
    with pytest.raises(ValueError):
        BitWriter().uint(-1, 4)
    with pytest.raises(ValueError):
        BitWriter().elias_gamma(0)
