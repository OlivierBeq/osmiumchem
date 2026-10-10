# -*- coding: utf-8 -*-

import pytest

from osmiumchem.codecs import SUITES, build_suite, codec_names
from osmiumchem.codecs.base import BaseCodec, StringCodec
from osmiumchem.identity import silence_rdkit
from osmiumchem.types import Bits

silence_rdkit()


@pytest.mark.parametrize("name", ["lossless", "lossy", "fast"])
def test_non_neural_suites_build_and_are_named(name):
    # The neural suite is left out: cheap to construct, but running it needs
    # several hundred megabytes of downloaded model weights.
    codecs = build_suite(name)
    assert codecs
    names = codec_names(codecs)
    assert len(names) == len(set(names)), "codec names must be unique within a suite"
    for codec in codecs:
        assert codec.family
        assert codec.kind in {"string", "bitstring"}


def test_lossless_suite_contains_both_representations_and_the_escape_variant():
    names = set(codec_names(build_suite("lossless")))
    assert "smiles+arith-o4" in names
    assert "selfies+arith-o4" in names
    assert "selfies-hypervalent+arith-o4" in names
    assert "graph-bits" in names


def test_lossy_suite_pairs_fingerprints_with_the_index_control():
    names = set(codec_names(build_suite("lossy")))
    assert "ecfp4-2048" in names
    assert "ecfp4-512-chiral" in names
    # Without the control there is nothing to compare a closed-world
    # fingerprint score against.
    assert "library-index" in names


def test_world_is_propagated_to_lookup_decoded_codecs():
    for codec in build_suite("lossy", world="open"):
        library = getattr(codec, "library", None)
        if library is not None:
            assert library.world == "open"


def test_unknown_suite_is_rejected_with_the_options_listed():
    with pytest.raises(ValueError, match="unknown suite"):
        build_suite("nonsense")
    assert "fast" in SUITES


def test_base_codec_is_abstract_enough_to_catch_mistakes():
    codec = BaseCodec()
    with pytest.raises(NotImplementedError):
        codec.encode("CCO")
    with pytest.raises(NotImplementedError):
        codec.decode(Bits(0, 1))
    described = codec.describe()
    assert set(described) >= {"name", "family", "kind", "size_model", "lossless_by_design"}


def test_base_codec_sizes_bits_and_bytes_but_refuses_nonsense():
    codec = BaseCodec()
    assert codec.payload_bits(Bits(0b101, 3)) == 3
    assert codec.payload_bits(b"ab") == 16
    with pytest.raises(TypeError, match="cannot size a payload"):
        codec.payload_bits(3.5)


def test_string_codec_requires_its_conversions_to_be_implemented():
    codec = StringCodec()
    with pytest.raises(NotImplementedError):
        codec.to_text("CCO")
    with pytest.raises(NotImplementedError):
        codec.from_text("CCO")


def test_string_codec_charges_unseen_tokens_at_a_wider_code():
    from osmiumchem.codecs.strings import SmilesCodec

    codec = SmilesCodec()
    codec.fit(["CCO", "CCC"])
    narrow = codec.payload_bits("CCO")
    # A molecule using tokens outside the fitted alphabet cannot come out
    # cheaper than one that stays inside it.
    wide = codec.payload_bits("[Fe+2]")
    assert narrow > 0
    assert wide >= codec.code_width
    assert codec.utf8_bits("CCO") == 24


def test_string_codec_decode_rejects_non_string_payloads():
    from osmiumchem.codecs.strings import SmilesCodec

    assert SmilesCodec().decode(Bits(0, 4)) is None


def test_entropy_floor_is_reported_for_known_tokens_only():
    import math

    from osmiumchem.codecs.strings import SmilesCodec

    codec = SmilesCodec()
    codec.fit(["CCO"])
    counts = {"C": 2, "O": 1}
    assert codec.entropy_floor_bits("CCO", counts) == pytest.approx(
        -2 * math.log2(2 / 3) - math.log2(1 / 3),
    )
    assert codec.entropy_floor_bits("N", counts) == math.inf
    assert codec.entropy_floor_bits("CCO", {}) == math.inf
