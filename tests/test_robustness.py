# -*- coding: utf-8 -*-

"""Corrupt-input handling: a decoder must decline, never invent a molecule."""

import pytest

from osmiumchem.codecs.entropy import (
    ArithmeticStringCodec,
    CompressorCodec,
    FlatCodeCodec,
    HuffmanStringCodec,
    _text_codec,
)
from osmiumchem.codecs.graph import GraphBitsCodec, RdkitBinaryCodec
from osmiumchem.codecs.vectors import FingerprintCodec, Quantizer
from osmiumchem.coding.alphabet import EscapingAlphabet
from osmiumchem.coding.huffman import HuffmanCode
from osmiumchem.coding.model import OrderKModel
from osmiumchem.identity import silence_rdkit
from osmiumchem.types import Bits

silence_rdkit()

FIT = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C[C@H](N)C(=O)O", "C/C=C/C"]


def _fitted(codec):
    codec.fit(FIT)
    return codec


@pytest.mark.parametrize(
    "codec",
    [
        FlatCodeCodec("smiles"),
        HuffmanStringCodec("smiles"),
        ArithmeticStringCodec("smiles", order=2),
        CompressorCodec("smiles", "zlib"),
        GraphBitsCodec(),
        RdkitBinaryCodec(),
    ],
    ids=lambda c: c.name,
)
def test_random_payloads_never_decode_into_a_wrong_molecule_silently(codec):
    # Feeding noise must yield None or a parseable molecule, never a crash.
    _fitted(codec)
    for seed in range(60):
        payload = Bits(seed * 7919 % (1 << 48), 48)
        result = codec.decode(payload)
        assert result is None or isinstance(result, str)


@pytest.mark.parametrize(
    "codec",
    [
        FlatCodeCodec("smiles"),
        HuffmanStringCodec("smiles"),
        ArithmeticStringCodec("smiles", order=2),
        GraphBitsCodec(),
        RdkitBinaryCodec(),
    ],
    ids=lambda c: c.name,
)
def test_truncated_payloads_are_declined(codec):
    _fitted(codec)
    payload = codec.encode("CC(=O)Oc1ccccc1C(=O)O")
    assert payload is not None
    truncated = Bits(payload.value >> 9, max(0, payload.length - 9))
    result = codec.decode(truncated)
    assert result is None or isinstance(result, str)


def test_codecs_decline_payloads_of_the_wrong_type():
    for codec in (FlatCodeCodec("smiles"), HuffmanStringCodec("smiles"), GraphBitsCodec()):
        _fitted(codec)
        assert codec.decode("not-a-bitstring") is None


def test_arithmetic_codec_must_be_fitted_before_use():
    codec = ArithmeticStringCodec("smiles", order=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        codec.encode("CCO")
    with pytest.raises(RuntimeError, match="must be fitted"):
        codec.decode(Bits(0, 8))


def test_arithmetic_codec_refuses_an_empty_fitting_corpus():
    with pytest.raises(ValueError, match="empty corpus"):
        ArithmeticStringCodec("smiles", order=1).fit(["not-a-molecule"])


def test_graph_codec_must_be_fitted_before_use():
    with pytest.raises(RuntimeError, match="must be fitted"):
        GraphBitsCodec().encode("CCO")


def test_unparseable_input_is_declined_by_every_family():
    for codec in (
        FlatCodeCodec("smiles"),
        ArithmeticStringCodec("smiles", order=1),
        CompressorCodec("smiles", "bz2"),
        GraphBitsCodec(),
        RdkitBinaryCodec(compress=True),
    ):
        _fitted(codec)
        assert codec.encode("this-is-not-a-molecule") is None


def test_compressor_declines_corrupt_blobs():
    codec = CompressorCodec("smiles", "lzma")
    assert codec.decode(Bits(0xDEADBEEF, 32)) is None


def test_unsupported_representation_and_algorithm_are_rejected():
    with pytest.raises(ValueError, match="unsupported representation"):
        _text_codec("inchi")
    with pytest.raises(ValueError, match="unsupported algorithm"):
        CompressorCodec("smiles", "brotli")


def test_unknown_fingerprint_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown fingerprint kind"):
        FingerprintCodec("not-a-fingerprint")


def test_fingerprint_declines_unparseable_input():
    codec = FingerprintCodec("morgan", 512)
    assert codec.encode_many(["definitely not a molecule"])[0] is None


def test_quantizer_validates_its_input():
    import numpy as np

    with pytest.raises(ValueError, match="non-empty 2-D matrix"):
        Quantizer().fit(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="non-empty 2-D matrix"):
        Quantizer().fit(np.zeros(4))
    quantizer = Quantizer(bits_per_dim=2, dims=2).fit(np.random.default_rng(0).normal(size=(50, 4)))
    # A payload of the wrong width is not this quantizer's output.
    assert quantizer.decode(Bits(0, 3)) is None


def test_quantizer_survives_a_constant_dimension():
    import numpy as np

    data = np.zeros((20, 3))
    data[:, 0] = np.arange(20)
    quantizer = Quantizer(bits_per_dim=4).fit(data)
    payload = quantizer.encode(data[5])
    assert payload.length == quantizer.total_bits


def test_escaping_alphabet_rejects_non_ascii_and_malformed_runs():
    alphabet = EscapingAlphabet().fit([["C", "O"]])
    # Non-ASCII cannot be escaped, so it is reported rather than mangled.
    assert alphabet.expand(["中"]) is None
    assert alphabet.to_ids(["中"]) is None
    # An unterminated escape run is a corrupt payload.
    assert alphabet.collapse([alphabet.escape, "C"]) is None
    assert alphabet.from_ids([999999]) is None


def test_escaping_alphabet_round_trips_unknown_tokens():
    alphabet = EscapingAlphabet().fit([["C", "O"]])
    tokens = ["C", "[Fe+2]", "O"]
    ids = alphabet.to_ids(tokens)
    assert ids is not None
    assert alphabet.from_ids(ids) == tokens


def test_huffman_handles_a_single_symbol_alphabet():
    code = HuffmanCode().fit_counts({"C": 5})
    assert code.alphabet == ["C"]
    payload = code.encode(["C"])
    assert payload is None  # the stop symbol is outside a one-symbol code


def test_huffman_declines_unknown_symbols_and_corrupt_payloads():
    code = HuffmanCode().fit([["C", "O"]])
    assert code.encode(["N"]) is None
    assert code.cost_bits(["N"]) is None
    assert code.cost_bits(["C"]) > 0
    assert code.decode(Bits(0, 0)) is None


def test_huffman_rejects_an_empty_alphabet():
    with pytest.raises(ValueError, match="empty alphabet"):
        HuffmanCode().fit_counts({})


def test_order_k_model_validates_its_configuration():
    with pytest.raises(ValueError, match="order must be non-negative"):
        OrderKModel(order=-1)
    with pytest.raises(ValueError, match="weight_base must be at least 1"):
        OrderKModel(weight_base=0)
    with pytest.raises(ValueError, match="alphabet size must be positive"):
        OrderKModel().fit([[0]], size=0, stop=0)
    with pytest.raises(ValueError, match="max_total"):
        OrderKModel(max_total=4).fit([[0]], size=10, stop=9)


def test_order_zero_model_ignores_context():
    model = OrderKModel(order=0).fit([[0, 1, 0]], size=3, stop=2)
    assert model.cumulative((0,)) == model.cumulative((1,))
    assert model.cumulative(()) == model.cumulative((0, 1, 0))


def test_graph_codec_round_trips_exotic_atoms_and_bonds():
    # Isotopes, radicals, multi-digit charges and dative bonds each travel in
    # their own bit field, so each one needs exercising.
    codec = GraphBitsCodec()
    codec.fit(FIT)
    from osmiumchem.identity import canonicalise_all, same_molecule

    exotic = canonicalise_all(
        [
            "[13CH4]",
            "[CH3]",  # radical
            "[Al+3]",
            "[O-2]",
            "[2H]O[2H]",
            "[Fe+2].[O-2]",
        ],
    )[0]
    for item in exotic:
        payload = codec.encode(item)
        assert payload is not None, item
        assert same_molecule(item, codec.decode(payload), reference_is_canonical=True), item


def test_graph_codec_declines_an_empty_molecule():
    codec = GraphBitsCodec()
    codec.fit(FIT)
    # A molecule with no atoms carries no graph to serialise.
    assert codec.encode("") is None


def test_flat_code_declines_payloads_that_never_terminate():
    codec = FlatCodeCodec("smiles")
    codec.fit(FIT)
    # All-ones is a run of out-of-range symbol indices with no terminator.
    assert codec.decode(Bits((1 << 64) - 1, 64)) is None


def test_compressor_round_trips_every_algorithm():
    from osmiumchem.identity import same_molecule

    for algorithm in ("zlib", "bz2", "lzma"):
        codec = CompressorCodec("smiles", algorithm)
        codec.fit(FIT)
        payload = codec.encode("CC(=O)Oc1ccccc1C(=O)O")
        assert payload is not None
        assert same_molecule(
            "CC(=O)Oc1ccccc1C(=O)O",
            codec.decode(payload),
        )
        assert codec.decode("not-bits") is None
