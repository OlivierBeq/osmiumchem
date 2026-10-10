# -*- coding: utf-8 -*-

"""Tests for error paths and accessors not exercised by the main suites."""

import numpy as np
import pytest
from benchmark import CodecResult, run_codec
from corpora import Corpus

from osmiumchem.bits import BitReader, BitWriter
from osmiumchem.codecs.base import BaseCodec
from osmiumchem.codecs.latents import LibraryIndexCodec
from osmiumchem.codecs.strings import (
    DeepSmilesCodec,
    InchiCodec,
    KekuleSmilesCodec,
    SelfiesCodec,
    string_codecs,
)
from osmiumchem.codecs.vectors import (
    BIT_BUDGET,
    FingerprintCodec,
    LatentCodec,
    LibraryDecoder,
    Quantizer,
    injectivity,
)
from osmiumchem.identity import mol_to_canonical, parse, silence_rdkit
from osmiumchem.types import Bits

silence_rdkit()


def test_bits_rejects_non_binary_iterables():
    with pytest.raises(ValueError, match="expected only 0 or 1"):
        Bits.from_iterable([1, 2])


def test_bits_from_empty_sources():
    assert Bits.from_string("") == Bits(0, 0)
    assert Bits.from_bytes(b"") == Bits(0, 0)


def test_writer_and_reader_reject_negative_widths():
    with pytest.raises(ValueError, match="width must be non-negative"):
        BitWriter().uint(0, -1)
    with pytest.raises(ValueError, match="width must be non-negative"):
        BitReader(Bits(0, 4)).uint(-1)
    with pytest.raises(ValueError, match="integers >= 1"):
        BitWriter().elias_delta(0)


def test_reader_reports_how_many_bits_were_missing():
    with pytest.raises(EOFError, match="only 2 remain"):
        BitReader(Bits(0b10, 2)).uint(5)


def test_writer_can_append_an_existing_payload():
    writer = BitWriter().bit(1)
    writer.bits(Bits(0b0110, 4))
    assert str(writer.done()) == "10110"


def test_identity_helpers_on_broken_input():
    assert parse("junk") is None
    assert mol_to_canonical(None) is None
    # Parsed without sanitisation, this carbon has an impossible valence, so
    # canonicalising it reports failure rather than emitting a molecule.
    broken = parse("C(C)(C)(C)(C)(C)C", sanitize=False)
    assert broken is not None
    assert mol_to_canonical(broken) is None


def test_library_index_requires_a_library_and_rejects_strangers():
    codec = LibraryIndexCodec()
    codec.prepare_library(["CCO", "c1ccccc1", "CCC"])
    payload = codec.encode("c1ccccc1")
    assert payload is not None
    assert codec.decode(payload) == "c1ccccc1"
    # Never seen, so it cannot be named.
    assert codec.encode("CCCCCCCC") is None
    assert codec.decode(None) is None
    assert codec.decode(Bits(99, 16)) is None


def test_library_index_cost_grows_logarithmically():
    small = LibraryIndexCodec()
    small.prepare_library([f"C{'C' * i}" for i in range(4)])
    big = LibraryIndexCodec()
    big.prepare_library([f"C{'C' * i}" for i in range(300)])
    assert small.encode("CC").length == 2
    assert big.encode("CC").length == 9


def test_library_decoder_validates_its_world():
    with pytest.raises(ValueError, match="world must be"):
        LibraryDecoder("halfway")
    decoder = LibraryDecoder()
    decoder.add(None, "CCO")
    assert len(decoder) == 0


def test_injectivity_of_an_empty_set():
    stats = injectivity([])
    assert stats["molecules"] == 0
    assert stats["ceiling"] == 0.0


def test_fingerprint_payload_sizes_follow_their_configuration():
    assert FingerprintCodec("maccs", 167).payload_size == 167
    assert FingerprintCodec("morgan", 512).payload_size == 512
    assert FingerprintCodec("morgan", 512, counted=True).payload_size == 2048
    # A refused molecule still reports what the encoding would have cost.
    assert FingerprintCodec("morgan", 512).payload_bits(None) == 512


def test_fingerprint_names_encode_their_configuration():
    assert FingerprintCodec("morgan", 1024, radius=3).name == "ecfp6-1024"
    assert FingerprintCodec("morgan", 512, chiral=True).name == "ecfp4-512-chiral"
    assert FingerprintCodec("maccs", 167).name == "maccs-167"
    assert FingerprintCodec("rdkit", 512).name == "rdkit-512"


def test_budget_constant_matches_the_stated_problem():
    assert BIT_BUDGET == 1000


def test_latent_codec_reports_native_cost_and_declines_non_finite():
    def encoder(smiles):
        out = np.tile(np.arange(8, dtype=np.float32), (len(smiles), 1))
        out += np.arange(len(smiles), dtype=np.float32)[:, None]
        return out

    codec = LatentCodec(encoder, "toy", bits_per_dim=2, dims=4)
    assert codec.native_bits is None  # nothing embedded yet
    codec.fit(["CCO", "CCC", "CCCC"])
    assert codec.native_bits == 8 * 32
    assert codec.quantizer.total_bits == 8
    payload = codec.encode("CCO")
    assert payload is not None and payload.length == 8
    assert codec.payload_bits(None) == 8
    assert codec.payload_for(np.array([np.nan] * 8)) is None


def test_latent_codec_handles_a_single_vector_response():
    def encoder(smiles):
        return np.arange(4, dtype=np.float32)

    codec = LatentCodec(encoder, "flat", bits_per_dim=1)
    codec.fit(["CCO"])
    assert codec.vectors(["CCO"]).ndim == 2


def test_quantizer_keeps_the_highest_variance_dimensions():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200, 5)) * np.array([0.01, 10.0, 0.02, 5.0, 0.03])
    quantizer = Quantizer(bits_per_dim=4, dims=2).fit(data)
    assert quantizer.total_bits == 8
    recovered = quantizer.decode(quantizer.encode(data[0]))
    assert recovered.shape == (2,)


def test_string_codec_family_includes_the_optional_members():
    names = {codec.name for codec in string_codecs()}
    assert {"smiles", "smiles-kekule", "selfies", "selfies-hypervalent", "inchi"} <= names


def test_kekule_codec_declines_what_cannot_be_kekulised():
    codec = KekuleSmilesCodec()
    assert codec.to_text("not-a-molecule") is None


def test_selfies_and_inchi_decline_broken_payloads():
    assert SelfiesCodec().from_text("[[[not selfies") is None
    assert InchiCodec().from_text("InChI=1S/garbage") is None
    assert InchiCodec().to_text("not-a-molecule") is None


def test_deepsmiles_declines_broken_payloads():
    codec = DeepSmilesCodec()
    assert codec.to_text("not-a-molecule") is None
    assert codec.from_text("))))((((") is None


def test_codec_result_handles_the_empty_case():
    empty = CodecResult(name="x", family="t")
    assert empty.accuracy == 0.0
    assert empty.median_bits is None
    assert empty.max_bits is None
    assert empty.within_budget() is None
    assert empty.bits_per_molecule is None
    assert empty.percentile(0.5) is None


def test_run_codec_survives_a_decoder_that_raises():
    class Exploding(BaseCodec):
        name = "exploding"
        kind = "bitstring"

        def encode(self, smiles):
            return Bits(1, 8)

        def decode(self, payload):
            msg = "decoder blew up"
            raise RuntimeError(msg)

    result = run_codec(Exploding(), Corpus(name="u", fit=["CCO"], evaluate=["CCO"]))
    assert result.error is not None
    assert "decoder blew up" in result.error
