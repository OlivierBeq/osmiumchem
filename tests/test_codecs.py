# -*- coding: utf-8 -*-

import pytest

from osmiumchem.codecs.entropy import entropy_codecs
from osmiumchem.codecs.graph import graph_codecs
from osmiumchem.codecs.strings import (
    KekuleSmilesCodec,
    SelfiesCodec,
    SmilesCodec,
    string_codecs,
)
from osmiumchem.codecs.vectors import (
    LibraryDecoder,
    Quantizer,
    fingerprint_codecs,
    injectivity,
)
from osmiumchem.identity import canonicalise_all, same_molecule, silence_rdkit
from osmiumchem.types import Bits
from tests.hard_cases import HARD_CASES

silence_rdkit()

HARD, _ = canonicalise_all(HARD_CASES)
# A small but chemically varied fitting corpus, kept clear of the hard cases so
# that escape handling is exercised rather than bypassed.
FIT = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "C[C@H](N)C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Clc1ccccc1",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "C/C=C/C",
    "[Na+].[Cl-]",
    "c1ccc2[nH]ccc2c1",
    "O=S(=O)(N)c1ccccc1",
]


def _round_trips(codec, molecules):
    """Return the molecules a codec fails to reproduce exactly."""
    failures = []
    for item in molecules:
        payload = codec.encode(item)
        decoded = codec.decode(payload) if payload is not None else None
        if payload is None or not same_molecule(item, decoded, reference_is_canonical=True):
            failures.append((item, decoded))
    return failures


LOSSLESS = [*entropy_codecs(orders=(0, 2)), *graph_codecs()]


@pytest.mark.parametrize("codec", LOSSLESS, ids=lambda c: c.name)
def test_lossless_codecs_reproduce_hard_cases(codec):
    # These codecs claim exact reversibility, so even one failure is a bug.
    codec.fit(FIT)
    assert _round_trips(codec, HARD) == []


@pytest.mark.parametrize("codec", LOSSLESS, ids=lambda c: c.name)
def test_lossless_codecs_handle_unseen_tokens(codec):
    # Tokens absent from the fitting corpus must be escaped, not dropped.
    codec.fit(["CCO", "c1ccccc1"])
    unusual = canonicalise_all(["C[SH](=O)(O)N=CCC(=O)N1CCCC1", "[13C@H](F)(Cl)Br", "[Fe+2]"])[0]
    assert _round_trips(codec, unusual) == []


def test_string_codecs_are_fitted_and_sized():
    for codec in string_codecs():
        codec.fit(FIT)
        payload = codec.encode("CCO")
        assert payload is None or isinstance(payload, str)
        if payload is not None:
            assert codec.payload_bits(payload) > 0


@pytest.mark.parametrize("codec", [SmilesCodec(), KekuleSmilesCodec(), SelfiesCodec()], ids=lambda c: c.name)
def test_reference_string_codecs_are_lossless_on_hard_cases(codec):
    codec.fit(FIT)
    assert _round_trips(codec, HARD) == []


def test_selfies_tokenizer_preserves_component_separators():
    # A SELFIES tokeniser that drops '.' fuses a salt into one connected
    # molecule, the worst kind of bug here.
    codec = SelfiesCodec()
    text = codec.to_text("[Na+].[Cl-]")
    assert text is not None
    assert "".join(codec.tokenize(text)) == text
    assert "." in codec.tokenize(text)


def test_graph_bits_preserves_both_kinds_of_stereo():
    (codec,) = [c for c in graph_codecs() if c.name == "graph-bits"]
    codec.fit(FIT)
    stereo = canonicalise_all(
        [
            "C[C@H](N)C(=O)O",
            "C[C@@H](N)C(=O)O",
            "C/C=C/C",
            r"C/C=C\C",
            "N[C@@H](C)C(=O)N[C@H](CC1=CC=CC=C1)C(=O)O",
        ],
    )[0]
    assert _round_trips(codec, stereo) == []
    # And the two enantiomers must not collide onto one payload.
    assert codec.encode(stereo[0]) != codec.encode(stereo[1])


def test_fingerprints_are_sized_as_declared():
    for codec in fingerprint_codecs():
        payloads = codec.encode_many(["CCO", "c1ccccc1"])
        for payload in payloads:
            assert payload is not None
            assert payload.length == codec.payload_size


def test_injectivity_detects_collisions():
    payloads = [Bits(1, 8), Bits(1, 8), Bits(2, 8), None]
    stats = injectivity(payloads)
    assert stats["molecules"] == 4
    assert stats["collided"] == 2
    assert stats["encode_failures"] == 1
    assert stats["ceiling"] == pytest.approx(0.25)


def test_library_decoder_refuses_ambiguous_payloads():
    library = LibraryDecoder()
    library.add(Bits(5, 8), "CCO")
    assert library.decode(Bits(5, 8)) == "CCO"
    # A second, different molecule on the same payload is a genuine collision,
    # so the decoder must decline rather than guess.
    library.add(Bits(5, 8), "CCC")
    assert library.decode(Bits(5, 8)) is None
    assert library.decode(Bits(9, 8)) is None


def test_quantizer_charges_exactly_what_it_claims():
    import numpy as np

    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 64))
    quantizer = Quantizer(bits_per_dim=3, dims=16).fit(data)
    assert quantizer.total_bits == 48
    payload = quantizer.encode(data[0])
    assert payload.length == 48
    recovered = quantizer.decode(payload)
    assert recovered is not None
    assert recovered.shape == (16,)


def test_quantizer_needs_fitting():
    import numpy as np

    with pytest.raises(RuntimeError):
        Quantizer().encode(np.zeros(4))


def test_embedding_cache_computes_each_molecule_once():
    import numpy as np

    from osmiumchem.codecs.latents import _cached, _EmbeddingCache

    calls = []

    def compute(smiles):
        calls.append(list(smiles))
        return np.arange(len(smiles) * 4, dtype=np.float32).reshape(len(smiles), 4)

    cache = _EmbeddingCache()
    first = _cached(cache, compute, ["CCO", "CCC"])
    assert first.shape == (2, 4)
    # A repeat request, and a request with one new molecule, must only compute
    # what is genuinely missing.
    second = _cached(cache, compute, ["CCO", "CCC"])
    assert np.array_equal(first, second)
    _cached(cache, compute, ["CCO", "c1ccccc1"])
    assert calls == [["CCO", "CCC"], ["c1ccccc1"]]


def test_embedding_cache_evicts_past_its_bound():
    import numpy as np

    from osmiumchem.codecs.latents import _EmbeddingCache

    cache = _EmbeddingCache(max_floats=8)
    cache.put(["a", "b"], np.zeros((2, 4), dtype=np.float32))
    assert cache.stored_floats == 8
    cache.put(["c"], np.zeros((1, 4), dtype=np.float32))
    # Oldest entry is dropped to stay within the bound.
    assert cache.stored_floats <= 8
    assert cache.get_missing(["a"]) == ["a"]


def test_arithmetic_payloads_are_deterministic_across_instances():
    # The coder is integer-only on purpose, so two independently fitted
    # instances emit bit-identical payloads. Any drift here makes an encoded
    # archive unreadable by a differently-built decoder.
    from osmiumchem.codecs.entropy import ArithmeticStringCodec

    targets = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C[C@@H](N)C(=O)O", "[13C@H](F)(Cl)Br"]
    first = ArithmeticStringCodec("smiles", order=3)
    first.fit(FIT)
    second = ArithmeticStringCodec("smiles", order=3)
    second.fit(FIT)
    for item in targets:
        left = first.encode(item)
        right = second.encode(item)
        assert left == right
        assert (left.value, left.length) == (right.value, right.length)


def test_escaped_token_costs_more_but_stays_lossless():
    # An out-of-alphabet token must round trip, at a price.
    from osmiumchem.codecs.entropy import ArithmeticStringCodec

    codec = ArithmeticStringCodec("smiles", order=3)
    codec.fit(FIT)
    ordinary = codec.encode("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    escaped = codec.encode("[13C@H](F)(Cl)Br")
    assert ordinary is not None and escaped is not None
    assert escaped.length > ordinary.length
    assert _round_trips(codec, ["[13C@H](F)(Cl)Br"]) == []


def test_selfies_hypervalent_preset_encodes_what_the_default_refuses():
    # Default valence rules refuse hypervalent iodine, enough on its own to
    # keep SELFIES under a 99.99% target on a drug-like corpus.
    stubborn = "Nc1ccc([IH]c2ccc(N)cc2)cc1"
    default = SelfiesCodec()
    widened = SelfiesCodec("hypervalent")
    assert default.to_text(stubborn) is None
    assert widened.to_text(stubborn) is not None
    assert _round_trips(widened, canonicalise_all([stubborn])[0]) == []
    # Widening must not break anything that already worked.
    widened.fit(FIT)
    assert _round_trips(widened, HARD) == []


def test_selfies_presets_do_not_leak_through_the_shared_cache():
    # selfies keeps valence rules in module-global state, so a value cached
    # under one preset must never be served under another.
    stubborn = "Nc1ccc([IH]c2ccc(N)cc2)cc1"
    widened = SelfiesCodec("hypervalent")
    default = SelfiesCodec()
    assert widened.to_text(stubborn) is not None
    assert default.to_text(stubborn) is None
    assert widened.to_text(stubborn) is not None


def test_chirality_flag_decides_whether_epimers_collide():
    from osmiumchem.codecs.vectors import FingerprintCodec

    # Same constitution, one differs only by a stereocentre.
    pair = [
        "O=C(NC1CC(C(=O)O)C12CCOCC2)OCc1ccccc1",
        "O=C(N[C@H]1CC(C(=O)O)C12CCOCC2)OCc1ccccc1",
    ]
    blind = FingerprintCodec("morgan", 2048, radius=2)
    aware = FingerprintCodec("morgan", 2048, radius=2, chiral=True)
    blind_payloads = blind.encode_many(pair)
    aware_payloads = aware.encode_many(pair)
    # Stereo-blind: identical payloads, so no decoder could ever separate them.
    assert blind_payloads[0] == blind_payloads[1]
    assert aware_payloads[0] != aware_payloads[1]
    # And the fix costs nothing in bits.
    assert aware_payloads[0].length == blind_payloads[0].length
