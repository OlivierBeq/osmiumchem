# -*- coding: utf-8 -*-

import pytest
from benchmark import (
    BenchmarkReport,
    CodecResult,
    classify_failure,
    run_benchmark,
    run_codec,
    wilson_interval,
)
from corpora import Corpus

from osmiumchem.codecs.entropy import ArithmeticStringCodec
from osmiumchem.codecs.strings import InchiCodec, SmilesCodec
from osmiumchem.identity import silence_rdkit

silence_rdkit()

MOLECULES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "C[C@H](N)C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Clc1ccccc1",
    "C/C=C/C",
]


def _corpus():
    return Corpus(name="unit", fit=list(MOLECULES), evaluate=list(MOLECULES))


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson_interval(99, 100)
    assert 0.0 < low < 0.99 < high <= 1.0
    # A perfect small sample must still admit a meaningful error rate.
    low, high = wilson_interval(100, 100)
    assert low < 1.0
    assert high == pytest.approx(1.0)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_perfect_small_run_cannot_certify_four_nines():
    # 100/100 is not evidence of 99.99%; the lower bound must fall short.
    result = CodecResult(name="x", family="t", molecules=100, successes=100)
    assert result.accuracy == 1.0
    assert not result.supports_target(0.9999)


def test_large_perfect_run_does_certify_four_nines():
    result = CodecResult(name="x", family="t", molecules=200_000, successes=200_000)
    assert result.supports_target(0.9999)


def test_result_bit_statistics():
    result = CodecResult(name="x", family="t", molecules=4, successes=4, bits=[10, 20, 30, 4000])
    assert result.median_bits in (20, 30)
    assert result.max_bits == 4000
    assert result.within_budget(1000) == pytest.approx(0.75)
    assert result.bits_per_molecule == pytest.approx(1015.0)


@pytest.mark.parametrize(
    "reference,decoded,expected",
    [
        ("CCO", None, "no-output"),
        ("CCO", "not-a-smiles", "unparseable"),
        ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "stereo-only"),
        ("CCO", "CCC", "formula-changed"),
        ("CCO", "COC", "formula-kept"),
    ],
)
def test_classify_failure(reference, decoded, expected):
    assert classify_failure(reference, decoded) == expected


def test_run_codec_scores_a_lossless_codec_perfectly():
    result = run_codec(SmilesCodec(), _corpus())
    assert result.molecules == len(MOLECULES)
    assert result.successes == len(MOLECULES)
    assert result.accuracy == 1.0
    assert result.error is None


def test_payload_sizes_are_recorded_even_when_decoding_fails():
    # A codec whose decoder always fails still has to report what its payload
    # costs, or the budget columns go blank for the lossy codecs whose size is
    # the interesting part.
    class NeverDecodes(SmilesCodec):
        name = "never-decodes"

        def decode(self, payload):
            return None

    result = run_codec(NeverDecodes(), _corpus())
    assert result.successes == 0
    assert len(result.bits) == len(MOLECULES)
    assert result.median_bits is not None
    assert result.within_budget(1000) is not None


def test_run_codec_records_lossy_failures_with_causes():
    result = run_codec(InchiCodec(), _corpus())
    assert result.molecules == len(MOLECULES)
    if result.successes < result.molecules:
        assert sum(result.failures.values()) == result.molecules - result.successes
        assert result.examples


def test_run_codec_reports_a_broken_codec_without_raising():
    class Broken(SmilesCodec):
        name = "broken"

        def fit(self, smiles):
            msg = "deliberate"
            raise RuntimeError(msg)

    result = run_codec(Broken(), _corpus())
    assert result.error is not None
    assert "deliberate" in result.error


def test_run_benchmark_collects_every_codec():
    codecs = [SmilesCodec(), ArithmeticStringCodec("smiles", order=2)]
    report = run_benchmark(codecs, _corpus())
    assert isinstance(report, BenchmarkReport)
    assert [item.name for item in report.results] == [codec.name for codec in codecs]
    assert report.molecules == len(MOLECULES)


def test_sorted_by_bits_puts_the_cheapest_first():
    report = run_benchmark([SmilesCodec(), ArithmeticStringCodec("smiles", order=2)], _corpus())
    ordered = report.sorted_by_bits()
    sizes = [item.median_bits for item in ordered if item.median_bits is not None]
    assert sizes == sorted(sizes)


def test_ceiling_is_not_reported_for_string_codecs():
    # A textual payload has no injectivity ceiling, and 0% would read as "every
    # molecule collides" rather than "not applicable".
    result = run_codec(SmilesCodec(), _corpus())
    assert result.ceiling is None


def test_ceiling_is_reported_for_bitstring_codecs():
    result = run_codec(ArithmeticStringCodec("smiles", order=2), _corpus())
    assert result.ceiling == pytest.approx(1.0)


def test_strict_budget_scores_oversized_payloads_as_failures():
    # A payload that decodes perfectly but overflows the budget is not a
    # solution to the problem as posed, so strict mode gives it no credit.
    codec = SmilesCodec()
    corpus = _corpus()
    lenient = run_codec(codec, corpus, budget=1000)
    assert lenient.accuracy == 1.0
    strict = run_codec(SmilesCodec(), corpus, budget=40, strict_budget=True)
    assert strict.accuracy < 1.0
    assert strict.failures["over-budget"] > 0
    # The size distribution is unchanged; only the scoring differs.
    assert strict.bits == lenient.bits


def test_strict_budget_is_a_no_op_when_everything_fits():
    corpus = _corpus()
    strict = run_codec(SmilesCodec(), corpus, budget=10_000, strict_budget=True)
    assert strict.accuracy == 1.0
    assert "over-budget" not in strict.failures
