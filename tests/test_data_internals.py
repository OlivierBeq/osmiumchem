# -*- coding: utf-8 -*-

import csv
import gzip

import corpora as corpora_module
import pytest
from corpora import SourceSpec, _iter_csv_column, _reservoir, build_corpus

from osmiumchem.identity import canonicalise_all, silence_rdkit
from tests.hard_cases import HARD_CASES

silence_rdkit()


def _hard_source(tmp_path):
    """Write the adversarial fixture out as a corpus file."""
    path = tmp_path / "hard.smi"
    path.write_text("\n".join(HARD_CASES) + "\n", encoding="utf-8")
    return [SourceSpec(path)]


def test_reservoir_keeps_everything_when_unlimited():
    assert _reservoir(iter(["a", "b", "", "c"]), None, seed=0) == ["a", "b", "c"]


def test_reservoir_samples_without_bias_towards_the_head():
    # Source corpora are often sorted, so a head slice comes out chemically
    # skewed. Over many seeds a reservoir reaches deep into the stream.
    population = [str(i) for i in range(1000)]
    picked = set()
    for seed in range(40):
        picked.update(_reservoir(iter(population), 10, seed=seed))
    assert len(picked) > 50
    assert any(int(item) > 500 for item in picked)


def test_reservoir_respects_its_limit_and_drops_blanks():
    drawn = _reservoir(iter(["a", "", "b", "c", "d"]), 2, seed=3)
    assert len(drawn) == 2
    assert all(item for item in drawn)


def test_csv_column_falls_back_to_any_smiles_like_header(tmp_path):
    path = tmp_path / "set.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "canonical_smiles"])
        writer.writerow(["1", "CCO"])
        writer.writerow(["2", " c1ccccc1 "])
        writer.writerow(["3", ""])
    # Asked for a column that does not exist, it should still find the SMILES.
    assert list(_iter_csv_column(path, "smiles")) == ["CCO", "c1ccccc1"]
    # And with no column asked for at all.
    assert list(_iter_csv_column(path, None)) == ["CCO", "c1ccccc1"]


def test_csv_column_raises_when_nothing_looks_like_smiles(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    with pytest.raises(KeyError, match="no SMILES-like column"):
        list(_iter_csv_column(path, "smiles"))


def test_lines_reader_takes_the_first_field(tmp_path):
    path = tmp_path / "named.smi"
    path.write_text("CCO ethanol\n\nc1ccccc1\tbenzene\n", encoding="utf-8")
    assert list(SourceSpec(path).stream()) == ["CCO", "c1ccccc1"]


def test_corpus_is_split_and_disjoint(tmp_path):
    corpus = build_corpus(_hard_source(tmp_path), evaluate=None, fit=10, seed=1, label="hard")
    assert corpus.fit
    assert corpus.evaluate
    assert not set(corpus.fit) & set(corpus.evaluate)
    assert "hard" in corpus.summary


def test_corpus_can_fit_on_different_sources(tmp_path):
    # Fitting on a foreign corpus is how transfer across chemical space is
    # measured, and the evaluation split stays untouched.
    donor = tmp_path / "donor.smi"
    donor.write_text("CCO\nCCC\nCCCC\n", encoding="utf-8")
    target = tmp_path / "target.smi"
    target.write_text("c1ccccc1\nCC(=O)O\n", encoding="utf-8")
    corpus = build_corpus(
        [SourceSpec(target)],
        evaluate=2,
        fit=3,
        fit_sources=[SourceSpec(donor)],
    )
    assert set(corpus.fit) == {"CCO", "CCC", "CCCC"}
    assert len(corpus.evaluate) == 2


def test_parallel_canonicalisation_matches_serial():
    molecules = ["CCO", "OCC", "c1ccccc1", "C1=CC=CC=C1", "junk"] * 500
    serial, serial_bad = canonicalise_all(molecules, workers=1)
    parallel, parallel_bad = canonicalise_all(molecules, workers=2)
    assert serial == parallel
    assert len(serial_bad) == len(parallel_bad)


def test_parquet_column_is_read_and_falls_back_on_header(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "set.parquet"
    pq.write_table(
        pa.table({"id": [1, 2, 3], "SMILES": ["CCO", " c1ccccc1 ", ""]}),
        path,
    )
    read = corpora_module._iter_parquet_column
    assert list(read(path, "SMILES")) == ["CCO", "c1ccccc1"]
    # A missing column name should still find the SMILES-like one.
    assert list(read(path, "smiles_col")) == ["CCO", "c1ccccc1"]
    # Distinct narrowing is what makes a huge activity table affordable.
    assert sorted(read(path, "SMILES", distinct=True)) == ["CCO", "c1ccccc1"]


def test_parquet_without_a_smiles_column_raises(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"alpha": [1], "beta": [2]}), path)
    with pytest.raises(KeyError, match="no SMILES-like column"):
        list(corpora_module._iter_parquet_column(path, "SMILES"))


def test_gzipped_sources_are_read_transparently(tmp_path):
    path = tmp_path / "set.smi.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("CCO\nc1ccccc1\n")
    assert list(SourceSpec(path).stream()) == ["CCO", "c1ccccc1"]


def test_build_corpus_caps_the_fitting_split_to_available_data(tmp_path):
    # Asking for more fitting molecules than exist must still leave an
    # evaluation split rather than consuming everything.
    corpus = build_corpus(_hard_source(tmp_path), evaluate=None, fit=10_000, seed=0)
    assert corpus.evaluate
    assert not set(corpus.fit) & set(corpus.evaluate)


def test_aggregate_deduplicates_on_canonical_smiles(tmp_path):
    # The same molecule spelled two ways must become one row, or it could land
    # in the fitting split from one file and the evaluation split from another.
    first = tmp_path / "a.smi"
    first.write_text("CCO\nc1ccccc1\n", encoding="utf-8")
    second = tmp_path / "b.smi"
    second.write_text("OCC\nC1=CC=CC=C1\nCCC\n", encoding="utf-8")
    target = tmp_path / "union.csv"
    manifest = corpora_module.aggregate([SourceSpec(first), SourceSpec(second)], target, workers=1)
    assert manifest["molecules"] == 3
    rows = target.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "smiles"
    assert sorted(rows[1:]) == ["CCC", "CCO", "c1ccccc1"]
    assert target.with_suffix(".csv.manifest.json").is_file()
