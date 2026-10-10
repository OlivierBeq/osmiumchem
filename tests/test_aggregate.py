# -*- coding: utf-8 -*-

"""Tests for aggregating several corpora into one file of unique molecules."""

import json

import pytest
from corpora import SourceSpec, aggregate

from osmiumchem.identity import silence_rdkit

silence_rdkit()


def _csv_source(tmp_path, name, rows, column="smiles"):
    """Write a one-column CSV and return a spec pointing at it."""
    path = tmp_path / f"{name}.csv"
    path.write_text(f"id,{column}\n" + "".join(f"{i},{row}\n" for i, row in enumerate(rows)), encoding="utf-8")
    return SourceSpec(path, column)


def _parquet_source(tmp_path, name, rows, column="SMILES"):
    """Write a one-column Parquet file and return a spec pointing at it."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table({column: rows}), path)
    return SourceSpec(path, column)


def test_aggregate_writes_one_row_per_molecule(tmp_path):
    first = _csv_source(tmp_path, "first", ["CCO", "c1ccccc1", "CCO"])
    second = _parquet_source(tmp_path, "second", ["CCN", "c1ccccc1"])
    target = tmp_path / "union.csv"

    manifest = aggregate([first, second], target, workers=1, chunk=2)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "smiles"
    molecules = lines[1:]
    assert sorted(molecules) == sorted({"CCO", "c1ccccc1", "CCN"})
    assert len(molecules) == len(set(molecules))
    assert manifest["molecules"] == 3


def test_a_molecule_written_two_ways_is_one_molecule(tmp_path):
    # Why canonicalisation comes before deduplication: these three strings are
    # one compound, and holding it three times would let a codec be fitted on a
    # molecule it is then scored on.
    source = _csv_source(tmp_path, "spellings", ["OCC", "CCO", "C(C)O"])
    target = tmp_path / "union.csv"
    manifest = aggregate([source], target, workers=1)
    assert manifest["molecules"] == 1
    assert target.read_text(encoding="utf-8").splitlines()[1:] == ["CCO"]


def test_unparseable_strings_are_counted_not_written(tmp_path):
    source = _csv_source(tmp_path, "mixed", ["CCO", "not-a-molecule", "c1ccccc1"])
    manifest = aggregate([source], tmp_path / "union.csv", workers=1)
    assert manifest["molecules"] == 2
    assert manifest["rejected"] == 1
    assert manifest["sources"][0]["values_read"] == 3


def test_the_manifest_records_what_each_source_contributed(tmp_path):
    first = _csv_source(tmp_path, "first", ["CCO", "c1ccccc1"])
    second = _parquet_source(tmp_path, "second", ["CCO", "CCN"])
    target = tmp_path / "union.csv"

    manifest = aggregate([first, second], target, workers=1, log=lambda line: None)

    stored = json.loads(target.with_suffix(target.suffix + ".manifest.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert [record["source"] for record in manifest["sources"]] == ["first.csv", "second.parquet"]
    # The second source repeats a molecule the first contributed, so it adds
    # one and the overlap stays visible in the manifest.
    assert manifest["sources"][0]["molecules_added"] == 2
    assert manifest["sources"][1]["values_read"] == 2
    assert manifest["sources"][1]["molecules_added"] == 1


def test_the_manifest_separates_source_rows_from_values_read(tmp_path):
    # A Parquet batch is narrowed to its distinct values before anything
    # crosses into Python, so the strings seen are not the rows. Reporting only
    # the former understates a large activity table by hundreds of millions.
    source = _parquet_source(tmp_path, "repeats", ["CCO"] * 40 + ["CCN"] * 60)
    manifest = aggregate([source], tmp_path / "union.csv", workers=1)
    record = manifest["sources"][0]
    assert record["source_rows"] == 100
    assert record["values_read"] == 2
    assert record["molecules_added"] == 2


def test_a_csv_source_reports_no_row_count(tmp_path):
    # Finding it would mean reading the file twice, which is not worth it.
    source = _csv_source(tmp_path, "plain", ["CCO", "CCN"])
    manifest = aggregate([source], tmp_path / "union.csv", workers=1)
    assert manifest["sources"][0]["source_rows"] is None
    assert manifest["sources"][0]["values_read"] == 2


def test_progress_is_reported_per_source(tmp_path):
    lines = []
    source = _csv_source(tmp_path, "first", ["CCO"])
    aggregate([source], tmp_path / "union.csv", workers=1, log=lines.append)
    assert any("first.csv" in line for line in lines)
    assert any("unique molecules" in line for line in lines)


def test_nothing_partial_is_left_behind(tmp_path):
    target = tmp_path / "union.csv"
    aggregate([_csv_source(tmp_path, "first", ["CCO"])], target, workers=1)
    assert target.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_chunking_does_not_change_the_result(tmp_path):
    rows = ["CCO", "CCN", "c1ccccc1", "OCC", "CC(=O)O", "CCO"]
    source = _csv_source(tmp_path, "rows", rows)
    one = aggregate([source], tmp_path / "a.csv", workers=1, chunk=1)
    many = aggregate([source], tmp_path / "b.csv", workers=1, chunk=1000)
    assert one["molecules"] == many["molecules"]
    assert (tmp_path / "a.csv").read_text() == (tmp_path / "b.csv").read_text()


def test_the_aggregate_reads_back_as_one_source(tmp_path):
    # The workflow it exists for: aggregate once, then pass the result as a
    # single --source with no canonicalisation left to redo.
    from corpora import build_corpus

    source = _csv_source(tmp_path, "first", ["CCO", "c1ccccc1", "CCN", "CC(=O)O"])
    target = tmp_path / "union.csv"
    aggregate([source], target, workers=1)

    corpus = build_corpus([SourceSpec(target, "smiles")], evaluate=2, fit=2, label="union")
    assert len(corpus.evaluate) == 2
    assert not set(corpus.fit) & set(corpus.evaluate)
    assert corpus.name == "union"


def test_parquet_batches_are_narrowed_before_crossing_into_python(tmp_path):
    # An activity table repeats a compound per measurement, so hundreds of
    # millions of rows would mean as many throwaway Python strings.
    from corpora import _iter_parquet_column

    source = _parquet_source(tmp_path, "repeats", ["CCO"] * 50 + ["CCN"] * 50)
    assert len(list(_iter_parquet_column(source.path, "SMILES"))) == 100
    assert list(_iter_parquet_column(source.path, "SMILES", distinct=True)) == ["CCO", "CCN"]
    assert list(source.stream(distinct=True)) == ["CCO", "CCN"]
