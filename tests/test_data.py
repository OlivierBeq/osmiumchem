# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from corpora import Corpus, SourceSpec, corpus_label, parse_source, read_sources

from osmiumchem.identity import canonical_smiles, canonicalise_all, silence_rdkit
from tests.hard_cases import HARD_CASES

silence_rdkit()


def test_hard_cases_are_all_valid_and_distinct():
    canonical = [canonical_smiles(item) for item in HARD_CASES]
    assert all(item is not None for item in canonical)
    assert len(set(canonical)) == len(canonical)


def test_hard_cases_cover_the_features_that_break_codecs():
    joined = " ".join(HARD_CASES)
    assert "@" in joined  # tetrahedral stereo
    assert "/" in joined  # double-bond stereo
    assert "+" in joined and "-" in joined  # charges
    assert "13C" in joined or "2H" in joined  # isotopes
    assert "." in joined  # multiple components


def test_canonicalise_all_dedupes_and_separates_failures():
    canonical, rejected = canonicalise_all(["CCO", "OCC", "c1ccccc1", "junk", ""])
    assert canonical == ["CCO", "c1ccccc1"]
    assert rejected == ["junk"]


def test_corpus_summary():
    corpus = Corpus(name="demo", fit=["CCO"], evaluate=["c1ccccc1", "CCC"])
    assert len(corpus) == 2
    assert "demo" in corpus.summary


@pytest.mark.parametrize(
    ("text", "path", "column"),
    [
        ("corpus.csv", "corpus.csv", None),
        ("corpus.csv:smiles", "corpus.csv", "smiles"),
        ("corpus.csv:", "corpus.csv", None),
        ("dir/sub/corpus.parquet:SMILES", "dir/sub/corpus.parquet", "SMILES"),
        # A drive letter is not a column.
        (r"C:\corpora\x.csv", r"C:\corpora\x.csv", None),  # illustrative path
    ],
)
def test_parse_source_splits_on_the_last_colon(text, path, column):
    # Compared as paths rather than strings: Windows renders a separator as a
    # backslash, which says nothing about whether the argument parsed right.
    spec = parse_source(text)
    assert spec.path == Path(path)
    assert spec.column == column


def test_parse_source_needs_a_path():
    with pytest.raises(Exception, match="file path"):
        parse_source(":smiles")


@pytest.mark.parametrize(
    ("name", "reader"),
    [("a.csv", "csv"), ("a.tsv", "csv"), ("a.parquet", "parquet"), ("a.smi", "lines"), ("a.smi.gz", "lines")],
)
def test_reader_comes_from_the_suffix(name, reader):
    assert SourceSpec(path=Path(name)).reader == reader


def test_corpus_label_defaults_to_the_first_source(tmp_path):
    one = SourceSpec(tmp_path / "protacs.csv")
    two = SourceSpec(tmp_path / "hiv.csv")
    assert corpus_label([one]) == "protacs"
    assert corpus_label([one, two]) == "protacs+1"
    assert corpus_label([one], "union") == "union"
    assert corpus_label([]) == "(none)"


def test_read_sources_needs_a_source():
    with pytest.raises(ValueError, match="--source"):
        read_sources([])


def test_read_sources_deduplicates_across_files(tmp_path):
    first = tmp_path / "a.smi"
    second = tmp_path / "b.smi"
    first.write_text("CCO\nc1ccccc1\n", encoding="utf-8")
    second.write_text("CCO\nCCC\n", encoding="utf-8")
    molecules = read_sources([SourceSpec(first), SourceSpec(second)])
    assert molecules == ["CCO", "c1ccccc1", "CCC"]


def test_missing_source_says_which_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such source file"):
        list(SourceSpec(tmp_path / "absent.smi").stream())
