# -*- coding: utf-8 -*-

import pytest

from osmiumchem.identity import canonical_smiles, parse, same_molecule, silence_rdkit

silence_rdkit()


@pytest.mark.parametrize(
    "left,right",
    [
        ("c1ccccc1", "C1=CC=CC=C1"),  # aromatic vs Kekule spelling
        ("[Na+].[Cl-]", "[Cl-].[Na+]"),  # component order
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("OCC", "CCO"),  # atom order
    ],
)
def test_equivalent_spellings_are_the_same_molecule(left, right):
    assert same_molecule(left, right)


@pytest.mark.parametrize(
    "left,right",
    [
        ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"),  # enantiomers
        ("C/C=C/C", r"C/C=C\C"),  # E vs Z
        ("CC[13CH3]", "CCC"),  # isotope dropped
        ("C[N+](C)(C)C", "CN(C)C"),  # charge dropped
        ("CCO", "CCC"),  # different molecule
    ],
)
def test_differences_that_must_not_be_forgiven(left, right):
    assert not same_molecule(left, right)


def test_none_and_garbage_never_match():
    assert not same_molecule(None, "C")
    assert not same_molecule("C", None)
    assert not same_molecule("not-a-smiles", "C")


def test_identical_invalid_strings_do_not_match_by_default():
    # Without the canonical precondition both sides must be re-parsed, so two
    # copies of the same invalid string must not be mistaken for a match.
    assert not same_molecule("not-a-smiles", "not-a-smiles")


def test_canonical_precondition_enables_fast_path():
    assert same_molecule("c1ccccc1", "c1ccccc1", reference_is_canonical=True)


def test_canonical_smiles_round_trips_and_rejects_junk():
    assert canonical_smiles("C1=CC=CC=C1") == "c1ccccc1"
    assert canonical_smiles("junk") is None
    assert canonical_smiles("") is None
    assert canonical_smiles(None) is None


def test_parse():
    assert parse("CCO") is not None
    assert parse("junk") is None
    assert parse(None) is None
