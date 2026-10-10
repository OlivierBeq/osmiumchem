# -*- coding: utf-8 -*-

"""Molecules chosen to break careless codecs.

A test fixture rather than a corpus: small enough to read, and every entry
exercises a feature a representation can quietly drop.
"""

from __future__ import annotations

#: Molecules chosen to break careless codecs. Each exercises a feature a
#: representation can quietly drop: tetrahedral and double-bond stereo, formal
#: charges, isotopes, radicals, dative and metal bonds, multi-component salts,
#: tautomer-sensitive aromatic heterocycles, ring-closure numbering past nine.
#: Used by the unit tests, where one failure is visible.
HARD_CASES: tuple[str, ...] = (
    "C",  # methane: the degenerate smallest case
    "O=C=O",
    "c1ccccc1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine, aromatic N heterocycle
    "C[C@H](N)C(=O)O",  # L-alanine, tetrahedral stereo
    "C[C@@H](N)C(=O)O",  # D-alanine, the mirror image
    "C/C=C/C",  # E double-bond stereo
    r"C/C=C\C",  # Z double-bond stereo
    "[Na+].[Cl-]",  # two components
    "[NH4+].[O-]S(=O)(=O)[O-].[NH4+]",  # charges and stoichiometry
    "CC[13CH3]",  # isotope
    "[2H]C([2H])([2H])O",  # deuterated methanol
    "C[N+](C)(C)CCO",  # quaternary ammonium
    "[O-][N+](=O)c1ccccc1",  # nitro written as charge-separated
    "c1ccc2c(c1)ccc1c2cccc1",  # fused aromatics
    "C1CCCCCCCCCCCCCCCCCC1",  # macrocycle, ring closure reuse
    "O=C(O)c1ccccc1O",  # salicylic acid
    "[Fe+2]",  # bare metal ion
    "[13C@H](F)(Cl)Br",  # isotope plus stereo on the same atom
    "C1=CC2=CC3=CC=CC=C3C=C2C=C1",
    "N[C@@H](C)C(=O)N[C@H](CC1=CC=CC=C1)C(=O)O",  # dipeptide, two stereocentres
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose, five stereocentres
    "C%10CCCCCCCCCC%10",  # two-digit ring closure
    "[S@](=O)(C)CC",  # sulfoxide stereo
    "F[P-](F)(F)(F)(F)F.[Li+]",  # hypervalent anion plus counterion
)
