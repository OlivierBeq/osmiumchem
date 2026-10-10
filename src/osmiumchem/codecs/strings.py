# -*- coding: utf-8 -*-

"""Textual molecular representations: SMILES, SELFIES, DeepSMILES and InChI.

The reference points, each with its own failure profile.

Canonical SMILES is a bijection with the molecule as far as the identity test
goes, the identity test *being* canonical SMILES. It should round-trip
perfectly and acts as the control: anything below 100% is a harness bug rather
than chemistry.

Kekulé SMILES drops the lower-case aromatic spelling. Also perfect, at a
slightly different token count, which is what isolates the saving from aromatic
notation.

SELFIES is robust in the generative direction, every string decoding to some
molecule, which is not the same as being lossless. Its strings run longer in
tokens than SMILES, trading bits for that robustness.

DeepSMILES drops ring-closure digits and branch-matching parentheses, the two
syntaxes that make SMILES brittle for generators. The question here is whether
it round-trips.

InChI is a canonical identifier rather than a representation meant to be
decoded. It normalises mobile hydrogens and some stereochemistry, so it fails
the identity test on a measurable fraction of molecules, and it is here to show
that canonical does not imply invertible.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem

from ..coding.model import _SELFIES_TOKEN_RE, _SMILES_TOKEN_RE
from .base import StringCodec

__all__ = [
    "DeepSmilesCodec",
    "InchiCodec",
    "KekuleSmilesCodec",
    "SelfiesCodec",
    "SmilesCodec",
    "string_codecs",
]


class SmilesCodec(StringCodec):
    """Canonical isomeric SMILES, tokenised into chemical tokens."""

    name = "smiles"
    family = "string"
    notes = "RDKit canonical isomeric SMILES; the benchmark's reference form"

    def tokenize(self, text: str) -> list[str]:
        """Split SMILES into bracket atoms, two-letter elements and symbols."""
        return _SMILES_TOKEN_RE.findall(text)

    def to_text(self, smiles: str) -> str | None:
        """Return the canonical SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    def from_text(self, text: str) -> str | None:
        """Return the SMILES unchanged, since the identity test canonicalises."""
        return text


class KekuleSmilesCodec(SmilesCodec):
    """Kekulé SMILES, with aromaticity written as alternating bonds."""

    name = "smiles-kekule"
    notes = "Kekulised SMILES; isolates the token cost of aromatic notation"

    def to_text(self, smiles: str) -> str | None:
        """Return the Kekulé SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:  # noqa: BLE001 - some inputs cannot be kekulised
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, kekuleSmiles=True)


#: Which SELFIES semantic-constraint preset is currently installed.
#:
#: ``selfies`` keeps its valence rules in module-global state, so two codecs
#: wanting different presets would fight over it. Each entry point below
#: installs its preset immediately before calling the library, and the cache is
#: keyed by preset, so a value computed under one rule set is never served
#: under another.
_ACTIVE_PRESET: str | None = None


def _apply_preset(preset: str) -> None:
    """Install a SELFIES semantic-constraint preset, if not already active."""
    global _ACTIVE_PRESET  # noqa: PLW0603 - mirrors selfies' own global state
    if _ACTIVE_PRESET != preset:
        import selfies

        selfies.set_semantic_constraints(preset)
        _ACTIVE_PRESET = preset


@lru_cache(maxsize=1 << 18)
def _encode_selfies(smiles: str, preset: str) -> str | None:
    """Return the SELFIES form of ``smiles``, or ``None`` if unsupported.

    The slowest conversion in the benchmark, and the whole SELFIES ladder
    (flat, Huffman, five arithmetic orders) converts the same fitting corpus.
    Uncached, a 100000-molecule fit split would be converted seven times over.
    """
    import selfies

    _apply_preset(preset)
    try:
        return selfies.encoder(smiles)
    except Exception:  # noqa: BLE001 - selfies raises several custom errors
        return None


class SelfiesCodec(StringCodec):
    """SELFIES, the self-referencing embedded string representation.

    Every SELFIES string decodes to a valid molecule, which is not the same as
    every molecule having a SELFIES string. Under the default valence rules the
    encoder refuses hypervalent species such as ``Nc1ccc([IH]c2ccc(N)cc2)cc1``,
    raising instead of returning. That costs about one molecule in twenty
    thousand of a drug-like corpus, enough on its own to hold SELFIES below a
    99.99% target.

    It is a configuration choice rather than a limitation: the ``hypervalent``
    preset widens the valence rules and encodes those molecules losslessly at
    no cost elsewhere. Both presets are offered so the report shows the
    difference.

    :param constraints: A ``selfies`` semantic-constraint preset, one of
        ``"default"``, ``"hypervalent"`` or ``"octet_rule"``.
    """

    family = "string"

    def __init__(self, constraints: str = "default") -> None:
        """Create the codec under a given constraint preset."""
        super().__init__()
        self.constraints = constraints
        self.name = "selfies" if constraints == "default" else f"selfies-{constraints}"
        self.notes = (
            "SELFIES v2, default valence rules; refuses to encode hypervalent species"
            if constraints == "default"
            else f"SELFIES v2 with the {constraints} valence preset"
        )

    def tokenize(self, text: str) -> list[str]:
        """Split SELFIES into its bracketed tokens."""
        return _SELFIES_TOKEN_RE.findall(text)

    def to_text(self, smiles: str) -> str | None:
        """Return the SELFIES string, or ``None`` if encoding is unsupported."""
        return _encode_selfies(smiles, self.constraints)

    def from_text(self, text: str) -> str | None:
        """Return the SMILES decoded from SELFIES."""
        import selfies

        _apply_preset(self.constraints)
        try:
            return selfies.decoder(text)
        except Exception:  # noqa: BLE001 - selfies raises several custom errors
            return None


class DeepSmilesCodec(StringCodec):
    """DeepSMILES, with ring closures and branches rewritten."""

    name = "deepsmiles"
    family = "string"
    notes = "DeepSMILES (rings+branches); removes the two brittle SMILES syntaxes"

    def __init__(self) -> None:
        """Create the codec and its converter."""
        super().__init__()
        import deepsmiles

        self._converter = deepsmiles.Converter(rings=True, branches=True)

    def tokenize(self, text: str) -> list[str]:
        """Split DeepSMILES using the SMILES token rules."""
        return _SMILES_TOKEN_RE.findall(text)

    def to_text(self, smiles: str) -> str | None:
        """Return the DeepSMILES string.

        The input is parsed first. The DeepSMILES converter is a syntactic
        rewrite that will transform a string which is not a molecule at all, so
        without the check this codec would accept input every other string
        codec rejects.
        """
        if Chem.MolFromSmiles(smiles) is None:
            return None
        try:
            return self._converter.encode(smiles)
        except Exception:  # noqa: BLE001 - converter raises on unsupported input
            return None

    def from_text(self, text: str) -> str | None:
        """Return the SMILES decoded from DeepSMILES."""
        try:
            return self._converter.decode(text)
        except Exception:  # noqa: BLE001 - malformed strings raise DecodeError
            return None


class InchiCodec(StringCodec):
    """InChI, a canonical identifier that is not designed to be decoded."""

    name = "inchi"
    family = "string"
    lossless_by_design = False
    notes = "canonical identifier; normalises mobile H and some stereo, so it loses molecules"

    def to_text(self, smiles: str) -> str | None:
        """Return the InChI string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            inchi = Chem.MolToInchi(mol)
        except Exception:  # noqa: BLE001 - the InChI library raises on odd valences
            return None
        return inchi or None

    def from_text(self, text: str) -> str | None:
        """Return the SMILES recovered from InChI."""
        try:
            mol = Chem.MolFromInchi(text)
        except Exception:  # noqa: BLE001 - malformed InChI raises
            return None
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def string_codecs() -> list[StringCodec]:
    """Return one instance of every available textual codec.

    DeepSMILES is optional, so a missing package skips it rather than failing
    the benchmark.
    """
    codecs: list[StringCodec] = [
        SmilesCodec(),
        KekuleSmilesCodec(),
        SelfiesCodec(),
        SelfiesCodec("hypervalent"),
    ]
    try:
        codecs.append(DeepSmilesCodec())
    except ImportError:  # pragma: no cover - optional dependency
        pass
    codecs.append(InchiCodec())
    return codecs
