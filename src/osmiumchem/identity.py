# -*- coding: utf-8 -*-

"""What it means for a decoded molecule to be *the same* molecule.

A 99.99% accuracy figure says nothing unless "same" is pinned down, so every
comparison in the benchmark goes through here.

The reference form is RDKit's canonical **isomeric** SMILES of the sanitised
molecule. Isomeric keeps stereochemistry, formal charges and isotopes in the
comparison, so dropping a stereocentre is an error rather than a saving.
Canonical makes the comparison blind to atom ordering, so a codec may emit
atoms in any order.

Two choices set the difficulty bar. Hydrogens are compared implicitly, after
RDKit's default sanitisation has folded their counts into the heavy atoms, so
explicit-H spellings need not survive verbatim. Tautomers and resonance forms
are not unified, since unifying them would excuse real information loss.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rdkit import Chem, RDLogger

__all__ = [
    "canonical_smiles",
    "canonicalise_all",
    "parse",
    "same_molecule",
    "silence_rdkit",
]


def silence_rdkit() -> None:
    """Suppress RDKit's C++ logging.

    A failing decoder produces malformed strings, and RDKit warns on every one
    of them. The results table records those failures instead.
    """
    RDLogger.DisableLog("rdApp.*")


def parse(smiles: str | None, *, sanitize: bool = True) -> Chem.Mol | None:
    """Parse a SMILES string into a molecule, returning ``None`` on failure.

    :param smiles: The string to parse; ``None`` and empty strings yield
        ``None``.
    :param sanitize: Whether to run RDKit sanitisation (valence checks,
        aromaticity perception). Turn it off only to inspect a broken result.
    :returns: The molecule, or ``None`` if the string is not valid.
    """
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles, sanitize=sanitize)
    except Exception:  # noqa: BLE001 - RDKit can raise a variety of C++ exceptions
        return None


def canonical_smiles(smiles: str | None) -> str | None:
    """Return the canonical isomeric SMILES of ``smiles``, or ``None``.

    The benchmark's normal form: inputs pass through it before encoding, and
    decoded molecules before comparison.

    :param smiles: Any valid SMILES string.
    :returns: Canonical isomeric SMILES, or ``None`` if parsing failed.
    """
    mol = parse(smiles)
    if mol is None:
        return None
    return mol_to_canonical(mol)


def mol_to_canonical(mol: Chem.Mol | None) -> str | None:
    """Return the canonical isomeric SMILES of a molecule, or ``None``.

    Takes a built molecule, which is what the structural codecs produce
    without going through a string first.

    :param mol: The molecule. An unsanitised one is sanitised here, and
        failure yields ``None``.
    """
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:  # noqa: BLE001 - sanitisation failures are expected for broken decodes
        return None


def same_molecule(
    reference: str | None,
    candidate: str | None,
    *,
    reference_is_canonical: bool = False,
) -> bool:
    """Return whether two SMILES denote the same molecule.

    Both sides are canonicalised first, so atom ordering and aromatic/Kekulé
    spelling do not matter, while stereochemistry, charge and isotopes do.

    :param reference: The expected molecule.
    :param candidate: Whatever the decoder returned.
    :param reference_is_canonical: Set when ``reference`` is already valid
        canonical SMILES, letting an exact string match short-circuit the
        comparison. Without it two copies of the same *invalid* string would
        compare equal, so the default re-parses both sides. The benchmark
        canonicalises up front and passes ``True``.
    :returns: ``True`` only if both parse and share a canonical form.
    """
    if reference is None or candidate is None:
        return False
    if reference_is_canonical and reference == candidate:
        return True
    left = canonical_smiles(reference)
    right = canonical_smiles(candidate)
    if left is None or right is None:
        return False
    return left == right


def canonicalise_all(smiles: Iterable[str], workers: int = 1) -> tuple[list[str], list[str]]:
    """Canonicalise many SMILES, returning the successes and the failures.

    Fitting a codec on a corpus starts here, so that no codec is penalised for
    a spelling difference it inherited from its input.

    :param smiles: Raw input strings.
    :param workers: Processes to use; ``1`` stays in-process.
    :returns: ``(canonical, rejected)``, deduplicated with order preserved.
    """
    silence_rdkit()
    # Blank entries are missing data rather than unparseable molecules, so
    # dropping them keeps the reject count meaningful.
    raw = [item for item in smiles if item and item.strip()]
    if workers > 1 and len(raw) > 2000:
        results = _canonicalise_parallel(raw, workers)
    else:
        results = [canonical_smiles(item) for item in raw]
    seen: set[str] = set()
    canonical: list[str] = []
    rejected: list[str] = []
    for source, result in zip(raw, results, strict=True):
        if result is None:
            rejected.append(source)
        elif result not in seen:
            seen.add(result)
            canonical.append(result)
    return canonical, rejected


def _canonicalise_parallel(raw: Sequence[str], workers: int) -> list[str | None]:
    """Canonicalise ``raw`` across ``workers`` processes, preserving order."""
    from concurrent.futures import ProcessPoolExecutor

    chunk = max(500, len(raw) // (workers * 8) + 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=silence_rdkit) as pool:
        return list(pool.map(canonical_smiles, raw, chunksize=chunk))
