# -*- coding: utf-8 -*-

"""Lossless binary encodings of the molecular graph itself.

Where :mod:`~osmiumchem.codecs.entropy` compresses a string describing a
molecule, this module writes the graph straight into bits: atoms, bonds,
charges, isotopes, hydrogen counts and stereochemistry, each in its own field.
The structural counterpart to a learned binary latent space, invertible by
construction rather than by training.

Stereochemistry is the whole difficulty, and where a naive graph serialiser
corrupts molecules without noticing.

RDKit stores tetrahedral chirality as a clockwise or anticlockwise tag whose
meaning depends on the order that atom's bonds happen to be stored in. Permute
the bond list and the same tag means the opposite enantiomer. A decoder
rebuilding the molecule will not reproduce the original ordering, so copying
the tag across flips about half of all stereocentres.

Two things fix it. The decoder adds bonds in a strict order: for each atom
``i`` in increasing order, every neighbour ``j < i`` in increasing order. Every
neighbour below ``i`` lands before any neighbour above it, so each atom's final
bond list comes out sorted by neighbour index, and the decoder's ordering is
fully determined. The encoder then takes the parity of the permutation from
each atom's actual bond order to that sorted order, and flips the tag when the
parity is odd.

The tag in the payload is therefore correct in the decoder's frame and
chirality survives exactly. Double-bond stereochemistry works the same way,
with the two reference atoms stored alongside the descriptor.

The module also wraps RDKit's own binary serialisation, which is lossless but
built for fidelity and speed rather than compactness, as a measure of what an
off-the-shelf binary molecule format costs against a 1000-bit budget.
"""

from __future__ import annotations

import lzma
from collections import Counter
from collections.abc import Sequence

from rdkit import Chem

from ..bits import BitReader, BitWriter
from ..coding.huffman import HuffmanCode
from ..types import Bits
from .base import BaseCodec

__all__ = ["GraphBitsCodec", "RdkitBinaryCodec", "graph_codecs"]

#: Bond orders that survive kekulisation, in payload order.
_BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.DATIVE,
)
_BOND_TYPE_INDEX = {bond: i for i, bond in enumerate(_BOND_TYPES)}

#: Tetrahedral chirality tags, in payload order.
_CHIRAL_TAGS = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
_CHIRAL_INDEX = {tag: i for i, tag in enumerate(_CHIRAL_TAGS)}

#: Double-bond stereo descriptors, in payload order.
_BOND_STEREO = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOANY,
)
_BOND_STEREO_INDEX = {stereo: i for i, stereo in enumerate(_BOND_STEREO)}


def _permutation_parity(actual: Sequence[int], target: Sequence[int]) -> int:
    """Return the parity of the permutation taking ``actual`` onto ``target``.

    :param actual: Neighbour indices in the order the molecule stores them.
    :param target: The same indices in the order the decoder will produce.
    :returns: ``0`` if the permutation is even, ``1`` if odd. An odd
        permutation inverts the meaning of a tetrahedral chirality tag.
    """
    position = {value: i for i, value in enumerate(target)}
    sequence = [position[value] for value in actual]
    seen = [False] * len(sequence)
    parity = 0
    for start in range(len(sequence)):
        if seen[start]:
            continue
        length = 0
        node = start
        while not seen[node]:
            seen[node] = True
            node = sequence[node]
            length += 1
        if length > 1:
            parity ^= (length - 1) & 1
    return parity


class GraphBitsCodec(BaseCodec):
    """Writes the molecular graph into a compact, exactly reversible bitstring.

    :param atom_code: Fitted Huffman code over atomic numbers; built by
        :meth:`fit`.
    """

    name = "graph-bits"
    family = "structural"
    kind = "bitstring"
    size_model = "exact"
    notes = "hand-rolled canonical graph serialisation; lossless including stereochemistry"

    def __init__(self) -> None:
        """Create an unfitted codec."""
        self._atom_code = HuffmanCode(stop_symbol="\x00stop")
        self._fitted = False

    def fit(self, smiles: Sequence[str]) -> None:
        """Fit the atomic-number code on the fitting corpus."""
        counts: Counter[str] = Counter()
        for item in smiles:
            mol = Chem.MolFromSmiles(item)
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                counts[str(atom.GetAtomicNum())] += 1
        # Every element up to bismuth gets a floor, so one missing from the
        # fitting corpus still has a code and the codec stays lossless.
        for number in range(1, 84):
            counts[str(number)] = counts.get(str(number), 0) + 1
        self._atom_code.fit_counts(counts)
        self._fitted = True

    def _prepare(self, smiles: str) -> Chem.Mol | None:
        """Return the molecule in canonical atom order, kekulised.

        Canonical ordering makes the payload independent of how the input
        SMILES was written. Kekulisation drops aromatic bond types, leaving
        every bond a plain order that sanitisation can perceive again.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            order = list(Chem.CanonicalRankAtoms(mol))
            # ``CanonicalRankAtoms`` gives each atom its rank, so inverting it
            # gives the permutation into canonical order.
            permutation = [0] * mol.GetNumAtoms()
            for index, rank in enumerate(order):
                permutation[rank] = index
            mol = Chem.RenumberAtoms(mol, permutation)
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:  # noqa: BLE001 - kekulisation fails on some inputs
            return None
        return mol

    def encode(self, smiles: str) -> Bits | None:
        """Serialise a molecule into bits."""
        if not self._fitted:
            msg = "graph-bits must be fitted before use"
            raise RuntimeError(msg)
        mol = self._prepare(smiles)
        if mol is None:
            return None
        writer = BitWriter()
        count = mol.GetNumAtoms()
        if count == 0:
            return None
        writer.elias_gamma(count)

        try:
            self._write_atoms(writer, mol, self._chirality_tags(mol))
            self._write_bonds(writer, mol)
        except (ValueError, KeyError):
            return None
        return writer.done()

    def _chirality_tags(self, mol: Chem.Mol) -> list[int]:
        """Return each atom's chirality tag, rebased onto the decoder's frame.

        A tetrahedral tag means something only relative to an atom's bond
        order. The decoder produces neighbours sorted by index, so an atom whose
        stored bond order is an odd permutation of that gets its tag flipped
        here. Without the correction about half of all stereocentres invert, and
        the identity test reports a different molecule.
        """
        tags: list[int] = []
        for atom in mol.GetAtoms():
            tag = _CHIRAL_INDEX.get(atom.GetChiralTag(), 0)
            if tag:
                actual = [bond.GetOtherAtomIdx(atom.GetIdx()) for bond in atom.GetBonds()]
                if _permutation_parity(actual, sorted(actual)):
                    tag = 2 if tag == 1 else 1
            tags.append(tag)
        return tags

    def _write_atoms(self, writer: BitWriter, mol: Chem.Mol, chiral: list[int]) -> None:
        """Write one field group per atom."""
        for atom in mol.GetAtoms():
            code, length = self._atom_code._encode_table[str(atom.GetAtomicNum())]  # noqa: SLF001
            writer.uint(code, length)
            # 3 bits of hydrogen count covers 0-7, enough for any organic atom.
            writer.uint(min(7, atom.GetTotalNumHs()), 3)
            charge = atom.GetFormalCharge()
            if charge == 0:
                writer.bit(0)
            else:
                writer.bit(1)
                writer.bit(1 if charge < 0 else 0)
                writer.elias_gamma(min(15, abs(charge)))
            isotope = atom.GetIsotope()
            if isotope == 0:
                writer.bit(0)
            else:
                writer.bit(1)
                writer.elias_gamma(isotope)
            radicals = atom.GetNumRadicalElectrons()
            if radicals == 0:
                writer.bit(0)
            else:
                writer.bit(1)
                writer.uint(min(7, radicals), 3)
            writer.uint(chiral[atom.GetIdx()], 2)

    def _write_bonds(self, writer: BitWriter, mol: Chem.Mol) -> None:
        """Write the adjacency, bond orders and double-bond stereo.

        Chirality tags are corrected once the neighbour ordering the decoder
        will use is known.
        """
        count = mol.GetNumAtoms()
        back: list[list[tuple[int, Chem.Bond]]] = [[] for _ in range(count)]
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            high, low = max(begin, end), min(begin, end)
            back[high].append((low, bond))
        for index in range(count):
            neighbours = sorted(back[index], key=lambda item: item[0])
            writer.elias_gamma(len(neighbours) + 1)
            previous = -1
            for low, bond in neighbours:
                writer.elias_gamma(low - previous)
                previous = low
                writer.uint(_BOND_TYPE_INDEX.get(bond.GetBondType(), 0), 2)
                stereo = _BOND_STEREO_INDEX.get(bond.GetStereo(), 0)
                writer.uint(stereo, 3)
                if stereo:
                    atoms = list(bond.GetStereoAtoms())
                    if len(atoms) == 2:
                        # RDKit pairs the first reference atom with the bond's
                        # begin atom, and the decoder always creates the bond as
                        # ``low -> index``, so a source bond running the other
                        # way needs the pair swapped.
                        if bond.GetBeginAtomIdx() != low:
                            atoms = [atoms[1], atoms[0]]
                        writer.bit(1)
                        writer.elias_gamma(atoms[0] + 1)
                        writer.elias_gamma(atoms[1] + 1)
                    else:
                        writer.bit(0)

    def decode(self, payload: Bits) -> str | None:
        """Rebuild a molecule from bits and return its SMILES."""
        if not isinstance(payload, Bits):
            return None
        reader = BitReader(payload)
        try:
            count = reader.elias_gamma()
            mol = Chem.RWMol()
            chiral: list[int] = []
            for _ in range(count):
                atom, tag = self._read_atom(reader)
                mol.AddAtom(atom)
                chiral.append(tag)
            stereo_jobs = self._read_bonds(reader, mol, count)
        except (EOFError, ValueError, KeyError, RuntimeError):
            return None

        for index, tag in enumerate(chiral):
            if tag:
                mol.GetAtomWithIdx(index).SetChiralTag(_CHIRAL_TAGS[tag])
        for begin, end, stereo, atoms in stereo_jobs:
            bond = mol.GetBondBetweenAtoms(begin, end)
            if bond is None:
                return None
            try:
                if atoms is not None:
                    bond.SetStereoAtoms(atoms[0], atoms[1])
                bond.SetStereo(_BOND_STEREO[stereo])
            except (RuntimeError, ValueError):
                # The reference atoms have to be bonded to this bond's ends,
                # and RDKit raises when a corrupt payload says otherwise.
                return None

        try:
            final = mol.GetMol()
            Chem.SanitizeMol(final)
            if stereo_jobs:
                # The SMILES writer emits ``/`` and ``\`` from the directions
                # of the neighbouring single bonds, not from the E/Z descriptor,
                # so the directions are derived from the descriptors just
                # restored. Without it every double bond comes back
                # unspecified.
                Chem.SetDoubleBondNeighborDirections(final)
            return Chem.MolToSmiles(final, isomericSmiles=True, canonical=True)
        except Exception:  # noqa: BLE001 - a corrupt payload can fail sanitisation
            return None

    def _read_atom(self, reader: BitReader) -> tuple[Chem.Atom, int]:
        """Read one atom's field group."""
        code = 0
        length = 0
        symbol = None
        while symbol is None:
            code = (code << 1) | reader.bit()
            length += 1
            symbol = self._atom_code._decode_table.get((length, code))  # noqa: SLF001
            if length > 64:
                msg = "malformed atomic-number code"
                raise ValueError(msg)
        atom = Chem.Atom(int(symbol))
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(reader.uint(3))
        if reader.bit():
            negative = reader.bit()
            magnitude = reader.elias_gamma()
            atom.SetFormalCharge(-magnitude if negative else magnitude)
        if reader.bit():
            atom.SetIsotope(reader.elias_gamma())
        if reader.bit():
            atom.SetNumRadicalElectrons(reader.uint(3))
        return atom, reader.uint(2)

    def _read_bonds(
        self,
        reader: BitReader,
        mol: Chem.RWMol,
        count: int,
    ) -> list[tuple[int, int, int, tuple[int, int] | None]]:
        """Read the adjacency and queue up double-bond stereo assignments."""
        jobs: list[tuple[int, int, int, tuple[int, int] | None]] = []
        for index in range(count):
            degree = reader.elias_gamma() - 1
            previous = -1
            for _ in range(degree):
                low = previous + reader.elias_gamma()
                previous = low
                bond_type = _BOND_TYPES[reader.uint(2)]
                mol.AddBond(low, index, bond_type)
                stereo = reader.uint(3)
                if stereo:
                    atoms: tuple[int, int] | None = None
                    if reader.bit():
                        first = reader.elias_gamma() - 1
                        second = reader.elias_gamma() - 1
                        atoms = (first, second)
                    jobs.append((low, index, stereo, atoms))
        return jobs


class RdkitBinaryCodec(BaseCodec):
    """RDKit's own binary molecule serialisation, optionally compressed.

    :param compress: Whether to run the pickle through LZMA.
    """

    family = "structural"
    kind = "bitstring"
    size_model = "exact"

    def __init__(self, *, compress: bool = False) -> None:
        """Create the codec."""
        self.compress = compress
        self.name = "rdkit-binary-lzma" if compress else "rdkit-binary"
        self.notes = "RDKit's native serialisation; lossless but built for fidelity, not compactness"

    def encode(self, smiles: str) -> Bits | None:
        """Serialise via RDKit's binary form."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        blob = mol.ToBinary()
        if self.compress:
            blob = lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)
        return Bits.from_bytes(blob)

    def decode(self, payload: Bits) -> str | None:
        """Rebuild from RDKit's binary form."""
        if not isinstance(payload, Bits):
            return None
        blob = payload.to_bytes()
        try:
            if self.compress:
                blob = lzma.decompress(blob)
            mol = Chem.Mol(blob)
        except Exception:  # noqa: BLE001 - corrupt payloads raise from C++
            return None
        try:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Exception:  # noqa: BLE001
            return None


def graph_codecs() -> list[BaseCodec]:
    """Return the structural codec family."""
    return [GraphBitsCodec(), RdkitBinaryCodec(), RdkitBinaryCodec(compress=True)]
