# -*- coding: utf-8 -*-

"""Vector representations: circular fingerprints and learned latent spaces.

The half of the benchmark that is not lossless by construction, and where the
bit budget starts to bite.

The fingerprints are Morgan/ECFP circular fingerprints, MACCS keys, RDKit path
fingerprints and atom pairs. They are hashes, and folding an unbounded space of
substructures into a fixed bit vector is many-to-one by design, so they are
expected to fail; the question is by how much and why. The latents are learned
continuous embeddings (CDDD, GROVER) quantised to a bit budget. A float vector
has no bit length until you say how a coordinate is stored, and this module
says so.

Float-to-bit accounting
-----------------------

A ``d``-dimensional float32 latent takes ``32 * d`` bits, so CDDD's 512
dimensions are 16384 bits, sixteen times the 1000-bit ceiling, before accuracy
enters the picture. Something has to give: fewer dimensions, fewer bits per
dimension, or both. :class:`Quantizer` charges exactly ``bits_per_dim * dims``
bits and reports the native cost beside it.

The injectivity ceiling
-----------------------

A poor result from a lossy code invites blaming the decoder, and that excuse
can be removed. When two molecules share a payload, no decoder can separate
them, however it was fitted, because the information is gone. The fraction of
evaluation molecules with a unique payload is therefore an upper bound on the
accuracy of any decoder, and it costs no training to compute.

:func:`injectivity` reports that ceiling, which answers whether a custom
decoder fitted to a latent space could reach 100%: below 99.99%, no amount of
decoder capacity helps. :class:`LibraryDecoder` then attains the ceiling by
construction, so measured accuracy and bound coincide and the comparison with
the lossless codecs is like for like.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from ..types import Bits
from .base import BaseCodec

__all__ = [
    "FingerprintCodec",
    "LatentCodec",
    "LibraryDecoder",
    "Quantizer",
    "fingerprint_codecs",
    "injectivity",
]

#: The benchmark's hard ceiling on payload size, in bits.
BIT_BUDGET = 1000


def injectivity(payloads: Sequence[Bits | None]) -> dict[str, float | int]:
    """Return how uniquely a code identifies the molecules it was computed on.

    :param payloads: One payload per molecule, in evaluation order; ``None``
        entries count as encoding failures.
    :returns: A mapping with the number of molecules, the number of distinct
        payloads, how many molecules share a payload with a different molecule,
        and ``ceiling``, the fraction with a unique payload. No decoder can
        exceed ``ceiling``.
    """
    total = len(payloads)
    keys = [(p.length, p.value) if p is not None else None for p in payloads]
    failures = sum(1 for key in keys if key is None)
    counts: dict[tuple[int, int], int] = {}
    for key in keys:
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    unique = sum(1 for key in keys if key is not None and counts[key] == 1)
    return {
        "molecules": total,
        "distinct_payloads": len(counts),
        "collided": total - failures - unique,
        "encode_failures": failures,
        "ceiling": unique / total if total else 0.0,
    }


#: Factories that rebuild an encoder from its name, keyed by the name
#: :class:`LatentCodec` reports it under.
#:
#: An encoder is a closure over a weight loader and an embedding cache, and
#: closures do not pickle. Its name determines it completely, so a saved codec
#: drops the closure and rebuilds it on load, which is what makes a latent
#: codebook storable. :mod:`osmiumchem.codecs.latents` registers the entries on
#: import, so any process that can build these codecs has them.
_ENCODER_FACTORIES: dict[str, Callable[[], Callable[[Sequence[str]], np.ndarray]]] = {}


def register_encoder(name: str, factory: Callable[[], Callable[[Sequence[str]], np.ndarray]]) -> None:
    """Register ``factory`` as the way to rebuild the encoder called ``name``."""
    _ENCODER_FACTORIES[name] = factory


def build_encoder(name: str) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a fresh encoder by name.

    :raises KeyError: If nothing registered that name, which means the module
        defining the encoder was not imported.
    """
    factory = _ENCODER_FACTORIES.get(name)
    if factory is None:
        known = ", ".join(sorted(_ENCODER_FACTORIES)) or "none registered"
        msg = f"no encoder registered as {name!r}; known encoders are {known}"
        raise KeyError(msg)
    return factory()


class LibraryDecoder:
    """Decodes a payload by looking it up among known molecules.

    The most generous decoder a lossy code can have, since it is handed the
    answer key. It fails only on a real collision, where one payload maps to
    several molecules and the payload itself cannot say which, so its accuracy
    is exactly the injectivity ceiling.

    It cannot produce a molecule outside its library, so it measures
    identification rather than reconstruction, and a decoder that has to
    synthesise a molecule from the payload alone, which is what an application
    needs, can only do worse. Both views are reported: ``closed`` puts the
    evaluation molecules in the library, ``open`` only the fitting split, and
    the gap is the work the library is doing rather than the code.

    A concrete example of how misleading a closed-world score gets: sixteen
    physicochemical descriptors (molecular weight, logP, TPSA, ring counts and
    so on) at sixteen bits each come to 256 bits and identify every molecule in
    a 3000-molecule evaluation set uniquely, scoring 100% closed-world. Nobody
    reconstructs a molecule from its molecular weight and logP. That 100% is
    the library answering, not the code, which is why ``open`` is the default
    world and why :class:`~osmiumchem.codecs.latents.LibraryIndexCodec` is
    reported alongside as the control.
    """

    __slots__ = ("_table", "world")

    def __init__(self, world: str = "closed") -> None:
        """Create an empty decoder.

        :param world: ``"closed"`` if the evaluation molecules are added to the
            library, ``"open"`` if only the fitting split is.
        """
        if world not in {"closed", "open"}:
            msg = f"world must be 'closed' or 'open', got {world!r}"
            raise ValueError(msg)
        self.world = world
        self._table: dict[tuple[int, int], str | None] = {}

    def __len__(self) -> int:
        """Return the number of payloads in the library."""
        return len(self._table)

    def add(self, payload: Bits | None, smiles: str) -> None:
        """Register ``smiles`` under ``payload``.

        A payload already mapped to a different molecule is marked ambiguous
        and decodes to ``None``, rather than returning whichever molecule was
        inserted first.
        """
        if payload is None:
            return
        key = (payload.length, payload.value)
        existing = self._table.get(key, ...)
        if existing is ...:
            self._table[key] = smiles
        elif existing != smiles:
            self._table[key] = None

    def decode(self, payload: Bits | None) -> str | None:
        """Return the unique molecule for ``payload``, if there is one."""
        if payload is None:
            return None
        return self._table.get((payload.length, payload.value))


@dataclass(slots=True)
class Quantizer:
    """Maps a float vector to a fixed number of bits, and back.

    Each retained dimension is clipped to a fitted range and split into
    ``2 ** bits_per_dim`` uniform levels. The range comes from quantiles rather
    than extremes, so one outlier molecule cannot flatten the scale for the
    rest.

    :param bits_per_dim: Bits spent per retained dimension.
    :param dims: Dimensions to retain, highest-variance first; ``None`` keeps
        all of them.
    :param lower_quantile: Lower clipping quantile.
    :param upper_quantile: Upper clipping quantile.
    """

    bits_per_dim: int = 1
    dims: int | None = None
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    _keep: np.ndarray | None = None
    _low: np.ndarray | None = None
    _span: np.ndarray | None = None

    @property
    def total_bits(self) -> int:
        """Return the exact payload size in bits."""
        if self._keep is None:
            return 0
        return int(self.bits_per_dim * self._keep.size)

    def fit(self, matrix: np.ndarray) -> Quantizer:
        """Fit the retained dimensions and their ranges.

        :param matrix: ``(n_molecules, n_dims)`` float array from the fitting
            split.
        :returns: ``self``, for chaining.
        """
        data = np.asarray(matrix, dtype=np.float64)
        if data.ndim != 2 or data.size == 0:
            msg = f"expected a non-empty 2-D matrix, got shape {data.shape}"
            raise ValueError(msg)
        variance = data.var(axis=0)
        keep_count = data.shape[1] if self.dims is None else min(self.dims, data.shape[1])
        # Highest-variance dimensions carry the most information per bit, with
        # ties broken by index to keep the choice reproducible.
        order = np.lexsort((np.arange(data.shape[1]), -variance))[:keep_count]
        self._keep = np.sort(order)
        kept = data[:, self._keep]
        low = np.quantile(kept, self.lower_quantile, axis=0)
        high = np.quantile(kept, self.upper_quantile, axis=0)
        span = high - low
        # A constant dimension carries no information and would divide by zero.
        span[span <= 0] = 1.0
        self._low = low
        self._span = span
        return self

    def encode(self, vector: np.ndarray) -> Bits:
        """Quantise one vector into a payload."""
        if self._keep is None or self._low is None or self._span is None:
            msg = "quantizer must be fitted before use"
            raise RuntimeError(msg)
        levels = (1 << self.bits_per_dim) - 1
        kept = np.asarray(vector, dtype=np.float64)[self._keep]
        scaled = (kept - self._low) / self._span
        codes = np.clip(np.rint(scaled * levels), 0, levels).astype(np.int64)
        value = 0
        for code in codes:
            value = (value << self.bits_per_dim) | int(code)
        return Bits(value, self.total_bits)

    def decode(self, payload: Bits) -> np.ndarray | None:
        """Return the de-quantised vector, each level mapped to its bin centre.

        Useful only to a decoder that consumes the vector again, such as a
        generative one. The original float is unrecoverable by construction.
        """
        if self._keep is None or self._low is None or self._span is None:
            msg = "quantizer must be fitted before use"
            raise RuntimeError(msg)
        if payload.length != self.total_bits:
            return None
        levels = (1 << self.bits_per_dim) - 1
        mask = (1 << self.bits_per_dim) - 1
        codes = np.empty(self._keep.size, dtype=np.int64)
        value = payload.value
        for position in range(self._keep.size - 1, -1, -1):
            codes[position] = value & mask
            value >>= self.bits_per_dim
        return self._low + (codes / levels) * self._span


class _VectorCodec(BaseCodec):
    """Shared behaviour for codecs whose payload comes from a vector.

    Vectors are computed in batches, which is far faster for both
    fingerprinting and neural inference, and cached so that filling the library
    and scoring a molecule do not compute them twice.
    """

    kind = "bitstring"
    size_model = "exact"

    def __init__(self, world: str = "closed") -> None:
        """Create the codec with an empty library."""
        self._library = LibraryDecoder(world)
        self._cache: dict[str, Bits | None] = {}

    @property
    def library(self) -> LibraryDecoder:
        """Return the lookup decoder."""
        return self._library

    def vectors(self, smiles: Sequence[str]) -> np.ndarray:
        """Return a ``(n, d)`` matrix of vectors for ``smiles``."""
        raise NotImplementedError

    def payload_for(self, vector: np.ndarray) -> Bits | None:
        """Turn one vector into a payload."""
        raise NotImplementedError

    def encode_many(self, smiles: Sequence[str]) -> list[Bits | None]:
        """Encode a batch, the way the benchmark drives these codecs."""
        pending = [item for item in smiles if item not in self._cache]
        if pending:
            matrix = self.vectors(pending)
            for item, vector in zip(pending, matrix, strict=True):
                self._cache[item] = self.payload_for(vector)
        return [self._cache.get(item) for item in smiles]

    def encode(self, smiles: str) -> Bits | None:
        """Encode a single molecule."""
        return self.encode_many([smiles])[0]

    def decode(self, payload: Bits | None) -> str | None:
        """Decode by library lookup."""
        return self._library.decode(payload)

    def populate_library(self, smiles: Sequence[str]) -> None:
        """Add molecules and their payloads to the lookup decoder."""
        for item, payload in zip(smiles, self.encode_many(smiles), strict=True):
            self._library.add(payload, item)


class FingerprintCodec(_VectorCodec):
    """A structural fingerprint, decoded by library lookup.

    :param kind_name: ``"morgan"``, ``"maccs"``, ``"rdkit"`` or ``"atompair"``.
    :param n_bits: Fingerprint length. Lengths above :data:`BIT_BUDGET` are
        kept on purpose, so the report shows which common choices do not fit
        the budget at all.
    :param radius: Morgan radius; ECFP4 is radius 2.
    :param counted: Use counts rather than a bit vector, which keeps strictly
        more information.
    :param chiral: Fold stereochemistry into the hash. Off is the RDKit
        default, and it is why plain ECFP cannot reach a 99.99% ceiling: with
        chirality excluded a molecule and its epimer hash identically, and no
        decoder can separate them. Turning it on raises the ceiling at no cost
        in bits, so the usual ECFP ceiling is an artefact of the default
        settings rather than a property of hashing.
    :param world: Library world, see :class:`LibraryDecoder`.
    """

    family = "fingerprint"
    lossless_by_design = False

    def __init__(
        self,
        kind_name: str = "morgan",
        n_bits: int = 1024,
        radius: int = 2,
        *,
        counted: bool = False,
        chiral: bool = False,
        world: str = "closed",
    ) -> None:
        """Create the fingerprint codec."""
        super().__init__(world)
        self.kind_name = kind_name
        self.n_bits = n_bits
        self.radius = radius
        self.counted = counted
        self.chiral = chiral
        suffix = ("-count" if counted else "") + ("-chiral" if chiral else "")
        if kind_name == "morgan":
            self.name = f"ecfp{2 * radius}-{n_bits}{suffix}"
        elif kind_name == "maccs":
            self.name = "maccs-167"
        else:
            self.name = f"{kind_name}-{n_bits}{suffix}"
        self.notes = (
            "hashed structural keys, stereochemistry included"
            if chiral
            else "hashed structural keys; stereo-blind, so epimers collide"
        )
        self._generator = self._build_generator()

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state, dropping the RDKit generator.

        ``FingerprintGenerator64`` is a Boost.Python object with pickling
        disabled upstream, so a codec holding one cannot be saved as it stands.
        The generator is configuration only, a deterministic function of the
        four fields beside it, so dropping it and rebuilding on load costs
        nothing and keeps these codecs usable from a saved codebook.
        """
        state = {slot: getattr(self, slot) for slot in self.__dict__ if slot != "_generator"}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and rebuild the generator."""
        self.__dict__.update(state)
        self._generator = self._build_generator()

    def _build_generator(self):  # noqa: ANN202 - RDKit generator type is opaque
        """Return the RDKit fingerprint generator for this configuration."""
        if self.kind_name == "morgan":
            return rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius,
                fpSize=self.n_bits,
                includeChirality=self.chiral,
            )
        if self.kind_name == "rdkit":
            return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=self.n_bits)
        if self.kind_name == "atompair":
            return rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=self.n_bits,
                includeChirality=self.chiral,
            )
        if self.kind_name == "maccs":
            return None
        msg = f"unknown fingerprint kind {self.kind_name!r}"
        raise ValueError(msg)

    @property
    def payload_size(self) -> int:
        """Return the payload size in bits.

        Count fingerprints charge four bits per position, so counts up to
        fifteen, which covers any drug-like molecule.
        """
        if self.kind_name == "maccs":
            return 167
        return self.n_bits * (4 if self.counted else 1)

    def vectors(self, smiles: Sequence[str]) -> np.ndarray:
        """Compute fingerprints as a ``(n, length)`` integer matrix."""
        width = 167 if self.kind_name == "maccs" else self.n_bits
        out = np.zeros((len(smiles), width), dtype=np.int16)
        for row, item in enumerate(smiles):
            mol = Chem.MolFromSmiles(item)
            if mol is None:
                out[row, :] = -1
                continue
            if self.kind_name == "maccs":
                fingerprint = MACCSkeys.GenMACCSKeys(mol)
                out[row, :] = np.frombuffer(
                    bytes(fingerprint.ToBitString()[:167], "ascii"),
                    dtype=np.uint8,
                ) - ord("0")
            elif self.counted:
                array = self._generator.GetCountFingerprintAsNumPy(mol)
                out[row, :] = np.clip(array, 0, 15).astype(np.int16)
            else:
                out[row, :] = self._generator.GetFingerprintAsNumPy(mol).astype(np.int16)
        return out

    def payload_for(self, vector: np.ndarray) -> Bits | None:
        """Pack a fingerprint into a payload."""
        if vector.size and vector[0] == -1:
            return None
        width = 4 if self.counted else 1
        value = 0
        for entry in vector:
            value = (value << width) | int(entry)
        return Bits(value, width * vector.size)

    def payload_bits(self, payload: Bits | None) -> int:
        """Return the payload size in bits."""
        if payload is None:
            return self.payload_size
        return payload.length


class LatentCodec(_VectorCodec):
    """A learned continuous embedding, quantised to a bit budget.

    :param encoder: Callable mapping a list of SMILES to a ``(n, d)`` float
        matrix.
    :param encoder_name: Name used in the report.
    :param bits_per_dim: Bits spent per retained dimension.
    :param dims: Dimensions retained, highest-variance first; ``None`` keeps
        all.
    :param native_dtype_bits: Bits per coordinate at native precision, for
        reporting what the unquantised latent would cost.
    :param world: Library world, see :class:`LibraryDecoder`.
    """

    family = "latent"
    lossless_by_design = False

    def __init__(
        self,
        encoder: Callable[[Sequence[str]], np.ndarray],
        encoder_name: str,
        bits_per_dim: int = 1,
        dims: int | None = None,
        native_dtype_bits: int = 32,
        *,
        world: str = "closed",
    ) -> None:
        """Create the latent codec."""
        super().__init__(world)
        self._encoder = encoder
        self.encoder_name = encoder_name
        self.native_dtype_bits = native_dtype_bits
        self.quantizer = Quantizer(bits_per_dim=bits_per_dim, dims=dims)
        label = "all" if dims is None else str(dims)
        self.name = f"{encoder_name}-q{bits_per_dim}x{label}"
        self.notes = f"{encoder_name} embedding quantised to {bits_per_dim} bit(s) per retained dimension"
        self._native_dims: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state, dropping the encoder closure.

        The encoder is rebuilt from :attr:`encoder_name` on load (see
        :data:`_ENCODER_FACTORIES`), so the quantiser ranges and the lookup
        library survive a save without the unpicklable part having to.
        """
        return {slot: getattr(self, slot) for slot in self.__dict__ if slot != "_encoder"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and rebuild the encoder from its name."""
        self.__dict__.update(state)
        self._encoder = build_encoder(self.encoder_name)

    @property
    def native_bits(self) -> int | None:
        """Return the cost of the unquantised latent, in bits."""
        if self._native_dims is None:
            return None
        return self._native_dims * self.native_dtype_bits

    def vectors(self, smiles: Sequence[str]) -> np.ndarray:
        """Return the encoder's embeddings."""
        matrix = np.asarray(self._encoder(list(smiles)), dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        self._native_dims = int(matrix.shape[1])
        return matrix

    def fit(self, smiles: Sequence[str]) -> None:
        """Fit the quantiser's dimensions and ranges on the fitting split."""
        matrix = self.vectors(list(smiles))
        self.quantizer.fit(matrix)

    def payload_for(self, vector: np.ndarray) -> Bits | None:
        """Quantise an embedding into a payload."""
        if not np.all(np.isfinite(vector)):
            return None
        return self.quantizer.encode(vector)

    def payload_bits(self, payload: Bits | None) -> int:
        """Return the payload size in bits."""
        if payload is None:
            return self.quantizer.total_bits
        return payload.length


def fingerprint_codecs(world: str = "closed") -> list[FingerprintCodec]:
    """Return the fingerprint ladder, from inside the budget to far outside it.

    The sizes bracket :data:`BIT_BUDGET`, with 512 bits inside it and 1024 and
    2048 outside. ECFP4 at 2048 bits is the field's usual choice, so where it
    lands against a 1000-bit ceiling is part of the point.
    """
    return [
        FingerprintCodec("morgan", 512, radius=2, world=world),
        FingerprintCodec("morgan", 1024, radius=2, world=world),
        FingerprintCodec("morgan", 2048, radius=2, world=world),
        FingerprintCodec("morgan", 1024, radius=3, world=world),
        FingerprintCodec("morgan", 512, radius=2, counted=True, world=world),
        # The stereo-aware variants cost the same number of bits, so any gain
        # in their ceiling comes from the flag alone.
        FingerprintCodec("morgan", 512, radius=2, chiral=True, world=world),
        FingerprintCodec("morgan", 2048, radius=2, chiral=True, world=world),
        FingerprintCodec("maccs", 167, world=world),
        FingerprintCodec("rdkit", 512, world=world),
        FingerprintCodec("atompair", 512, world=world),
        FingerprintCodec("atompair", 512, chiral=True, world=world),
    ]
