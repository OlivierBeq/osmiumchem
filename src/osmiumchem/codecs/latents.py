# -*- coding: utf-8 -*-

"""Learned latent spaces: CDDD, GROVER, and a descriptor baseline.

The codecs the bit budget was written for. A neural encoder returns floats, and
turning those into a payload raises two questions.

How many bits is a float vector? CDDD emits 512 float32 coordinates, 16384
bits, sixteen times the 1000-bit ceiling, and GROVER's base model emits more.
Neither fits at native precision, so quantisation is the entry fee rather than
an optimisation. :class:`CdddNativeCodec` sweeps bits per dimension to show
what the budget costs in fidelity.

Can a fitted decoder make up the difference? Not past the injectivity ceiling.
Once two molecules share a payload the distinction is gone, so
:func:`~osmiumchem.codecs.vectors.injectivity` bounds every decoder including an
ideal one. Where a decoder exists, it gets measured: CDDD ships a trained
beam-search decoder, so :class:`CdddNativeCodec` quantises the latent,
de-quantises it back to bin centres and asks that decoder what molecule it is.

GROVER has no decoder, being a representation learner for property prediction,
so it is reported through the injectivity bound and through library lookup,
with the bound standing for the best any future decoder could manage.

Model weights download on first use and are cached by the upstream packages.
Both encoders are imported lazily, so the rest of the benchmark runs without
JAX installed.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from ..types import Bits
from .base import BaseCodec
from .vectors import LatentCodec, Quantizer, build_encoder, register_encoder

__all__ = [
    "CdddNativeCodec",
    "LibraryIndexCodec",
    "cddd_encoder",
    "descriptor_encoder",
    "grover_encoder",
    "latent_codecs",
]

_CDDD_MODEL = None
_GROVER_MODEL = None


class _EmbeddingCache:
    """Remembers recent embeddings so sibling codecs do not recompute them.

    One encoder serves several codecs: CDDD alone is read by three quantisation
    settings and four native-decoder settings, each of which would otherwise
    re-embed the whole fitting corpus. Neural inference dominates their runtime,
    so this cache is the difference between a run of minutes and one of an hour.

    The bound is on stored floats rather than molecules, since the encoders
    differ more than sixfold in width (CDDD 512, GROVER-base 3200) and a
    molecule count would mean very different memory for each. Eviction is
    first-in, first-out, which suits the access pattern: the codecs sweep the
    same molecule list in the same order, so one full pass is the working set.

    :param max_floats: Approximate ceiling on retained floats, about 400 MB of
        float32 by default.
    """

    __slots__ = ("_order", "_store", "max_floats", "stored_floats")

    def __init__(self, max_floats: int = 100_000_000) -> None:
        """Create an empty cache."""
        self.max_floats = max_floats
        self._store: dict[str, np.ndarray] = {}
        self._order: deque[str] = deque()
        self.stored_floats = 0

    def get_missing(self, smiles: Sequence[str]) -> list[str]:
        """Return the molecules not already cached, without duplicates."""
        seen: set[str] = set()
        missing: list[str] = []
        for item in smiles:
            if item not in self._store and item not in seen:
                seen.add(item)
                missing.append(item)
        return missing

    def put(self, smiles: Sequence[str], matrix: np.ndarray) -> None:
        """Store embeddings, evicting oldest entries past the bound."""
        for item, row in zip(smiles, matrix, strict=True):
            if item in self._store:
                continue
            vector = np.ascontiguousarray(row, dtype=np.float32)
            self._store[item] = vector
            self._order.append(item)
            self.stored_floats += vector.size
        while self.stored_floats > self.max_floats and self._order:
            oldest = self._order.popleft()
            evicted = self._store.pop(oldest, None)
            if evicted is not None:
                self.stored_floats -= evicted.size

    def stack(self, smiles: Sequence[str]) -> np.ndarray:
        """Return the embeddings for ``smiles``, which must all be cached."""
        return np.stack([self._store[item] for item in smiles])


_CDDD_CACHE = _EmbeddingCache()
_GROVER_CACHE = _EmbeddingCache()


def _cached(cache: _EmbeddingCache, compute, smiles: Sequence[str]) -> np.ndarray:  # noqa: ANN001
    """Return embeddings for ``smiles``, computing only what is missing."""
    missing = cache.get_missing(smiles)
    if missing:
        cache.put(missing, np.asarray(compute(missing)))
    return cache.stack(list(smiles))


def cddd_encoder(batch_size: int = 512):  # noqa: ANN201 - returns a closure
    """Return a callable embedding SMILES with pretrained CDDD.

    The model loads once and is reused. The first call downloads about 200 MB
    of weights. Results are cached across codecs.
    """

    def compute(smiles: Sequence[str]) -> np.ndarray:
        global _CDDD_MODEL  # noqa: PLW0603 - a single shared model is intended
        if _CDDD_MODEL is None:
            from jax_cddd.inference import CDDDModel

            _CDDD_MODEL = CDDDModel()
        return np.asarray(_CDDD_MODEL.seq_to_emb(list(smiles), chunk_size=batch_size))

    def encode(smiles: Sequence[str]) -> np.ndarray:
        return _cached(_CDDD_CACHE, compute, smiles)

    return encode


def grover_encoder(model_type: str = "base", batch_size: int = 256):  # noqa: ANN201
    """Return a callable embedding SMILES with pretrained GROVER.

    Results are cached across codecs, as for CDDD.
    """

    def compute(smiles: Sequence[str]) -> np.ndarray:
        global _GROVER_MODEL  # noqa: PLW0603 - a single shared model is intended
        if _GROVER_MODEL is None:
            from jax_grover.inference import GroverModel

            _GROVER_MODEL = GroverModel(model_type=model_type)
        return np.asarray(_GROVER_MODEL.embed_smiles(list(smiles), chunk_size=batch_size))

    def encode(smiles: Sequence[str]) -> np.ndarray:
        return _cached(_GROVER_CACHE, compute, smiles)

    return encode


#: Descriptors used by the dependency-free latent baseline.
_DESCRIPTORS = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumAromaticRings",
    "BalabanJ",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "HallKierAlpha",
    "MaxPartialCharge",
)


def descriptor_encoder():  # noqa: ANN201 - returns a closure
    """Return a callable embedding SMILES as physicochemical descriptors.

    A weak latent space on purpose. It needs no download, and it shows that a
    handful of real-valued properties cannot identify a molecule however finely
    they are quantised.
    """
    functions = [(name, getattr(Descriptors, name)) for name in _DESCRIPTORS]

    def encode(smiles: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(smiles), len(functions)), dtype=np.float32)
        for row, item in enumerate(smiles):
            mol = Chem.MolFromSmiles(item)
            if mol is None:
                out[row, :] = np.nan
                continue
            for column, (_, function) in enumerate(functions):
                try:
                    value = function(mol)
                except Exception:  # noqa: BLE001 - some descriptors fail on odd valences
                    value = 0.0
                out[row, column] = 0.0 if value is None or not math.isfinite(value) else value
        return out

    return encode


# Registered so a saved codec can rebuild its encoder (see
# :data:`osmiumchem.codecs.vectors._ENCODER_FACTORIES`). The names match the
# ones :class:`~osmiumchem.codecs.vectors.LatentCodec` reports under.
register_encoder("descriptors", descriptor_encoder)
register_encoder("cddd", cddd_encoder)
register_encoder("grover", grover_encoder)


class CdddNativeCodec(BaseCodec):
    """CDDD's latent space decoded by CDDD's own trained decoder.

    The only codec here with a learned decoder, and so the direct test of
    whether a neural latent space can carry a molecule losslessly inside a bit
    budget. Encode to 512 floats, quantise to ``bits_per_dim`` per coordinate,
    de-quantise back to bin centres, hand that to the pretrained beam search.

    All 512 dimensions are kept whatever the budget, because the decoder's
    input layer expects them, so the budget goes entirely on precision per
    coordinate. One bit per dimension is 512 bits and fits; two is 1024 and
    does not, which brackets the ceiling.

    :param bits_per_dim: Bits per coordinate, or ``None`` for the full float32
        latent at ``32 * 512`` bits.
    :param beam_width: Decoder beam width. A wider search costs time and can
        only help accuracy.
    """

    family = "latent"
    kind = "bitstring"
    size_model = "exact"
    lossless_by_design = False

    def __init__(self, bits_per_dim: int | None = 1, beam_width: int = 10) -> None:
        """Create the codec."""
        self.bits_per_dim = bits_per_dim
        self.beam_width = beam_width
        label = "f32" if bits_per_dim is None else f"q{bits_per_dim}"
        self.name = f"cddd-native-{label}"
        self.notes = (
            "CDDD latent at full float32 precision, decoded by its own beam search"
            if bits_per_dim is None
            else f"CDDD latent at {bits_per_dim} bit(s)/dim, decoded by its own beam search"
        )
        self._encoder = cddd_encoder()
        self._quantizer = Quantizer(bits_per_dim=bits_per_dim or 1)
        self._dims: int | None = None

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state, dropping the encoder closure."""
        return {slot: getattr(self, slot) for slot in self.__dict__ if slot != "_encoder"}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and rebuild CDDD's encoder."""
        self.__dict__.update(state)
        self._encoder = build_encoder("cddd")

    @property
    def native_bits(self) -> int | None:
        """Return the cost of the unquantised float32 latent."""
        return None if self._dims is None else 32 * self._dims

    @property
    def payload_size(self) -> int:
        """Return the payload size in bits."""
        if self._dims is None:
            return 0
        if self.bits_per_dim is None:
            return 32 * self._dims
        return self.bits_per_dim * self._dims

    def fit(self, smiles: Sequence[str]) -> None:
        """Fit the quantiser's per-dimension ranges on the fitting split."""
        matrix = self._encoder(list(smiles))
        self._dims = int(matrix.shape[1])
        if self.bits_per_dim is not None:
            self._quantizer.fit(matrix)

    def _to_payload(self, vector: np.ndarray) -> Bits | None:
        """Turn one embedding into a payload."""
        if not np.all(np.isfinite(vector)):
            return None
        if self.bits_per_dim is None:
            # The exact float32 bit pattern, which is what storing the latent
            # actually costs.
            value = 0
            for word in vector.astype(np.float32).view(np.uint32):
                value = (value << 32) | int(word)
            return Bits(value, 32 * vector.size)
        return self._quantizer.encode(vector)

    def _from_payload(self, payload: Bits) -> np.ndarray | None:
        """Recover an embedding (or its bin centres) from a payload."""
        if self.bits_per_dim is None:
            if self._dims is None or payload.length != 32 * self._dims:
                return None
            words = np.empty(self._dims, dtype=np.uint32)
            value = payload.value
            for position in range(self._dims - 1, -1, -1):
                words[position] = value & 0xFFFFFFFF
                value >>= 32
            return words.view(np.float32)
        return self._quantizer.decode(payload)

    def encode_many(self, smiles: Sequence[str]) -> list[Bits | None]:
        """Encode a batch of molecules."""
        matrix = self._encoder(list(smiles))
        return [self._to_payload(row) for row in matrix]

    def encode(self, smiles: str) -> Bits | None:
        """Encode one molecule."""
        return self.encode_many([smiles])[0]

    def decode_many(self, payloads: Sequence[Bits | None]) -> list[str | None]:
        """Decode a batch through CDDD's beam-search decoder."""
        global _CDDD_MODEL  # noqa: PLW0603 - a single shared model is intended
        vectors: list[np.ndarray] = []
        slots: list[int] = []
        out: list[str | None] = [None] * len(payloads)
        for index, payload in enumerate(payloads):
            if payload is None:
                continue
            vector = self._from_payload(payload)
            if vector is None:
                continue
            vectors.append(vector)
            slots.append(index)
        if not vectors:
            return out
        if _CDDD_MODEL is None:
            self._encoder(["C"])
        matrix = np.asarray(vectors, dtype=np.float32)
        decoded = _CDDD_MODEL.emb_to_seq(matrix, beam_width=self.beam_width)
        for slot, item in zip(slots, decoded, strict=True):
            if isinstance(item, (list, tuple)):
                item = item[0] if item else None
            out[slot] = item if isinstance(item, str) else None
        return out

    def decode(self, payload: Bits | None) -> str | None:
        """Decode one payload."""
        return self.decode_many([payload])[0]

    def payload_bits(self, payload: Bits | None) -> int:
        """Return the payload size in bits."""
        return self.payload_size if payload is None else payload.length


class LibraryIndexCodec(BaseCodec):
    """Encodes a molecule as its index in a known library.

    A control, and an important one. In information terms, any codec decoded by
    library lookup only names an entry in that library, which costs
    ``ceil(log2(N))`` bits, about 18 for a quarter of a million molecules. When
    a 2048-bit fingerprint scores perfectly through lookup, this codec matches
    it with a hundredth of the bits, so the 2048 bits were not what identified
    the molecule.

    Its limitation is the point: it cannot represent a molecule outside the
    library, and it carries no chemical information. Read it as the lower bound
    on bits for closed-world identification, and as a reminder that a
    closed-world score is not reconstruction.
    """

    name = "library-index"
    family = "control"
    kind = "bitstring"
    size_model = "exact"
    lossless_by_design = False
    notes = "index into a fixed library; the true bit cost of closed-world lookup"

    def __init__(self) -> None:
        """Create an empty codec."""
        self._index: dict[str, int] = {}
        self._molecules: list[str] = []
        self._width = 1

    def prepare_library(self, smiles: Sequence[str]) -> None:
        """Register the library the index refers to."""
        self._molecules = list(dict.fromkeys(smiles))
        self._index = {item: i for i, item in enumerate(self._molecules)}
        self._width = max(1, max(1, len(self._molecules) - 1).bit_length())

    def encode(self, smiles: str) -> Bits | None:
        """Return the molecule's library index."""
        position = self._index.get(smiles)
        if position is None:
            return None
        return Bits(position, self._width)

    def decode(self, payload: Bits | None) -> str | None:
        """Return the molecule at the encoded index."""
        if payload is None or payload.value >= len(self._molecules):
            return None
        return self._molecules[payload.value]


def latent_codecs(
    *,
    include_cddd: bool = True,
    include_grover: bool = True,
    world: str = "closed",
) -> list[BaseCodec]:
    """Return the latent-space codecs that can be constructed here.

    The quantisation levels bracket the 1000-bit budget, so the report shows a
    configuration that fits and one that does not.

    :param include_cddd: Whether to include CDDD codecs.
    :param include_grover: Whether to include GROVER codecs.
    :param world: Library world for lookup-decoded codecs.
    """
    codecs: list[BaseCodec] = [
        LatentCodec(descriptor_encoder(), "descriptors", bits_per_dim=16, world=world),
        LatentCodec(descriptor_encoder(), "descriptors", bits_per_dim=32, world=world),
    ]
    if include_cddd:
        codecs.extend(
            [
                LatentCodec(cddd_encoder(), "cddd", bits_per_dim=1, world=world),
                LatentCodec(cddd_encoder(), "cddd", bits_per_dim=2, dims=500, world=world),
                LatentCodec(cddd_encoder(), "cddd", bits_per_dim=8, dims=125, world=world),
                CdddNativeCodec(bits_per_dim=1),
                CdddNativeCodec(bits_per_dim=2),
                CdddNativeCodec(bits_per_dim=4),
                CdddNativeCodec(bits_per_dim=None),
            ],
        )
    if include_grover:
        codecs.extend(
            [
                # GROVER-base emits 3200 dimensions: 102400 bits at float32,
                # still 3200 at one bit each. Fitting the budget means dropping
                # dimensions as well as precision, so only the highest-variance
                # 1000 (or 500) coordinates are kept.
                LatentCodec(grover_encoder(), "grover", bits_per_dim=1, dims=1000, world=world),
                LatentCodec(grover_encoder(), "grover", bits_per_dim=2, dims=500, world=world),
                LatentCodec(grover_encoder(), "grover", bits_per_dim=8, dims=125, world=world),
            ],
        )
    return codecs
