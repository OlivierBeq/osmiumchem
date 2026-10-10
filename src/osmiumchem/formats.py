# -*- coding: utf-8 -*-

"""Reading molecules in, writing molecules out, and writing payloads down.

Codecs take canonical SMILES and return a payload, which is the right internal
contract since molecular identity here *is* canonical SMILES. It is not what
someone at a shell prompt holding a SELFIES string or an InChI has. This module
is the adapter in both directions.

Format detection is ordered, and the order matters, because ``[C][C][O]`` is
both valid SELFIES and valid SMILES. :func:`detect_format` looks for tokens
that only SELFIES allows (``[Ring1]``, ``[Branch2]``, ``[=C]``) and otherwise
prefers SMILES; ``--in`` overrides it, and the format used is always reported
back so a wrong guess shows up.

A payload's bit length is part of the payload: leading zeros carry information
and the length is rarely a multiple of eight, so hex alone would lose data.
The ``hex`` form is ``<bits>:<hex digits>``, the ``bits`` form is the binary
string, and both read back exactly.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rdkit import Chem

from .identity import canonical_smiles
from .types import Bits

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .codecs.base import BaseCodec

__all__ = [
    "INPUT_FORMATS",
    "OUTPUT_FORMATS",
    "PAYLOAD_FORMATS",
    "detect_format",
    "format_payload",
    "parse_molecule",
    "parse_payload",
    "payload_form",
    "render_molecule",
]

#: Representations accepted as input. ``auto`` runs :func:`detect_format`.
INPUT_FORMATS = ("auto", "smiles", "selfies", "inchi", "deepsmiles")

#: Representations a decoded molecule can be written as.
#:
#: ``inchikey`` is a hash, so it cannot be read back and
#: :func:`parse_molecule` refuses it. It is here for pasting into a search box.
OUTPUT_FORMATS = ("smiles", "smiles-kekule", "selfies", "inchi", "inchikey", "deepsmiles")

#: Textual forms a payload can be written in. ``auto`` picks ``text`` for
#: string-payload codecs and ``bits`` for bit-sequence ones.
PAYLOAD_FORMATS = ("auto", "bits", "hex", "text")

#: Bracket tokens legal in SELFIES and illegal in SMILES: ring and branch
#: instructions, the padding tokens, and the bond-order prefixes that SELFIES
#: writes inside the bracket and SMILES outside it.
_SELFIES_ONLY = re.compile(r"\[(?:Ring\d|Branch\d|nop|epsilon)\]|\[[=#/\\]")

#: ``<bit length>:<hex digits>``, the self-describing hexadecimal payload form.
_HEX_PAYLOAD = re.compile(r"\A(\d+):([0-9a-fA-F]*)\Z")


def _selfies_codec(preset: str = "hypervalent"):  # noqa: ANN202 - avoids a circular import
    """Return a SELFIES string codec.

    ``hypervalent`` rather than ``default``, because the job here is to accept
    what the user has. The narrower preset rejects a small share of real
    molecules, which is a result worth measuring in the benchmark but only an
    obstacle at a prompt.
    """
    from .codecs.strings import SelfiesCodec

    return SelfiesCodec(preset)


def _deepsmiles_codec():  # noqa: ANN202 - avoids a circular import
    """Return a DeepSMILES codec, or ``None`` if the package is absent."""
    from .codecs.strings import DeepSmilesCodec

    try:
        return DeepSmilesCodec()
    except ImportError:  # pragma: no cover - optional dependency
        return None


def detect_format(text: str) -> str | None:
    """Guess which representation ``text`` is written in.

    :param text: The string to classify.
    :returns: One of :data:`INPUT_FORMATS` excluding ``auto``, or ``None`` if
        nothing parses it.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("InChI="):
        return "inchi"
    if _SELFIES_ONLY.search(stripped):
        return "selfies"
    if Chem.MolFromSmiles(stripped) is not None:
        return "smiles"
    # Not a molecule as written, so what is left are the representations that
    # need decoding first.
    if _selfies_codec().from_text(stripped):
        return "selfies"
    deep = _deepsmiles_codec()
    if deep is not None and canonical_smiles(deep.from_text(stripped)):
        return "deepsmiles"
    return None


def parse_molecule(text: str, fmt: str = "auto") -> tuple[str | None, str]:
    """Convert a molecule in any supported representation to canonical SMILES.

    :param text: The molecule as written by the user.
    :param fmt: One of :data:`INPUT_FORMATS`.
    :returns: ``(canonical SMILES or None, format used)``. The second element
        is what ``auto`` resolved to, so a bad guess is visible instead of
        hiding behind a failure.
    :raises ValueError: If ``fmt`` is not a supported input format.
    """
    if fmt not in INPUT_FORMATS:
        known = ", ".join(INPUT_FORMATS)
        msg = f"unsupported input format {fmt!r}; choose one of {known}"
        raise ValueError(msg)
    stripped = text.strip()
    used = detect_format(stripped) if fmt == "auto" else fmt
    if used is None:
        return None, "unrecognised"
    if used == "smiles":
        return canonical_smiles(stripped), used
    if used == "selfies":
        return canonical_smiles(_selfies_codec().from_text(stripped)), used
    if used == "inchi":
        from .codecs.strings import InchiCodec

        return canonical_smiles(InchiCodec().from_text(stripped)), used
    deep = _deepsmiles_codec()
    if deep is None:  # pragma: no cover - optional dependency
        return None, used
    return canonical_smiles(deep.from_text(stripped)), used


def render_molecule(smiles: str, fmt: str = "smiles") -> str | None:
    """Write a canonical SMILES in another representation.

    :param smiles: Canonical SMILES.
    :param fmt: One of :data:`OUTPUT_FORMATS`.
    :returns: The rendered string, or ``None`` if this molecule cannot be
        written that way (SELFIES refuses some species, InChI refuses others).
    :raises ValueError: If ``fmt`` is not a supported output format.
    """
    if fmt not in OUTPUT_FORMATS:
        known = ", ".join(OUTPUT_FORMATS)
        msg = f"unsupported output format {fmt!r}; choose one of {known}"
        raise ValueError(msg)
    if fmt == "smiles":
        return smiles
    if fmt == "inchikey":
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol) or None
        except Exception:  # noqa: BLE001 - the InChI library raises on odd valences
            return None
    if fmt == "selfies":
        return _selfies_codec().to_text(smiles)
    if fmt == "deepsmiles":
        deep = _deepsmiles_codec()
        return None if deep is None else deep.to_text(smiles)
    from .codecs.strings import InchiCodec, KekuleSmilesCodec

    if fmt == "inchi":
        return InchiCodec().to_text(smiles)
    return KekuleSmilesCodec().to_text(smiles)


def payload_form(codec: BaseCodec, fmt: str = "auto") -> str:
    """Resolve ``auto`` to a concrete payload form for ``codec``."""
    if fmt != "auto":
        return fmt
    return "text" if getattr(codec, "kind", "bitstring") == "string" else "bits"


def format_payload(payload: object, fmt: str = "bits") -> str:
    """Write a payload as text, exactly and reversibly.

    :param payload: A :class:`~osmiumchem.types.Bits` or a ``str``.
    :param fmt: ``"bits"``, ``"hex"`` or ``"text"``.
    :returns: The payload as text.
    :raises TypeError: If the payload is neither bits nor a string.
    :raises ValueError: If ``fmt`` is unknown, or asks for a form the payload
        cannot take.
    """
    if isinstance(payload, str):
        if fmt == "text":
            return payload
        # Refused rather than UTF-8 encoded: a string codec is charged under a
        # flat code over its fitted alphabet, so UTF-8 bits would misstate the
        # size and ``parse_payload`` could not read them back.
        msg = (
            f"the payload of a string codec has no {fmt!r} form; use 'text'. Its bit cost under "
            "the codec's alphabet code is reported separately."
        )
        raise ValueError(msg)
    if not isinstance(payload, Bits):
        msg = f"cannot write a payload of type {type(payload).__name__} as text"
        raise TypeError(msg)
    if fmt == "bits":
        return str(payload)
    if fmt == "hex":
        digits = (payload.length + 3) // 4
        return f"{payload.length}:{payload.value:0{digits}x}"
    if fmt == "text":
        msg = "a bit-sequence payload has no textual form; use 'bits' or 'hex'"
        raise ValueError(msg)
    msg = f"unsupported payload format {fmt!r}"
    raise ValueError(msg)


def parse_payload(text: str, kind: str = "bitstring") -> Bits | str:
    """Read a payload back from its textual form.

    The form comes from the text itself, not from a flag, so anything
    :func:`format_payload` printed can be pasted straight back in. ``kind``
    settles the one real ambiguity: for a string codec the text *is* the
    payload and must not be read as binary.

    :param text: Payload text, as written by :func:`format_payload`.
    :param kind: ``"string"`` or ``"bitstring"``, from ``codec.kind``.
    :returns: The payload, ready to hand to ``codec.decode``.
    :raises ValueError: If the text is not a payload of the expected kind.
    """
    stripped = text.strip()
    if kind == "string":
        if not stripped:
            msg = "empty payload"
            raise ValueError(msg)
        return stripped
    match = _HEX_PAYLOAD.match(stripped)
    if match is not None:
        length = int(match.group(1))
        digits = match.group(2)
        value = int(digits, 16) if digits else 0
        if value.bit_length() > length:
            msg = f"hex payload needs {value.bit_length()} bits but declares {length}"
            raise ValueError(msg)
        return Bits(value, length)
    if stripped and set(stripped) <= {"0", "1"}:
        return Bits.from_string(stripped)
    msg = (
        f"cannot read {stripped[:32]!r} as a bit-sequence payload; expected a string of "
        "0 and 1, or '<bit length>:<hex digits>'"
    )
    raise ValueError(msg)
