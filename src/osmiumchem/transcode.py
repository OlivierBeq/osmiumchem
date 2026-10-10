# -*- coding: utf-8 -*-

"""Encode and decode one molecule at a time.

Where the benchmark compares techniques over 200000 molecules, this module
runs a single molecule through one of them.

:func:`encode` takes a molecule and returns a payload with its exact bit cost.
:func:`decode` takes a payload and returns a molecule in the requested
representation. :func:`round_trip` does both and adds the verdict, since "it
decoded" and "it decoded to the same molecule" are different claims. The
verdict comes from :func:`~osmiumchem.identity.same_molecule`, the test the
benchmark scores with, so the two agree.

:func:`encode_many`, :func:`decode_many` and :func:`round_trip_many` take
lists and return one result per entry, in order. They route through the
codec's own batch path where there is one, which is what makes a latent codec
usable at all: a neural encoder handles five hundred molecules in about the
time it takes to handle one.

:mod:`osmiumchem.api` wraps all six together with a fitted codec and is the
easier way in; these are the steps underneath.

A successful round trip usually does not return the string that went in.
``CN1C=NC2=C1C(=O)N(C)C(=O)N2C`` comes back as ``Cn1c(=O)c2c(ncn2C)n(C)c1=O``;
both are caffeine. Identity is molecular rather than textual, and codecs
encode the canonical form.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .codecs.vectors import BIT_BUDGET
from .formats import (
    format_payload,
    parse_molecule,
    parse_payload,
    payload_form,
    render_molecule,
)
from .identity import canonical_smiles, same_molecule

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .codecs.base import BaseCodec

__all__ = [
    "Decoded",
    "Encoded",
    "RoundTripped",
    "decode",
    "decode_many",
    "encode",
    "encode_many",
    "round_trip",
    "round_trip_many",
]


@dataclass(slots=True)
class Encoded:
    """The result of encoding one molecule.

    :param ok: Whether a payload was produced.
    :param input: The molecule as the user wrote it.
    :param input_format: The representation it was read as.
    :param canonical: Canonical SMILES of the parsed molecule.
    :param payload: The payload object, ready for ``codec.decode``.
    :param text: The payload written as text, reversibly.
    :param form: Which textual form ``text`` uses.
    :param bits: Exact payload size in bits.
    :param budget: Budget the size was compared against.
    :param reason: Why encoding failed, when it did.
    """

    ok: bool
    input: str
    input_format: str = "unrecognised"
    canonical: str | None = None
    payload: Any = None
    text: str | None = None
    form: str = "bits"
    bits: int | None = None
    budget: int = BIT_BUDGET
    reason: str | None = None

    @property
    def within_budget(self) -> bool | None:
        """Whether the payload fits the budget, or ``None`` if there is none."""
        return None if self.bits is None else self.bits <= self.budget


@dataclass(slots=True)
class Decoded:
    """The result of decoding one payload.

    :param ok: Whether a molecule came back.
    :param payload_text: The payload text that was read.
    :param bits: Exact payload size in bits.
    :param smiles: Canonical SMILES recovered, if any.
    :param rendered: ``smiles`` written in the requested output format.
    :param output_format: The representation ``rendered`` is in.
    :param reason: Why decoding failed, when it did.
    """

    ok: bool
    payload_text: str
    bits: int | None = None
    smiles: str | None = None
    rendered: str | None = None
    output_format: str = "smiles"
    reason: str | None = None


@dataclass(slots=True)
class RoundTripped:
    """An encode followed by a decode, with the identity verdict.

    Molecular identity and size stay separate, as they do in the benchmark:
    ``ok`` is about the molecule, ``within_budget`` about the size. An exact
    round trip can still be too large, which is what ``--strict-budget``
    distinguishes. :attr:`verdict` collapses the two for display.

    :param ok: Whether the molecule came back *as the same molecule*.
    :param encoded: The encode step.
    :param decoded: The decode step.
    :param identical: Whether the recovered molecule equals the input one.
    :param reason: The first thing that went wrong, when something did.
    """

    ok: bool
    encoded: Encoded
    decoded: Decoded | None = None
    identical: bool = False
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def bits(self) -> int | None:
        """Return the payload size in bits."""
        return self.encoded.bits

    @property
    def within_budget(self) -> bool | None:
        """Whether the payload fits the budget."""
        return self.encoded.within_budget

    @property
    def verdict(self) -> str:
        """Return ``"ok"``, ``"over-budget"`` or ``"failed"``.

        ``"over-budget"`` means the molecule survived but the payload is larger
        than the budget allows, which is not the same as losing it.
        """
        if not self.identical:
            return "failed"
        return "over-budget" if self.within_budget is False else "ok"


def encode(
    codec: BaseCodec,
    molecule: str,
    *,
    input_format: str = "auto",
    payload_format: str = "auto",
    budget: int = BIT_BUDGET,
) -> Encoded:
    """Encode one molecule with a fitted codec.

    :param codec: A *fitted* codec, usually from
        :func:`~osmiumchem.api.load_codec`.
    :param molecule: The molecule, in any format :data:`INPUT_FORMATS` covers.
    :param input_format: How to read ``molecule``; ``auto`` detects it.
    :param payload_format: Textual form for the payload; ``auto`` picks one to
        suit the codec.
    :param budget: Bit budget to compare the payload against.
    :returns: An :class:`Encoded`. Failures are reported rather than raised,
        apart from misuse such as an unknown format.
    """
    parsed, used = parse_molecule(molecule, input_format)
    if parsed is None:
        return _unreadable(molecule, used, budget)

    try:
        payload = codec.encode(parsed)
    except Exception as exc:  # noqa: BLE001 - a codec that cannot encode is reported
        return Encoded(
            ok=False,
            input=molecule,
            input_format=used,
            canonical=parsed,
            reason=f"{type(exc).__name__}: {exc}",
            budget=budget,
        )
    return _encoded(codec, molecule, used, parsed, payload, None, payload_format, budget)


def decode(
    codec: BaseCodec,
    payload: str | Any,
    *,
    output_format: str = "smiles",
) -> Decoded:
    """Decode one payload with a fitted codec.

    :param codec: The *same* fitted codec that produced the payload. Another
        fitting split gives another code, under which the payload decodes to a
        different molecule or to nothing; see :mod:`osmiumchem.codebook`.
    :param payload: Payload text as written by :func:`encode`, or a payload
        object.
    :param output_format: Representation to return the molecule in.
    :returns: A :class:`Decoded`.
    """
    text, obj, unreadable = _read_payload(codec, payload, output_format)
    if unreadable is not None:
        return unreadable

    try:
        recovered = codec.decode(obj)
    except Exception as exc:  # noqa: BLE001 - a codec that cannot decode is reported
        return Decoded(
            ok=False,
            payload_text=text,
            bits=_payload_bits(codec, obj),
            output_format=output_format,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return _decoded(codec, text, obj, recovered, output_format)


def round_trip(
    codec: BaseCodec,
    molecule: str,
    *,
    input_format: str = "auto",
    output_format: str = "smiles",
    payload_format: str = "auto",
    budget: int = BIT_BUDGET,
) -> RoundTripped:
    """Encode then decode one molecule, and check it is the same molecule.

    :param codec: A fitted codec.
    :param molecule: The molecule, in any supported representation.
    :param input_format: How to read ``molecule``; ``auto`` detects it.
    :param output_format: Representation for the recovered molecule.
    :param payload_format: Textual form for the payload.
    :param budget: Bit budget to compare the payload against.
    :returns: A :class:`RoundTripped` whose ``ok`` is true only when the
        molecule came back identical, judged as the benchmark judges it.
    """
    encoded = encode(
        codec,
        molecule,
        input_format=input_format,
        payload_format=payload_format,
        budget=budget,
    )
    if not encoded.ok:
        return RoundTripped(ok=False, encoded=encoded, reason=encoded.reason)

    decoded = decode(codec, encoded.payload, output_format=output_format)
    if decoded.smiles is None:
        return RoundTripped(ok=False, encoded=encoded, decoded=decoded, reason=decoded.reason)
    return _judge(encoded, decoded)


def _payload_bits(codec: BaseCodec, payload: Any) -> int | None:  # noqa: ANN401 - codec-specific
    """Return the payload's bit cost, or ``None`` if the codec cannot size it."""
    try:
        return codec.payload_bits(payload)
    except (TypeError, ValueError):  # pragma: no cover - exotic payload types
        return None


def _read_payload(
    codec: BaseCodec,
    payload: str | Any,
    output_format: str,
) -> tuple[str, Any, Decoded | None]:
    """Turn payload text into a payload object, or say why it cannot be read.

    :returns: ``(payload as text, payload object, failure)``, where
        ``failure`` is a :class:`Decoded` if the text is not a payload of this
        codec's kind and ``None`` otherwise.
    """
    if isinstance(payload, str):
        try:
            return payload, parse_payload(payload, getattr(codec, "kind", "bitstring")), None
        except ValueError as exc:
            failure = Decoded(
                ok=False,
                payload_text=payload,
                output_format=output_format,
                reason=str(exc),
            )
            return payload, None, failure
    try:
        text = format_payload(payload, payload_form(codec))
    except (TypeError, ValueError):  # pragma: no cover - exotic payload types
        text = repr(payload)
    return text, payload, None


def _decoded(
    codec: BaseCodec,
    text: str,
    payload: Any,  # noqa: ANN401 - payload types are codec-specific
    recovered: str | None,
    output_format: str,
) -> Decoded:
    """Assemble the :class:`Decoded` for a payload the codec has resolved."""
    bits = _payload_bits(codec, payload)
    if recovered is None:
        return Decoded(
            ok=False,
            payload_text=text,
            bits=bits,
            output_format=output_format,
            reason=f"{codec.name} recovered no molecule from this payload",
        )
    canonical = canonical_smiles(recovered)
    if canonical is None:
        return Decoded(
            ok=False,
            payload_text=text,
            bits=bits,
            output_format=output_format,
            reason=f"decoded to {recovered!r}, which is not a parseable molecule",
        )
    rendered = render_molecule(canonical, output_format)
    return Decoded(
        ok=rendered is not None,
        payload_text=text,
        bits=bits,
        smiles=canonical,
        rendered=rendered,
        output_format=output_format,
        reason=None if rendered is not None else f"cannot write this molecule as {output_format}",
    )


def _unreadable(molecule: str, used: str, budget: int) -> Encoded:
    """Return the result for a string no supported representation could read."""
    hint = (
        "could not read it as a molecule in any supported representation"
        if used == "unrecognised"
        else f"read as {used}, but it did not parse into a molecule"
    )
    return Encoded(ok=False, input=molecule, input_format=used, reason=hint, budget=budget)


def _encoded(
    codec: BaseCodec,
    molecule: str,
    used: str,
    parsed: str,
    payload: Any,  # noqa: ANN401 - payload types are codec-specific
    reason: str | None,
    payload_format: str,
    budget: int,
) -> Encoded:
    """Assemble the :class:`Encoded` for a molecule that was parsed.

    Shared by the single and batch paths, so the two cannot disagree on what a
    payload costs or how it is written.
    """
    if payload is None:
        return Encoded(
            ok=False,
            input=molecule,
            input_format=used,
            canonical=parsed,
            reason=reason or f"{codec.name} declined to encode this molecule",
            budget=budget,
        )
    form = payload_form(codec, payload_format)
    return Encoded(
        ok=True,
        input=molecule,
        input_format=used,
        canonical=parsed,
        payload=payload,
        text=format_payload(payload, form),
        form=form,
        bits=codec.payload_bits(payload),
        budget=budget,
    )


def _judge(encoded: Encoded, decoded: Decoded) -> RoundTripped:
    """Compare a decoded molecule against the one that went in."""
    # ``encoded.canonical`` came out of ``canonical_smiles``, which is the
    # precondition for ``same_molecule`` skipping the re-canonicalisation.
    identical = same_molecule(encoded.canonical, decoded.smiles, reference_is_canonical=True)
    reason = None
    if not identical:
        reason = "decoded to a different molecule"
    elif not encoded.within_budget:
        reason = f"round trip exact, but {encoded.bits} bits exceeds the {encoded.budget}-bit budget"
    return RoundTripped(
        ok=identical,
        encoded=encoded,
        decoded=decoded,
        identical=identical,
        reason=reason,
    )


def _encode_batch(codec: BaseCodec, molecules: Sequence[str]) -> list[tuple[Any, str | None]]:
    """Encode canonical SMILES in bulk, through the codec's batch path if it has one.

    A neural encoder evaluates 500 molecules in roughly the time it takes to
    evaluate one, hence the batch path. When a batch raises, the molecules are
    retried one at a time, so one molecule the encoder cannot handle costs its
    own result rather than the whole list.
    """
    if not molecules:
        return []
    batch = getattr(codec, "encode_many", None)
    if callable(batch):
        try:
            produced = list(batch(molecules))
        except Exception:  # noqa: BLE001 - retried one at a time below
            produced = None
        else:
            # A batch result of the wrong length cannot be matched back to its
            # inputs, and guessing the alignment would mislabel molecules.
            if len(produced) == len(molecules):
                return [(payload, None) for payload in produced]
            produced = None
    results: list[tuple[Any, str | None]] = []
    for molecule in molecules:
        try:
            results.append((codec.encode(molecule), None))
        except Exception as exc:  # noqa: BLE001 - a codec that cannot encode is reported
            results.append((None, f"{type(exc).__name__}: {exc}"))
    return results


def encode_many(
    codec: BaseCodec,
    molecules: Iterable[str],
    *,
    input_format: str = "auto",
    payload_format: str = "auto",
    budget: int = BIT_BUDGET,
) -> list[Encoded]:
    """Encode several molecules, using the codec's batch path where there is one.

    :param codec: A fitted codec.
    :param molecules: Molecules, in any supported representation.
    :param input_format: How to read them; ``auto`` detects each one.
    :param payload_format: Textual form for the payloads.
    :param budget: Bit budget to compare each payload against.
    :returns: One :class:`Encoded` per input, in order. Molecules that could
        not be read or encoded come back as failed results, so one bad entry
        does not cost the batch.
    """
    items = list(molecules)
    results: list[Encoded | None] = [None] * len(items)
    parsed: list[tuple[int, str, str]] = []
    for index, molecule in enumerate(items):
        canonical, used = parse_molecule(molecule, input_format)
        if canonical is None:
            results[index] = _unreadable(molecule, used, budget)
        else:
            parsed.append((index, used, canonical))

    produced = _encode_batch(codec, [canonical for _, _, canonical in parsed])
    for (index, used, canonical), (payload, reason) in zip(parsed, produced, strict=True):
        results[index] = _encoded(
            codec,
            items[index],
            used,
            canonical,
            payload,
            reason,
            payload_format,
            budget,
        )
    return [result for result in results if result is not None]


def decode_many(
    codec: BaseCodec,
    payloads: Iterable[str | Any],
    *,
    output_format: str = "smiles",
) -> list[Decoded]:
    """Decode several payloads, using the codec's batch path where there is one.

    :param codec: The same fitted codec that produced the payloads.
    :param payloads: Payload texts or payload objects.
    :param output_format: Representation to return the molecules in.
    :returns: One :class:`Decoded` per payload, in order.
    """
    items = list(payloads)
    batch = getattr(codec, "decode_many", None)
    if not callable(batch):
        return [decode(codec, payload, output_format=output_format) for payload in items]

    results: list[Decoded | None] = [None] * len(items)
    readable: list[tuple[int, str, Any]] = []
    for index, payload in enumerate(items):
        text, obj, failure = _read_payload(codec, payload, output_format)
        if failure is not None:
            results[index] = failure
        else:
            readable.append((index, text, obj))
    try:
        recovered = list(batch([obj for _, _, obj in readable]))
    except Exception:  # noqa: BLE001 - retried one at a time below
        recovered = None
    if recovered is None or len(recovered) != len(readable):
        for index, _, obj in readable:
            results[index] = decode(codec, obj, output_format=output_format)
    else:
        for (index, text, obj), smiles in zip(readable, recovered, strict=True):
            results[index] = _decoded(codec, text, obj, smiles, output_format)
    return [result for result in results if result is not None]


def round_trip_many(
    codec: BaseCodec,
    molecules: Iterable[str],
    *,
    input_format: str = "auto",
    output_format: str = "smiles",
    payload_format: str = "auto",
    budget: int = BIT_BUDGET,
) -> list[RoundTripped]:
    """Encode then decode several molecules and judge each one.

    Both halves use their batch paths, which is what makes checking a few
    thousand molecules under a latent codec practical. The verdicts match
    :func:`round_trip`.

    :returns: One :class:`RoundTripped` per input, in order.
    """
    encoded = encode_many(
        codec,
        molecules,
        input_format=input_format,
        payload_format=payload_format,
        budget=budget,
    )
    pending = [index for index, item in enumerate(encoded) if item.ok]
    decoded = decode_many(
        codec,
        [encoded[index].payload for index in pending],
        output_format=output_format,
    )
    results: list[RoundTripped] = [RoundTripped(ok=False, encoded=item, reason=item.reason) for item in encoded]
    for index, outcome in zip(pending, decoded, strict=True):
        if outcome.smiles is None:
            results[index] = RoundTripped(
                ok=False,
                encoded=encoded[index],
                decoded=outcome,
                reason=outcome.reason,
            )
        else:
            results[index] = _judge(encoded[index], outcome)
    return results
