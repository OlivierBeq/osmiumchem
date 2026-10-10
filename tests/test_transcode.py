# -*- coding: utf-8 -*-

"""Tests for the by-name, one-molecule encode/decode interface."""

import lzma
import pickle

import pytest

from osmiumchem import codebook as cb
from osmiumchem.cli_transcode import run
from osmiumchem.codecs import catalogue, resolve_codec
from osmiumchem.codecs.strings import SmilesCodec
from osmiumchem.formats import (
    detect_format,
    format_payload,
    parse_molecule,
    parse_payload,
    payload_form,
    render_molecule,
)
from osmiumchem.identity import same_molecule, silence_rdkit
from osmiumchem.transcode import decode, encode, round_trip
from osmiumchem.types import Bits

silence_rdkit()

CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
CAFFEINE_CANONICAL = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
CAFFEINE_INCHI = "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"

FIT = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "C[C@H](N)C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Clc1ccccc1",
    "C/C=C/C",
    "[Na+].[Cl-]",
    "O=S(=O)(N)c1ccccc1",
]


def _fitted(name):
    """Return a codec fitted on the small local corpus."""
    codec = resolve_codec(name, world="open")
    codec.fit(FIT)
    return codec


# --------------------------------------------------------------------------
# resolving a codec from its name
# --------------------------------------------------------------------------


def test_every_catalogued_name_resolves():
    # The catalogue is what the CLI advertises, so every name in it has to come
    # back as a codec with that exact name.
    for name in catalogue():
        assert resolve_codec(name).name == name


def test_catalogue_covers_the_benchmarked_families():
    names = catalogue()
    for expected in ("smiles", "smiles+arith-o4", "selfies-hypervalent+huffman", "graph-bits"):
        assert expected in names


def test_unknown_codec_suggests_near_misses():
    with pytest.raises(KeyError, match="smiles\\+arith-o4"):
        resolve_codec("smiles+arith-o5")


def test_resolve_returns_a_fresh_instance_each_time():
    # A shared instance would leak one caller's fitted state into another's.
    first = resolve_codec("smiles+arith-o2")
    second = resolve_codec("smiles+arith-o2")
    assert first is not second


# --------------------------------------------------------------------------
# reading molecules in, writing them out
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (CAFFEINE, "smiles"),
        (CAFFEINE_INCHI, "inchi"),
        ("[C][N][C][=N][C][Ring1][Branch1]", "selfies"),
        ("[C][C][=Branch1][C][=O][O]", "selfies"),
    ],
)
def test_detect_format(text, expected):
    assert detect_format(text) == expected


def test_detect_format_rejects_nonsense():
    assert detect_format("not a molecule at all") is None
    assert detect_format("   ") is None


@pytest.mark.parametrize("fmt", ["auto", "smiles"])
def test_parse_molecule_canonicalises(fmt):
    parsed, used = parse_molecule(CAFFEINE, fmt)
    assert parsed == CAFFEINE_CANONICAL
    assert used == "smiles"


def test_parse_molecule_accepts_inchi_and_selfies():
    from_inchi, used_inchi = parse_molecule(CAFFEINE_INCHI)
    assert from_inchi == CAFFEINE_CANONICAL
    assert used_inchi == "inchi"

    selfies_text = render_molecule(CAFFEINE_CANONICAL, "selfies")
    from_selfies, used_selfies = parse_molecule(selfies_text, "selfies")
    assert from_selfies == CAFFEINE_CANONICAL
    assert used_selfies == "selfies"


def test_parse_molecule_reports_the_format_it_used_on_failure():
    parsed, used = parse_molecule("@@@@", "smiles")
    assert parsed is None
    assert used == "smiles"
    parsed, used = parse_molecule("@@@@")
    assert (parsed, used) == (None, "unrecognised")


def test_parse_molecule_rejects_an_unknown_format():
    with pytest.raises(ValueError, match="unsupported input format"):
        parse_molecule(CAFFEINE, "mol2")


@pytest.mark.parametrize("fmt", ["smiles", "smiles-kekule", "selfies", "inchi", "inchikey"])
def test_render_molecule_produces_something_for_each_format(fmt):
    assert render_molecule(CAFFEINE_CANONICAL, fmt)


def test_render_molecule_rejects_an_unknown_format():
    with pytest.raises(ValueError, match="unsupported output format"):
        render_molecule(CAFFEINE_CANONICAL, "mol2")


def test_rendered_representations_read_back_to_the_same_molecule():
    # Each output format is the same molecule in another notation. inchikey is
    # a hash, so it is excluded by construction.
    for fmt in ("smiles", "smiles-kekule", "selfies", "inchi"):
        text = render_molecule(CAFFEINE_CANONICAL, fmt)
        back, _ = parse_molecule(text, "auto")
        assert same_molecule(CAFFEINE_CANONICAL, back, reference_is_canonical=True), fmt


# --------------------------------------------------------------------------
# writing payloads down
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["bits", "hex"])
def test_payload_text_round_trips_exactly(fmt):
    # Leading zero bits are information, so a text form that lost them would
    # corrupt payloads. Every form has to be exactly reversible, including for
    # payloads that are not a whole number of bytes.
    for length in range(1, 40):
        for value in (0, 1, (1 << length) - 1, (1 << (length - 1))):
            payload = Bits(value, length)
            assert parse_payload(format_payload(payload, fmt), "bitstring") == payload


def test_hex_payload_keeps_its_length():
    payload = Bits(0b0001, 12)
    assert format_payload(payload, "hex") == "12:001"
    assert parse_payload("12:001", "bitstring") == payload


def test_bits_payload_form_is_the_binary_string():
    assert format_payload(Bits(0b1011, 6), "bits") == "001011"


def test_payload_form_follows_the_codec_kind():
    assert payload_form(_fitted("smiles")) == "text"
    assert payload_form(_fitted("smiles+arith-o2")) == "bits"
    assert payload_form(_fitted("smiles"), "hex") == "hex"


def test_string_payloads_refuse_a_binary_form():
    # A string codec's bits come from a flat alphabet code rather than UTF-8,
    # so UTF-8 bits here would misstate the size and not read back.
    with pytest.raises(ValueError, match="no 'hex' form"):
        format_payload(CAFFEINE, "hex")
    assert format_payload(CAFFEINE, "text") == CAFFEINE


def test_payload_text_rejects_malformed_input():
    with pytest.raises(ValueError, match="expected a string of"):
        parse_payload("beef", "bitstring")
    with pytest.raises(ValueError, match="declares"):
        parse_payload("2:ff", "bitstring")
    with pytest.raises(ValueError, match="empty payload"):
        parse_payload("   ", "string")


def test_format_payload_rejects_unknown_forms_and_types():
    with pytest.raises(ValueError, match="unsupported payload format"):
        format_payload(Bits(1, 1), "base64")
    with pytest.raises(ValueError, match="no textual form"):
        format_payload(Bits(1, 1), "text")
    with pytest.raises(TypeError, match="cannot write a payload"):
        format_payload(42, "bits")


# --------------------------------------------------------------------------
# encode / decode / round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["smiles", "selfies-hypervalent", "smiles+huffman", "smiles+arith-o2", "graph-bits"],
)
def test_round_trip_recovers_the_molecule(name):
    result = round_trip(_fitted(name), CAFFEINE)
    assert result.ok, result.reason
    assert result.verdict == "ok"
    assert result.decoded.smiles == CAFFEINE_CANONICAL
    assert result.bits > 0


def test_round_trip_accepts_any_input_representation():
    codec = _fitted("smiles+arith-o2")
    selfies_text = render_molecule(CAFFEINE_CANONICAL, "selfies")
    for text in (CAFFEINE, CAFFEINE_INCHI, selfies_text):
        result = round_trip(codec, text)
        assert result.ok, (text, result.reason)
        assert result.decoded.smiles == CAFFEINE_CANONICAL


def test_round_trip_writes_the_requested_output_format():
    result = round_trip(_fitted("smiles+arith-o2"), CAFFEINE, output_format="inchi")
    assert result.ok
    assert result.decoded.rendered == CAFFEINE_INCHI


def test_payload_decodes_in_a_separate_codec_instance():
    # The property the codebook exists for: a payload means nothing on its own
    # and exactly one thing to an identically fitted codec.
    encoded = encode(_fitted("smiles+arith-o2"), CAFFEINE)
    assert encoded.ok
    decoded = decode(_fitted("smiles+arith-o2"), encoded.text)
    assert decoded.smiles == CAFFEINE_CANONICAL


def test_a_differently_fitted_codec_does_not_silently_agree():
    # Read against the wrong frequency table, a payload may fail, but it may
    # not come back as a plausible different molecule that looks like a
    # success.
    encoded = encode(_fitted("smiles+arith-o2"), CAFFEINE)
    other = resolve_codec("smiles+arith-o2", world="open")
    other.fit(["CCO", "CCC", "CCCC"])
    decoded = decode(other, encoded.text)
    assert decoded.smiles != CAFFEINE_CANONICAL or not decoded.ok


def test_over_budget_is_distinguished_from_lost():
    result = round_trip(_fitted("smiles+arith-o2"), CAFFEINE, budget=4)
    assert result.identical
    assert result.verdict == "over-budget"
    assert result.within_budget is False
    assert "exceeds" in result.reason


def test_encode_reports_an_unreadable_molecule():
    result = encode(_fitted("smiles+arith-o2"), "not a molecule")
    assert not result.ok
    assert result.bits is None
    assert "could not read it" in result.reason


def test_encode_reports_a_molecule_the_codec_declines():
    # The default SELFIES valence rules refuse hypervalent iodine, the measured
    # reason plain SELFIES misses a 99.99% target.
    result = encode(_fitted("selfies"), "Nc1ccc([IH]c2ccc(N)cc2)cc1")
    assert not result.ok
    assert "declined" in result.reason


def test_encode_reports_a_format_mismatch():
    result = encode(_fitted("smiles+arith-o2"), CAFFEINE, input_format="inchi")
    assert not result.ok
    assert "read as inchi" in result.reason


def test_decode_reports_a_malformed_payload():
    result = decode(_fitted("smiles+arith-o2"), "not bits")
    assert not result.ok
    assert "expected a string of" in result.reason


def test_decode_reports_a_payload_with_no_molecule():
    result = decode(_fitted("graph-bits"), "1" * 24)
    assert not result.ok
    assert result.reason


def test_decode_accepts_a_payload_object():
    codec = _fitted("smiles+arith-o2")
    encoded = encode(codec, CAFFEINE)
    assert decode(codec, encoded.payload).smiles == CAFFEINE_CANONICAL


def test_round_trip_surfaces_an_encode_failure():
    result = round_trip(_fitted("smiles+arith-o2"), "@@@")
    assert not result.ok
    assert result.verdict == "failed"
    assert result.decoded is None


def test_encode_wraps_a_raising_codec():
    class Exploding(SmilesCodec):
        def encode(self, smiles):
            msg = "boom"
            raise RuntimeError(msg)

    result = encode(Exploding(), CAFFEINE)
    assert not result.ok
    assert "RuntimeError: boom" in result.reason


def test_decode_wraps_a_raising_codec():
    class Exploding(SmilesCodec):
        def decode(self, payload):
            msg = "boom"
            raise RuntimeError(msg)

    result = decode(Exploding(), CAFFEINE)
    assert not result.ok
    assert "RuntimeError: boom" in result.reason


def test_decode_reports_a_codec_that_returns_junk():
    class Junk(SmilesCodec):
        def decode(self, payload):
            return "not a molecule"

    result = decode(Junk(), CAFFEINE)
    assert not result.ok
    assert "not a parseable molecule" in result.reason


# --------------------------------------------------------------------------
# codebooks
# --------------------------------------------------------------------------


def test_codebook_round_trips_through_a_file(tmp_path):
    book = cb.Codebook(codec=_fitted("smiles+arith-o2"), provenance=cb.Provenance("x", "y", 1, 0))
    path = cb.save(book, tmp_path / "book.oc.xz")
    assert path.exists()
    reloaded = cb.load(path)
    assert reloaded.provenance.codec == "x"
    encoded = encode(book.codec, CAFFEINE)
    assert decode(reloaded.codec, encoded.text).smiles == CAFFEINE_CANONICAL


def test_codebook_rejects_a_file_that_is_not_one(tmp_path):
    path = tmp_path / "junk.oc.xz"
    path.write_bytes(b"not lzma at all")
    with pytest.raises((ValueError, OSError)):
        cb.load(path)


def test_codebook_rejects_a_pickle_that_is_not_a_codebook(tmp_path):
    path = tmp_path / "other.oc.xz"
    with lzma.open(path, "wb") as handle:
        pickle.dump({"magic": "something-else"}, handle)
    with pytest.raises(ValueError, match="not an osmiumchem codebook"):
        cb.load(path)


def test_codebook_from_another_package_version_still_loads(tmp_path):
    # The shipped codebooks would break on every release if the version were
    # a gate. A codebook pickles its own parameters, so FORMAT is the gate.
    path = tmp_path / "old.oc.xz"
    with lzma.open(path, "wb") as handle:
        pickle.dump(
            {
                "magic": cb.MAGIC,
                "format": cb.FORMAT,
                "provenance": {"codec": "smiles", "corpus": "d", "molecules": 1, "seed": 0, "version": "0.0.0"},
            },
            handle,
        )
        pickle.dump(SmilesCodec(), handle)
    assert cb.load(path).provenance.version == "0.0.0"


def test_codebook_rejects_a_foreign_format(tmp_path):
    path = tmp_path / "future.oc.xz"
    with lzma.open(path, "wb") as handle:
        pickle.dump({"magic": cb.MAGIC, "format": cb.FORMAT + 1, "provenance": {}}, handle)
    with pytest.raises(ValueError, match="codebook format"):
        cb.load(path)


def test_fingerprint_codecs_survive_pickling():
    # RDKit's fingerprint generators cannot be pickled, so the codec has to
    # drop and rebuild them; without that, no fingerprint codebook can be saved.
    codec = resolve_codec("ecfp4-512", world="open")
    codec.fit(FIT)
    codec.populate_library(FIT)
    revived = pickle.loads(pickle.dumps(codec))  # noqa: S301 - data this test just produced
    assert revived.encode(FIT[0]) == codec.encode(FIT[0])
    assert revived.decode(revived.encode(FIT[0])) == codec.decode(codec.encode(FIT[0]))


def test_auto_path_is_stable_and_filesystem_safe():
    provenance = cb.Provenance("selfies-hypervalent+arith-o4", "multitask", 100, 7)
    path = cb.auto_path(provenance)
    assert path == cb.auto_path(provenance)
    # Both separators, since the name has to be safe on every platform rather
    # than on the one the test happens to run on.
    assert not set(path.name) & set('/\\:*?"<>|')
    assert path.parent == cb.cache_root()


def test_provenance_key_ignores_incidental_fields():
    base = cb.Provenance("smiles", "d", 10, 0, fitted=10, created=1.0)
    other = cb.Provenance("smiles", "d", 10, 0, fitted=9, created=2.0)
    assert base.key == other.key
    assert "fitted on 10 molecules" in base.summary()


def test_fit_codebook_on_refuses_an_empty_corpus():
    with pytest.raises(ValueError, match="no molecules to fit"):
        cb.fit_codebook_on("smiles+arith-o2", [])
    # Unparseable input strings leave nothing usable either.
    with pytest.raises(ValueError, match="no molecules to fit"):
        cb.fit_codebook_on("smiles+arith-o2", ["not-a-molecule"])


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


def _book(tmp_path, name="smiles+arith-o2"):
    """Write a codebook fitted on the local corpus and return its path."""
    book = cb.Codebook(
        codec=_fitted(name),
        provenance=cb.Provenance(name, "local", len(FIT), 0, fitted=len(FIT)),
    )
    return cb.save(book, tmp_path / f"{name.replace('+', '_')}.oc.xz")


def test_cli_codecs_lists_names(capsys):
    assert run(["codecs"]) == 0
    assert "smiles+arith-o4" in capsys.readouterr().out


def test_cli_codecs_filters_by_family(capsys):
    assert run(["codecs", "--family", "entropy"]) == 0
    out = capsys.readouterr().out
    assert "smiles+arith-o4" in out
    assert "graph-bits" not in out


def test_cli_codecs_reports_an_empty_family(capsys):
    assert run(["codecs", "--family", "nonexistent"]) == 2
    assert "no codec in family" in capsys.readouterr().err


def test_cli_codecs_emits_json(capsys):
    assert run(["codecs", "--family", "entropy", "--json"]) == 0
    assert '"family": "entropy"' in capsys.readouterr().out


def test_cli_encode_then_decode_with_a_saved_codebook(tmp_path, capsys):
    path = _book(tmp_path)
    assert run(["encode", "--codec", "smiles+arith-o2", "--codebook", str(path), CAFFEINE]) == 0
    payload = capsys.readouterr().out.split("\t")[0].strip()

    assert run(["decode", "--codec", "smiles+arith-o2", "--codebook", str(path), payload]) == 0
    assert CAFFEINE_CANONICAL in capsys.readouterr().out


def test_cli_roundtrip_reports_ok(tmp_path, capsys):
    path = _book(tmp_path)
    code = run(["roundtrip", "--codec", "smiles+arith-o2", "--codebook", str(path), CAFFEINE])
    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_cli_roundtrip_json_carries_the_verdict(tmp_path, capsys):
    path = _book(tmp_path)
    run(["roundtrip", "--codec", "smiles+arith-o2", "--codebook", str(path), "--json", CAFFEINE])
    assert '"verdict": "ok"' in capsys.readouterr().out


def test_cli_roundtrip_exits_nonzero_on_failure(tmp_path, capsys):
    path = _book(tmp_path)
    code = run(["roundtrip", "--codec", "smiles+arith-o2", "--codebook", str(path), "@@@"])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_roundtrip_marks_over_budget(tmp_path, capsys):
    path = _book(tmp_path)
    code = run(
        ["roundtrip", "--codec", "smiles+arith-o2", "--codebook", str(path), "--budget", "4", CAFFEINE],
    )
    assert code == 1
    assert "OVER" in capsys.readouterr().out


def test_cli_reads_molecules_from_stdin(tmp_path, capsys, monkeypatch):
    import io

    path = _book(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("CCO\nc1ccccc1\n"))
    assert run(["roundtrip", "--codec", "smiles+arith-o2", "--codebook", str(path), "-"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_cli_requires_something_to_do(tmp_path, capsys):
    path = _book(tmp_path)
    assert run(["encode", "--codec", "smiles+arith-o2", "--codebook", str(path)]) == 2
    assert "no molecules given" in capsys.readouterr().err
    assert run(["decode", "--codec", "smiles+arith-o2", "--codebook", str(path)]) == 2
    assert "no payloads given" in capsys.readouterr().err


def test_cli_reports_an_unknown_codec_without_a_traceback(tmp_path, capsys):
    molecules = tmp_path / "fit.smi"
    molecules.write_text("\n".join(FIT), encoding="utf-8")
    assert run(["encode", "--codec", "nope", "--fit-on", str(molecules), "CCO"]) == 2
    assert "unknown codec" in capsys.readouterr().err


def test_cli_reports_a_bad_payload_format_without_a_traceback(tmp_path, capsys):
    path = _book(tmp_path, "smiles")
    code = run(
        ["encode", "--codec", "smiles", "--codebook", str(path), "--payload-format", "hex", "CCO"],
    )
    assert code == 2
    assert "no 'hex' form" in capsys.readouterr().err


def test_cli_encode_reports_per_molecule_failures(tmp_path, capsys):
    path = _book(tmp_path)
    code = run(["encode", "--codec", "smiles+arith-o2", "--codebook", str(path), "CCO", "@@@"])
    assert code == 1
    out = capsys.readouterr().out
    assert "! @@@" in out


def test_cli_decode_reports_per_payload_failures(tmp_path, capsys):
    path = _book(tmp_path)
    assert run(["decode", "--codec", "smiles+arith-o2", "--codebook", str(path), "zzz"]) == 1
    assert "!" in capsys.readouterr().out


def test_cli_fit_writes_a_codebook(tmp_path, capsys):
    molecules = _fit_file(tmp_path)
    target = tmp_path / "fitted.oc.xz"
    code = run(["fit", "--codec", "smiles+arith-o2", "--fit-on", str(molecules), "--out", str(target)])
    assert code == 0
    assert target.exists()
    assert str(target) in capsys.readouterr().out
    # And the file it wrote is the one --codebook reads back.
    assert run(["encode", "--codec", "smiles+arith-o2", "--codebook", str(target), CAFFEINE]) == 0


def test_cli_fit_emits_json(tmp_path, capsys):
    molecules = _fit_file(tmp_path)
    target = tmp_path / "fitted.oc.xz"
    run(["fit", "--codec", "smiles+arith-o2", "--fit-on", str(molecules), "--out", str(target), "--json"])
    out = capsys.readouterr().out
    assert '"fitted_now": true' in out
    assert '"corpus": "fit"' in out


def test_cli_library_file_is_canonicalised(tmp_path, capsys):
    # A non-canonical library entry collides with its own canonical form and
    # marks the payload ambiguous, so adding a molecule would stop it
    # decoding.
    library = tmp_path / "lib.smi"
    library.write_text(f"{CAFFEINE}\n", encoding="utf-8")
    code = run(
        [
            "roundtrip",
            "--codec",
            "ecfp4-512",
            "--world",
            "open",
            "--fit-on",
            str(_fit_file(tmp_path)),
            "--library",
            str(library),
            CAFFEINE,
        ],
    )
    assert code == 0
    assert CAFFEINE_CANONICAL in capsys.readouterr().out


def test_cli_refuses_a_library_without_anything_to_fit(tmp_path, capsys):
    # An existing codebook has its library baked in, so extra molecules could
    # only be silently dropped, and that would change what decodes.
    library = tmp_path / "lib.smi"
    library.write_text(f"{CAFFEINE}\n", encoding="utf-8")
    assert run(["roundtrip", "--codec", "ecfp4-512", "--library", str(library), CAFFEINE]) == 2
    assert "--library needs --fit-on" in capsys.readouterr().err

    book = _book(tmp_path, "smiles+arith-o2")
    code = run(
        ["roundtrip", "--codec", "smiles+arith-o2", "--codebook", str(book), "--library", str(library), CAFFEINE],
    )
    assert code == 2
    assert "--library needs --fit-on" in capsys.readouterr().err


def test_cli_refuses_two_sources_for_one_codec(tmp_path, capsys):
    book = _book(tmp_path, "smiles+arith-o2")
    code = run(
        [
            "encode",
            "--codec",
            "smiles+arith-o2",
            "--codebook",
            str(book),
            "--fit-on",
            str(_fit_file(tmp_path)),
            CAFFEINE,
        ],
    )
    assert code == 2
    assert "pass one" in capsys.readouterr().err


def _fit_file(tmp_path, name="fit"):
    """Write the local corpus out as a file for ``--fit-on``."""
    path = tmp_path / f"{name}.smi"
    path.write_text("\n".join(FIT) + "\n", encoding="utf-8")
    return path


def test_main_dispatches_to_the_transcode_commands(capsys):
    from osmiumchem.__main__ import main

    assert main(["codecs", "--family", "structural"]) == 0
    assert "graph-bits" in capsys.readouterr().out


def test_cli_fit_falls_back_to_the_cache_with_no_out(tmp_path, monkeypatch, capsys):
    # A codebook nobody can read again cannot decode these payloads, so an
    # in-memory fit with nowhere to go still lands somewhere.
    monkeypatch.setenv("OSMIUMCHEM_CACHE", str(tmp_path))
    assert run(["fit", "--codec", "smiles+arith-o2", "--fit-on", str(_fit_file(tmp_path))]) == 0
    written = list(tmp_path.rglob("*.oc.xz"))
    assert len(written) == 1
    assert str(written[0]) in capsys.readouterr().out


def test_cache_root_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMIUMCHEM_CACHE", str(tmp_path))
    assert cb.cache_root() == tmp_path / "codebooks"
    monkeypatch.delenv("OSMIUMCHEM_CACHE")
    assert cb.cache_root().name == "codebooks"


def test_deepsmiles_is_a_usable_input_and_output():
    pytest.importorskip("deepsmiles")
    text = render_molecule(CAFFEINE_CANONICAL, "deepsmiles")
    assert text
    parsed, used = parse_molecule(text, "deepsmiles")
    assert parsed == CAFFEINE_CANONICAL
    assert used == "deepsmiles"
    result = round_trip(_fitted("smiles+arith-o2"), CAFFEINE, output_format="deepsmiles")
    assert result.ok


def test_library_extras_are_recorded_in_the_provenance():
    # Extra library molecules change what a lookup-decoded codec can decode, so
    # two codebooks that differ only by them must not be mistaken for each
    # other: the digest is part of the cache key.
    # Not one of FIT, which already holds caffeine. This has to be a molecule
    # the open-world library has never seen.
    absent = "CC(=O)Nc1ccc(O)cc1"

    plain = cb.fit_codebook_on("ecfp4-512", FIT, corpus="local", world="open")
    extra = cb.fit_codebook_on("ecfp4-512", FIT, corpus="local", world="open", library=[absent])
    assert extra.provenance.library_digest
    assert not plain.provenance.library_digest
    assert cb.auto_path(plain.provenance) != cb.auto_path(extra.provenance)

    assert round_trip(extra.codec, absent).ok
    assert not round_trip(plain.codec, absent).ok


def test_library_digest_is_order_insensitive_and_empty_when_unused():
    assert cb.library_digest(None) == ""
    assert cb.library_digest([]) == ""
    assert cb.library_digest(["CCO", "CCC"]) == cb.library_digest(["CCC", "CCO"])
    assert cb.library_digest(["CCO"]) != cb.library_digest(["CCC"])


def test_latent_codecs_survive_pickling():
    # LatentCodec holds an encoder closure, which cannot be pickled, so it is
    # dropped and rebuilt from the encoder name. `descriptors` is the one
    # encoder that needs no downloaded weights.
    codec = resolve_codec("descriptors-q16xall", world="open")
    codec.fit(FIT)
    codec.populate_library(FIT)
    revived = pickle.loads(pickle.dumps(codec))  # noqa: S301 - data this test just produced
    assert revived.encode(FIT[0]) == codec.encode(FIT[0])
    assert revived.decode(revived.encode(FIT[0])) == codec.decode(codec.encode(FIT[0]))


def test_encoder_registry_names_the_known_encoders():
    from osmiumchem.codecs.vectors import build_encoder

    assert callable(build_encoder("descriptors"))
    with pytest.raises(KeyError, match="no encoder registered"):
        build_encoder("not-an-encoder")
