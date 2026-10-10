# -*- coding: utf-8 -*-

"""Tests for the Python interface: a codec by name, from import to payload."""

import pytest

import osmiumchem as oc
from osmiumchem import api
from osmiumchem import codebook as cb
from osmiumchem.codecs import catalogue, resolve_codec
from osmiumchem.codecs.base import BaseCodec
from osmiumchem.codecs.vectors import BIT_BUDGET
from osmiumchem.identity import silence_rdkit
from osmiumchem.transcode import decode_many, encode_many, round_trip, round_trip_many

silence_rdkit()

CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
CAFFEINE_CANONICAL = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

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


def _local(name="smiles+arith-o2", **kwargs):
    """Fit a codec on the small local corpus, with nothing on disk."""
    return api.fit_codec(name, FIT, label="local", **kwargs)


# --------------------------------------------------------------------------
# finding a codec to ask for
# --------------------------------------------------------------------------


def test_list_codecs_matches_the_catalogue():
    names = api.list_codecs()
    assert names == sorted(catalogue())
    assert "smiles+arith-o4" in names


def test_list_codecs_filters_by_family():
    entropy = api.list_codecs(family="entropy")
    assert "smiles+arith-o4" in entropy
    assert "graph-bits" not in entropy
    assert api.list_codecs(family="no-such-family") == []


def test_every_listed_name_opens():
    # Every advertised name has to resolve, or the interface is offering
    # codecs it cannot build.
    for name in api.list_codecs():
        assert resolve_codec(name).name == name


def test_describe_codecs_carries_what_a_choice_needs():
    records = api.describe_codecs()
    assert len(records) == len(catalogue())
    assert [(item["family"], item["name"]) for item in records] == sorted(
        (item["family"], item["name"]) for item in records
    )
    fingerprints = api.describe_codecs(family="fingerprint")
    assert fingerprints
    for record in fingerprints:
        # A hash is not invertible, which is worth knowing before expecting a
        # round trip.
        assert record["lossless_by_design"] is False
        assert set(record) >= {"name", "family", "kind", "size_model", "notes"}


# --------------------------------------------------------------------------
# fitting on your own molecules
# --------------------------------------------------------------------------


def test_fit_codec_round_trips_a_molecule():
    codec = _local()
    result = codec.round_trip(CAFFEINE)
    assert result.verdict == "ok"
    assert result.decoded.smiles == CAFFEINE_CANONICAL
    assert result.bits > 0


def test_fit_codec_canonicalises_and_drops_what_will_not_parse():
    codec = api.fit_codec("smiles+arith-o2", ["OCC", "nonsense", "c1ccccc1"])
    assert codec.provenance.fitted == 2
    assert codec.round_trip("CCO").ok


def test_fit_codec_refuses_a_corpus_with_nothing_in_it():
    with pytest.raises(ValueError, match="no molecules to fit"):
        api.fit_codec("smiles+arith-o2", ["nonsense", ""])


def test_fit_codec_labels_different_corpora_differently():
    # The label is the provenance and the provenance is the cache key, so two
    # corpora sharing one would be mistaken for each other.
    first = api.fit_codec("smiles+arith-o2", FIT)
    second = api.fit_codec("smiles+arith-o2", FIT[:5])
    assert first.provenance.corpus.startswith("custom:")
    assert first.provenance.corpus != second.provenance.corpus
    assert _local().provenance.corpus == "local"


def test_fit_codec_is_not_cached():
    # "This list of molecules" has no reproducible name, so a shared-cache
    # entry could be picked up by a call that meant something else.
    assert _local().path is None
    assert "not cached" in _local().summary()


# --------------------------------------------------------------------------
# the defaults a transcoder carries
# --------------------------------------------------------------------------


def test_defaults_apply_to_every_call_and_are_overridable():
    codec = _local(output_format="selfies")
    assert codec.round_trip(CAFFEINE).decoded.rendered.startswith("[")
    assert codec.round_trip(CAFFEINE, output_format="smiles").decoded.rendered == CAFFEINE_CANONICAL


def test_the_budget_is_the_projects_ceiling_by_default():
    codec = _local()
    assert codec.budget == BIT_BUDGET
    assert codec.round_trip(CAFFEINE).within_budget


def test_a_tighter_budget_separates_over_budget_from_lost():
    codec = _local(budget=8)
    result = codec.round_trip(CAFFEINE)
    assert result.verdict == "over-budget"
    assert result.identical and not result.within_budget
    assert codec.round_trip(CAFFEINE, budget=BIT_BUDGET).verdict == "ok"


def test_input_format_default_reads_other_representations():
    codec = _local(input_format="selfies")
    selfies = codec.round_trip(CAFFEINE, input_format="smiles").decoded.smiles
    assert selfies == CAFFEINE_CANONICAL
    # Declared as SELFIES, a SMILES string is not readable as one.
    assert not codec.round_trip(CAFFEINE).ok


def test_payload_format_default_reaches_the_payload_text():
    codec = _local(payload_format="hex")
    encoded = codec.encode(CAFFEINE)
    assert encoded.form == "hex"
    assert encoded.text.startswith(f"{encoded.bits}:")
    assert codec.decode(encoded.text).smiles == CAFFEINE_CANONICAL


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_format": "morse"}, "input format"),
        ({"output_format": "morse"}, "output format"),
        ({"payload_format": "morse"}, "payload format"),
    ],
)
def test_unsupported_formats_are_refused_before_any_fitting(kwargs, message):
    # No codebook ships for this corpus, so an error naming the format shows
    # the check ran first. A typo should not cost a fit.
    with pytest.raises(ValueError, match=message):
        api.load_codec("smiles+arith-o2", corpus="no-such-corpus", **kwargs)
    with pytest.raises(ValueError, match=message):
        api.fit_codec("smiles+arith-o2", FIT, **kwargs)
    with pytest.raises(ValueError, match=message):
        api.Transcoder(codebook=_local().codebook, **kwargs)


def test_repr_and_summary_say_what_is_fitted():
    codec = _local()
    assert "smiles+arith-o2" in repr(codec)
    assert "local" in repr(codec)
    assert "fitted on 10 molecules" in codec.summary()


def test_the_codec_itself_stays_reachable():
    codec = _local()
    assert codec.name == "smiles+arith-o2"
    assert codec.codec is codec.codebook.codec
    # The lower-level functions take the codec, and have to agree with the
    # wrapper.
    assert round_trip(codec.codec, CAFFEINE).bits == codec.round_trip(CAFFEINE).bits


# --------------------------------------------------------------------------
# saving the fitted state, so another process can decode
# --------------------------------------------------------------------------


def test_saved_codebook_decodes_what_the_original_encoded(tmp_path):
    original = _local()
    path = original.save(tmp_path / "local.oc.xz")
    assert path.exists()
    assert original.path == path

    reloaded = api.Transcoder.load(str(path))
    payload = original.encode(CAFFEINE).text
    assert reloaded.decode(payload).smiles == CAFFEINE_CANONICAL


def test_load_takes_the_same_defaults(tmp_path):
    path = _local().save(tmp_path / "local.oc.xz")
    reloaded = api.Transcoder.load(path, budget=12, output_format="inchikey")
    assert reloaded.budget == 12
    assert reloaded.round_trip(CAFFEINE).decoded.rendered.endswith("-N")


def test_save_without_a_path_uses_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMIUMCHEM_CACHE", str(tmp_path))
    codec = _local()
    written = codec.save()
    assert written == cb.auto_path(codec.provenance)
    assert written.exists()


# --------------------------------------------------------------------------
# loading a codec: shipped with the package, or from a file you wrote
# --------------------------------------------------------------------------


def _ship(tmp_path, monkeypatch, codec="smiles+arith-o2", corpus="demo"):
    """Put a codebook where a shipped one would be, and point the lookup there."""
    monkeypatch.setattr(cb, "bundled_root", lambda: tmp_path)
    book = api.fit_codec(codec, FIT, label=corpus).codebook
    return cb.save(book, cb.bundled_path(codec, corpus))


def test_load_codec_takes_a_shipped_codebook(tmp_path, monkeypatch):
    target = _ship(tmp_path, monkeypatch)
    codec = api.load_codec("smiles+arith-o2", corpus="demo")
    assert not codec.fitted_now
    assert codec.path == target
    assert codec.provenance.corpus == "demo"
    assert codec.round_trip(CAFFEINE).decoded.smiles == CAFFEINE_CANONICAL


def test_shipped_codebooks_are_listed_by_corpus(tmp_path, monkeypatch):
    _ship(tmp_path, monkeypatch, corpus="demo")
    _ship(tmp_path, monkeypatch, corpus="other")
    assert api.list_codebooks() == ["demo", "other"]
    assert api.list_codebooks("smiles+arith-o2") == ["demo", "other"]
    assert api.list_codebooks("smiles+arith-o4") == []


def test_save_codec_then_load_codec_shares_one_table(tmp_path):
    # The custom-codec contract: fit, save, and another process decodes it.
    target = tmp_path / "series.oc.xz"
    fitted = _local()
    assert api.save_codec(fitted, target) == target

    reloaded = api.load_codec(target)
    assert not reloaded.fitted_now
    assert reloaded.provenance.corpus == "local"
    payload = fitted.encode(CAFFEINE)
    assert reloaded.decode(payload.text).smiles == CAFFEINE_CANONICAL
    assert reloaded.encode(CAFFEINE).bits == payload.bits


def test_load_codec_reads_a_path_given_as_a_string(tmp_path):
    target = tmp_path / "series.oc.xz"
    api.save_codec(_local(), target)
    # Both the suffix and the file existing are enough to mean "a path".
    assert api.load_codec(str(target)).provenance.corpus == "local"
    assert api.load_codec(target).provenance.corpus == "local"


def test_load_codec_refuses_a_path_with_a_corpus(tmp_path):
    target = tmp_path / "series.oc.xz"
    api.save_codec(_local(), target)
    with pytest.raises(ValueError, match="own provenance"):
        api.load_codec(target, corpus="union")


def test_load_codec_names_a_missing_file():
    with pytest.raises(FileNotFoundError, match="no such codebook"):
        api.load_codec("absent.oc.xz")


def test_load_codec_lists_the_corpora_it_has(tmp_path, monkeypatch):
    _ship(tmp_path, monkeypatch, corpus="demo")
    with pytest.raises(KeyError, match="available: demo"):
        api.load_codec("smiles+arith-o2", corpus="nope")


def test_load_codec_explains_why_a_fingerprint_ships_nothing(tmp_path, monkeypatch):
    # Their fitted state is a library of molecules, so shipping one would ship
    # the corpus. The message has to send the caller to fit_codec.
    _ship(tmp_path, monkeypatch, corpus="demo")
    with pytest.raises(KeyError, match="fit_codec"):
        api.load_codec("ecfp4-512")


def test_load_codec_treats_a_mistyped_name_as_a_typo(tmp_path, monkeypatch):
    # Not as "nothing ships for it": the name is resolved first, so the message
    # is the list of near spellings.
    _ship(tmp_path, monkeypatch, corpus="demo")
    with pytest.raises(KeyError, match="unknown codec"):
        api.load_codec("smiles+arith-o5")


def test_a_fingerprint_can_still_be_fitted_with_a_library():
    # An open-world library holds the fitting split and nothing else, so a
    # molecule it has not seen has nothing to resolve to until it is added.
    outsider = "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"  # salbutamol, not in FIT
    plain = api.fit_codec("ecfp4-512", FIT, label="local", world="open")
    extended = api.fit_codec("ecfp4-512", FIT, label="local", world="open", library=[outsider])
    assert not plain.round_trip(outsider).ok
    assert extended.round_trip(outsider).ok


def test_fit_codec_rejects_an_unknown_name():
    with pytest.raises(KeyError, match="smiles\\+arith-o4"):
        api.fit_codec("smiles+arith-o5", FIT)


# --------------------------------------------------------------------------
# batches, and the codec batch paths they are there to use
# --------------------------------------------------------------------------


class _Batched(BaseCodec):
    """A codec that records which path a caller took.

    A latent codec encodes a few hundred molecules in about the time it takes
    to encode one, so a wrapper that looped instead of batching would make
    those codecs unusable here.
    """

    name = "batch-probe"
    kind = "bitstring"

    def __init__(self, inner):
        self.inner = inner
        self.batches = []
        self.singles = []

    def encode(self, smiles):
        self.singles.append(smiles)
        return self.inner.encode(smiles)

    def encode_many(self, smiles):
        self.batches.append(list(smiles))
        return [self.inner.encode(item) for item in smiles]

    def decode(self, payload):
        return self.inner.decode(payload)

    def decode_many(self, payloads):
        self.batches.append(list(payloads))
        return [self.inner.decode(payload) for payload in payloads]

    def payload_bits(self, payload):
        return self.inner.payload_bits(payload)


class _RaisingBatch(_Batched):
    """A codec whose batch path fails outright."""

    def encode_many(self, smiles):
        msg = "no batch today"
        raise RuntimeError(msg)

    def decode_many(self, payloads):
        msg = "no batch today"
        raise RuntimeError(msg)


class _ShortBatch(_Batched):
    """A codec whose batch path returns the wrong number of results."""

    def encode_many(self, smiles):
        return [self.inner.encode(item) for item in smiles][:-1]

    def decode_many(self, payloads):
        return [self.inner.decode(payload) for payload in payloads][:-1]


def _probe(kind=_Batched):
    """Return a probe wrapping a codec fitted on the local corpus."""
    inner = resolve_codec("smiles+arith-o2")
    inner.fit(FIT)
    return kind(inner)


def test_batches_keep_order_and_isolate_failures():
    codec = _local()
    molecules = [CAFFEINE, "nonsense", "CCO"]
    results = codec.round_trip_many(molecules)
    assert [result.verdict for result in results] == ["ok", "failed", "ok"]
    assert [result.encoded.input for result in results] == molecules
    assert results[1].decoded is None
    assert "any supported representation" in results[1].reason


def test_batches_agree_with_one_at_a_time():
    codec = _local()
    molecules = [CAFFEINE, "CCO", "nonsense", "c1ccccc1O"]
    batched = codec.round_trip_many(molecules)
    singly = [codec.round_trip(molecule) for molecule in molecules]
    assert [(item.verdict, item.bits) for item in batched] == [(item.verdict, item.bits) for item in singly]


def test_an_empty_batch_is_an_empty_list():
    codec = _local()
    assert codec.encode_many([]) == []
    assert codec.decode_many([]) == []
    assert codec.round_trip_many([]) == []


def test_the_codecs_batch_path_is_the_one_used():
    codec = _probe()
    results = encode_many(codec, [CAFFEINE, "CCO"])
    assert all(result.ok for result in results)
    assert codec.batches == [[CAFFEINE_CANONICAL, "CCO"]]
    assert codec.singles == []

    decoded = decode_many(codec, [result.payload for result in results])
    assert [item.smiles for item in decoded] == [CAFFEINE_CANONICAL, "CCO"]
    assert len(codec.batches) == 2


def test_unreadable_molecules_never_reach_the_batch():
    codec = _probe()
    results = encode_many(codec, ["nonsense", CAFFEINE])
    assert [result.ok for result in results] == [False, True]
    assert codec.batches == [[CAFFEINE_CANONICAL]]


@pytest.mark.parametrize("kind", [_RaisingBatch, _ShortBatch])
def test_a_broken_batch_path_falls_back_one_at_a_time(kind):
    # Trusting either failure mode loses the whole list. Retrying per molecule
    # costs time and keeps the results.
    codec = _probe(kind)
    results = round_trip_many(codec, [CAFFEINE, "CCO"])
    assert [result.verdict for result in results] == ["ok", "ok"]
    assert codec.singles == [CAFFEINE_CANONICAL, "CCO"]


def test_a_failure_inside_the_fallback_is_reported_not_raised():
    codec = _probe(_RaisingBatch)

    def refuse(smiles):
        msg = "cannot encode that"
        raise RuntimeError(msg)

    codec.encode = refuse
    results = encode_many(codec, [CAFFEINE])
    assert not results[0].ok
    assert "cannot encode that" in results[0].reason


def test_decoding_to_another_molecule_is_a_failure_not_a_pass():
    # The verdict is molecular identity, not "something came back". A codec
    # fitted on another corpus reads the same bits as a different molecule, and
    # that counts as a loss.
    codec = _probe()
    codec.decode = lambda payload: "CCO"
    result = round_trip(codec, CAFFEINE)
    assert result.verdict == "failed"
    assert result.reason == "decoded to a different molecule"


def test_a_malformed_payload_in_a_batch_is_isolated():
    codec = _probe()
    good = encode_many(codec, [CAFFEINE])[0]
    results = decode_many(codec, ["not a payload", good.text])
    assert [result.ok for result in results] == [False, True]
    assert "bit-sequence payload" in results[0].reason


def test_a_batch_decode_that_recovers_nothing_is_a_failure():
    codec = _probe()
    codec.decode_many = lambda payloads: [None for _ in payloads]
    results = round_trip_many(codec, [CAFFEINE])
    assert results[0].verdict == "failed"
    assert "recovered no molecule" in results[0].reason


# --------------------------------------------------------------------------
# the package surface
# --------------------------------------------------------------------------


def test_all_and_the_lazy_table_agree():
    # Two lists of the same names, one for tooling and one for the import hook.
    assert set(oc.__all__) == set(oc._EXPORTS) | {"__version__"}
    assert oc.__all__ == sorted(oc.__all__)


def test_every_advertised_name_is_importable():
    for name in oc.__all__:
        assert getattr(oc, name) is not None


def test_the_quickstart_in_the_docstring_works():
    codec = oc.fit_codec("smiles+arith-o2", FIT)
    result = codec.round_trip(CAFFEINE)
    assert (result.verdict, result.decoded.smiles) == ("ok", CAFFEINE_CANONICAL)


def test_submodules_resolve_without_a_second_import():
    assert oc.codebook.save.__name__ == "save"
    assert oc.transcode.__name__ == "osmiumchem.transcode"


def test_an_unknown_attribute_still_raises():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = oc.nope


def test_dir_lists_the_lazy_names():
    listed = dir(oc)
    assert listed == sorted(listed)
    assert {"load_codec", "save_codec", "fit_codec", "list_codecs", "Transcoder", "api"} <= set(listed)
