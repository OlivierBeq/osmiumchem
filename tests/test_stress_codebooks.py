# -*- coding: utf-8 -*-

"""Load every shipped codebook and put every hard case through all of them.

210 codebooks ship: 30 string and entropy codecs fitted on each of seven
corpora. Each one is a separate frozen table, and a payload is only meaningful
to the table that produced it, so "it works" has to be established per
codebook rather than per codec.

Two things make this a stress test rather than a smoke test. The molecules are
the adversarial fixture, not drug-like examples, so between five and twelve of
them hold tokens absent from any given corpus and reach the escape machinery.
And the payload is carried across as *text*, which is the form that crosses a
process boundary between ``encode`` and ``decode`` and is not what
:func:`~osmiumchem.transcode.round_trip` exercises.
"""

from __future__ import annotations

import pytest

from osmiumchem import api
from osmiumchem import codebook as cb
from osmiumchem.identity import canonicalise_all, same_molecule, silence_rdkit
from tests.hard_cases import HARD_CASES

silence_rdkit()

HARD, _ = canonicalise_all(HARD_CASES)

#: Ordinary drug-like molecules, for the claims that only hold in
#: distribution. Every shipped corpus is drawn from medicinal chemistry, so
#: these are the kind of input the tables were fitted for.
IN_DISTRIBUTION, _ = canonicalise_all(
    [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
        "CC(=O)Nc1ccc(O)cc1",  # paracetamol
        "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # salbutamol
        "Cc1ccc(cc1)S(=O)(=O)N",  # a sulfonamide
        "OC(=O)c1ccccc1O",  # salicylic acid
        "CN1CCC[C@H]1c1cccnc1",  # nicotine
    ],
)

#: Families whose fitted state is counts and alphabets, and so the ones a
#: codebook ships for at all.
SHIPPED_FAMILIES = ("string", "entropy")

#: Codecs whose payload is read against the fitted table. The container codecs
#: (`zlib`, `bz2`, `lzma`) and the plain string codecs are not: their payload
#: carries its own framing, and the fitted state only sets the flat code used
#: for bit accounting.
MODELLED = ("+arith-o", "+huffman", "+flat")


def _grid() -> list[tuple[str, str]]:
    """Return every ``(codec, corpus)`` pair a codebook ships for.

    Read from the filenames rather than the provenance inside each file, so
    collecting the tests does not mean opening 210 of them;
    ``tests/test_codebooks.py`` is what checks the two agree.
    """
    root = cb.bundled_root()
    if not root.is_dir():
        return []
    pairs = []
    for path in sorted(root.glob(f"*{cb.SUFFIX}")):
        codec, separator, corpus = path.name.removesuffix(cb.SUFFIX).rpartition("__")
        if separator:
            pairs.append((codec, corpus))
    return pairs


GRID = _grid()
CORPORA = sorted({corpus for _, corpus in GRID})

pytestmark = pytest.mark.skipif(not GRID, reason="no codebooks have been generated in this checkout")


def _ids(pair: tuple[str, str]) -> str:
    """Name a grid case so a failure says which codebook broke."""
    return f"{pair[0]}@{pair[1]}"


@pytest.mark.parametrize("pair", GRID, ids=_ids)
def test_every_shipped_codebook_handles_every_hard_case(pair):
    codec_name, corpus = pair
    codec = api.load_codec(codec_name, corpus=corpus)
    lossless = codec.codec.describe()["lossless_by_design"]

    failures = []
    for molecule in HARD:
        encoded = codec.encode(molecule)
        # Encoding must never silently decline: a codec that cannot take a
        # molecule says so, and that is already a failure for this family.
        assert encoded.ok, f"{codec_name}@{corpus} refused {molecule}: {encoded.reason}"
        assert encoded.bits > 0
        assert encoded.text

        # Decode from the text, not from the payload object, because the text
        # is what a second process gets.
        decoded = codec.decode(encoded.text)
        if not decoded.ok or not same_molecule(molecule, decoded.smiles, reference_is_canonical=True):
            failures.append((molecule, decoded.smiles))

    if lossless:
        assert failures == [], f"{codec_name}@{corpus} lost {len(failures)} of {len(HARD)}"
    else:
        # Only InChI, whose mobile-hydrogen convention normalises tautomers.
        # It still has to load and encode everywhere; what it may not do is
        # claim to be lossless.
        assert failures, f"{codec_name} is declared lossy but lost nothing; the claim is now wrong"


@pytest.mark.parametrize("corpus", CORPORA)
def test_hard_cases_reach_the_escape_machinery(corpus):
    # If every hard token were already in the alphabet, the test above would
    # not be stressing escapes at all, so prove some are not.
    codec = api.load_codec("smiles+arith-o4", corpus=corpus).codec
    observed = codec._alphabet.observed
    needing_escapes = [molecule for molecule in HARD if set(codec._tokens(molecule)) - observed]
    assert needing_escapes, f"{corpus}: no hard case holds an unseen token"


@pytest.mark.parametrize("corpus", CORPORA)
def test_each_corpus_ships_every_shippable_codec(corpus):
    expected = {record["name"] for record in api.describe_codecs() if record["family"] in SHIPPED_FAMILIES}
    shipped = {codec for codec, label in GRID if label == corpus}
    assert shipped == expected


def test_the_grid_is_the_whole_product():
    assert len(GRID) == len(CORPORA) * len({codec for codec, _ in GRID})
    assert sorted(CORPORA) == api.list_codebooks()


@pytest.mark.parametrize(
    "codec_name",
    sorted({codec for codec, _ in GRID if any(mark in codec for mark in MODELLED)}),
)
def test_a_modelled_payload_means_nothing_to_another_corpus(codec_name):
    # The claim the whole codebook machinery rests on: these payloads are read
    # against the table that wrote them, so reading one against a different
    # table gives a different molecule or nothing.
    if len(CORPORA) < 2:
        pytest.skip("needs two corpora to compare")
    writer = api.load_codec(codec_name, corpus=CORPORA[0])
    reader = api.load_codec(codec_name, corpus=CORPORA[1])

    survived = 0
    for molecule in HARD:
        decoded = reader.decode(writer.encode(molecule).text)
        if decoded.ok and decoded.smiles == molecule:
            survived += 1
    # Not zero: a one-token message such as methane can land on the same bits
    # under both tables by coincidence. The point is that it is the exception.
    assert survived <= len(HARD) // 5, f"{codec_name}: {survived}/{len(HARD)} crossed tables intact"


@pytest.mark.parametrize("corpus", CORPORA)
def test_more_context_never_costs_more_on_any_corpus(corpus):
    # Each corpus is a different table, and a mis-fitted one shows up here as
    # an inversion. True even on the adversarial fixture, because an escaped
    # run is coded under the same model as everything else.
    ladder = [f"smiles+arith-o{order}" for order in range(5)]
    totals = [sum(api.load_codec(name, corpus=corpus).encode(m).bits for m in HARD) for name in ladder]
    assert totals == sorted(totals, reverse=True), dict(zip(ladder, totals, strict=True))


@pytest.mark.parametrize("corpus", CORPORA)
def test_modelling_pays_on_molecules_the_corpus_resembles(corpus):
    # The ladder's headline, which only holds in distribution: skew beats a
    # flat code, and fractional bits beat whole ones.
    #
    # Deliberately not asserted on HARD, where Huffman loses to the flat code.
    # Most of that input is escaped literals, coded one symbol at a time
    # through a table whose short codes went to tokens it does not contain,
    # while a flat code charges every symbol the same ceil(log2(|alphabet|)).
    flat, huffman, order0 = (
        sum(api.load_codec(name, corpus=corpus).encode(m).bits for m in IN_DISTRIBUTION)
        for name in ("smiles+flat", "smiles+huffman", "smiles+arith-o0")
    )
    assert flat > huffman >= order0, {"flat": flat, "huffman": huffman, "arith-o0": order0}
