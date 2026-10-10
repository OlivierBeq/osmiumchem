# Encoding and decoding one molecule

`docs/RESULTS.md` compares techniques over 200,000 molecules. This is the other
direction: taking one of those codec names and running *your* molecule through
it.

```console
$ osmiumchem roundtrip --codec smiles+arith-o4 "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
loaded codebook: smiles+arith-o4 fitted on 99999 molecules from union (...)
OK    57 bits  Cn1c(=O)c2c(ncn2C)n(C)c1=O
```

Caffeine, encoded and recovered, in 57 bits.

```python
>>> import osmiumchem as oc
>>> oc.load_codec("smiles+arith-o4").round_trip("CN1C=NC2=C1C(=O)N(C)C(=O)N2C").bits
57
```

Nothing was downloaded and nothing was fitted. Either interface does the whole
job, and they read the same codebooks, so a payload written by one is read by
the other. The commands come first below; [From Python](#from-python) covers
the same ground in the API, which is the better side to be on if you have your
own corpus to fit.

## Contents

**Command line** — [The five commands](#the-five-commands) ·
[Codebooks](#codebooks-the-one-thing-to-understand) ·
[Molecules in, molecules out](#molecules-in-molecules-out) ·
[Payloads as text](#payloads-as-text) ·
[What the verdict distinguishes](#what-the-verdict-distinguishes) ·
[Fingerprints decode by lookup](#fingerprints-decode-by-lookup)

**Python** — [Three verbs](#from-python) ·
[Fitting on your own series](#fit-on-the-chemistry-you-intend-to-encode) ·
[Batches](#batches-use-the-codecs-own-batch-path) ·
[Failures](#failures-are-results-not-exceptions) ·
[Picking a codec](#picking-a-codec)

## The five commands

| command | what it does |
| --- | --- |
| `codecs` | list every name the other commands accept |
| `roundtrip` | encode, decode, and say whether the same molecule came back |
| `encode` | molecule → payload, with its exact bit cost |
| `decode` | payload → molecule |
| `fit` | fit a codec on your own molecules and save its codebook |

Any codec name from the results tables works, and `osmiumchem codecs` prints
all 57 with their families. The benchmark itself is not one of these commands:
it reads corpora from files, which the package knows nothing about, so it lives
in `scripts/benchmark.py`.

## Codebooks, the one thing to understand

A fitted codec is a *shared artefact*. `smiles+arith-o4` is a frozen table of
token frequencies, and a payload produced under one table is meaningless under
another: the same 57 bits read against a different table decode to a different
molecule, or to nothing at all. So `encode` and `decode` have to be holding the
same one.

This is handled by shipping the tables. One codebook per codec per benchmark
corpus is built into the package, so both commands load the same file and
cannot disagree:

```console
$ osmiumchem encode --codec smiles+arith-o4 "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
101000100101011100101101110110111110001000100100101000000    57 bits

$ osmiumchem decode --codec smiles+arith-o4 101000100101011100101101110110111110001000100100101000000
Cn1c(=O)c2c(ncn2C)n(C)c1=O    57 bits
```

`--corpus` chooses which corpus the table was fitted on, defaulting to `union`,
the broadest. The labels are all that remains of the corpora themselves, which
are not redistributable and are not in the package:

```console
$ osmiumchem codecs --corpora
cyp450
hiv
multitask
protacs
union
warheads
zinc250k

$ osmiumchem roundtrip --codec smiles+arith-o4 --corpus protacs "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
```

Only the `string` and `entropy` codecs ship a codebook. A fingerprint or latent
codec decodes by looking a payload up in a *library of molecules*, so its
fitted state is a corpus; shipping one would ship the data. Use `--fit-on` for
those.

### Fitting your own instead

`--fit-on FILE` fits on the molecules in a file, one SMILES per line, rather
than loading a shipped table. `fit` does it once and keeps the result, which
`--codebook` reads back:

```console
$ osmiumchem fit --codec smiles+arith-o4 --fit-on my_series.smi --out series.oc.xz
fitted codebook: smiles+arith-o4 fitted on 2500 molecules from my_series (...)
series.oc.xz

$ osmiumchem encode --codec smiles+arith-o4 --codebook series.oc.xz "CCO"
```

With no `--out`, `fit` writes to `~/.cache/osmiumchem/codebooks`, which
`OSMIUMCHEM_CACHE` moves. A codebook nobody can read again cannot decode the
payloads it produced, so it always lands somewhere.

> Codebooks are pickles, so loading one runs the code inside it. The shipped
> ones are part of the package and as trusted as the code reading them;
> anything else is not. Load only codebooks you made yourself.

## Molecules in, molecules out

Input is not restricted to SMILES. `--in` accepts `smiles`, `selfies`, `inchi`
and `deepsmiles`, and defaults to `auto`:

```console
$ osmiumchem roundtrip --codec smiles+arith-o4 \
    "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"
OK    57 bits  Cn1c(=O)c2c(ncn2C)n(C)c1=O
```

Detection is ordered, and the order is documented because it cannot always be
right: `[C][C][O]` is a valid SELFIES string *and* a valid SMILES. Detection
looks for tokens that are only legal in SELFIES (`[Ring1]`, `[Branch2]`,
`[=C]`) and otherwise prefers SMILES. The format it settled on is always
reported back, and `--in` overrides it.

`--out` writes the recovered molecule as `smiles`, `smiles-kekule`, `selfies`,
`inchi`, `inchikey` or `deepsmiles`. Pass `-` to read molecules from stdin, one
per line.

### Why the output is not the string you typed

`CN1C=NC2=C1C(=O)N(C)C(=O)N2C` comes back as `Cn1c(=O)c2c(ncn2C)n(C)c1=O`, and
that is a success rather than a near miss. Both strings are caffeine. The
codecs encode the canonical form, so the canonical form is what returns, and
the verdict uses the same molecular identity test the benchmark scores with: a
molecule that passes here would have counted there.

## Payloads as text

A payload is an exact number of bits, and the bit length is part of the
payload, since leading zeros carry information and payloads are rarely a whole
number of bytes. Both textual forms are therefore exactly reversible.

| `--payload-format` | caffeine under `smiles+arith-o4` |
| --- | --- |
| `bits` | `101000100101011100101101110110111110001000100100101000000` |
| `hex` | `57:144ae5bb7c44940` — `<bit length>:<hex digits>` |
| `text` | for string codecs, where the payload *is* the text |

A string codec refuses `bits` and `hex` rather than emitting UTF-8. Those
codecs are charged a flat code over their fitted alphabet rather than 8 bits
per byte, so UTF-8 would both misstate the size and fail to read back.

## What the verdict distinguishes

```console
$ osmiumchem roundtrip --codec smiles+arith-o4 --budget 40 - < molecules.smi
OK    16 bits  CCO
OVER  57 bits  Cn1c(=O)c2c(ncn2C)n(C)c1=O  round trip exact, but 57 bits exceeds the 40-bit budget
FAIL  - bits  -  could not read it as a molecule in any supported representation
```

`OVER` is not `FAIL`. Losing a molecule and keeping it in too many bits are
different failures, and the whole PROTAC result in `docs/RESULTS.md` turns on
telling them apart.

The exit status is 0 only when every molecule is `OK`. `--json` emits one
object per molecule with `verdict`, `bits`, `within_budget`, `payload` and
`recovered`.

## Fingerprints decode by lookup

The fingerprint and latent codecs have no decoder; they are resolved against a
library. That is also why none of them ships a codebook: the library is a
corpus. `--fit-on` supplies one and `--library` adds molecules explicitly,
which is enough to reproduce the benchmark's fingerprint ceiling on two
molecules:

```console
$ cat epimers.smi
CC(C)c1cccc([C@@H](O)CCN)c1
CC(C)c1cccc([C@H](O)CCN)c1

$ osmiumchem encode --codec ecfp4-512 --world open --fit-on epimers.smi \
    --payload-format hex - < epimers.smi
512:40010002400408008000820000000000...    512 bits
512:40010002400408008000820000000000...    512 bits   # byte-identical

$ osmiumchem roundtrip --codec ecfp4-512 --world open --fit-on epimers.smi - < epimers.smi
FAIL  512 bits  -  ecfp4-512 recovered no molecule from this payload
FAIL  512 bits  -  ecfp4-512 recovered no molecule from this payload

$ osmiumchem roundtrip --codec ecfp4-512-chiral --world open --fit-on epimers.smi - < epimers.smi
OK    512 bits  CC(C)c1cccc([C@@H](O)CCN)c1
OK    512 bits  CC(C)c1cccc([C@H](O)CCN)c1
```

Two epimers hash to the same 512 bits under stereo-blind ECFP4, so the library
marks that payload ambiguous and refuses it rather than returning whichever
molecule was inserted first. Switching on `includeChirality` separates them at
no cost in bits. That is the 99.8920% ceiling in `docs/RESULTS.md`, on two
molecules you can inspect.

## From Python

The commands are a thin layer over the package, and the package is three verbs:
load a codec, fit one, save one.

```python
import osmiumchem as oc

codec = oc.load_codec("smiles+arith-o4")
result = codec.round_trip("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")

result.verdict          # 'ok'
result.bits             # 57
result.encoded.text     # '101000100101...'
result.decoded.smiles   # 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'
```

`load_codec` takes either a codec name or a path to a codebook file, so one
call covers both the shipped tables and anything you fitted yourself:

```python
oc.load_codec("smiles+arith-o4")                     # shipped, corpus='union'
oc.load_codec("smiles+arith-o4", corpus="protacs")   # shipped, another corpus
oc.load_codec("series.oc.xz")                        # one you wrote

oc.list_codebooks()                     # corpora with a shipped codebook
oc.list_codebooks("smiles+arith-o4")    # ...for one codec
```

The object it returns holds the fitted codec — the codebook — and the defaults
you would otherwise repeat at every call, so a pipeline sets them once:

```python
codec = oc.load_codec(
    "smiles+arith-o4",
    corpus="protacs",
    input_format="inchi", output_format="inchikey", budget=512,
)
```

Those keywords are the `--codec/--corpus/--world/--library` group, spelled out,
with the same defaults and the same files. A payload produced at a shell prompt
therefore decodes here, and the other way round. `codec.summary()` says what is
fitted and where it came from.

### Fit on the chemistry you intend to encode

`fit_codec` fits on molecules you pass instead of loading a shipped table. It
is usually the better choice from Python, and the thing the command line can
only do from a file. The fitted table *is* the code, so it pays to match your
series. On 100 held-out PROTACs under `smiles+arith-o4`:

| fitted on | median bits | max bits | recovered |
| --- | --- | --- | --- |
| `union`, 100,000 molecules (the shipped default) | 208 | 494 | 100/100 |
| 2,500 other PROTACs (`fit_codec`) | **146** | 376 | 100/100 |

```python
codec = oc.fit_codec("smiles+arith-o4", my_series)   # canonicalised for you
oc.save_codec(codec, "series.oc.xz")                 # keep it, or it is gone
codec = oc.load_codec("series.oc.xz")                # anywhere, any process
```

Nothing is cached automatically here, because a list of molecules has no
reproducible name the way a corpus label does. Saving is on you, and it is what
another process needs in order to decode these payloads.

### Batches use the codec's own batch path

`encode_many`, `decode_many` and `round_trip_many` return one result per input,
in order, with failures in place. They are not loops for convenience: a latent
codec evaluates five hundred molecules in roughly the time it takes to evaluate
one, and these route through that path where the codec has one.

```python
results = codec.round_trip_many(smiles_list)
sum(r.verdict == "ok" for r in results) / len(results)
```

### Failures are results, not exceptions

A molecule that cannot be read, a codec that declines, a payload that resolves
to nothing: each comes back as a result with `ok = False` and a `reason`, so a
loop over a few thousand molecules needs no `try` block.

Exceptions are kept for misuse: an unknown codec name (`KeyError`, listing the
closest spellings), a codec or corpus with no shipped codebook (`KeyError`,
listing what there is) or an unsupported format (`ValueError`, raised before
any fitting is paid for).

### Picking a codec

```python
oc.list_codecs()                            # all 57 names
oc.list_codecs(family="entropy")            # one family
oc.describe_codecs(family="fingerprint")    # kind, size model, notes
```

Read `lossless_by_design` in those records before expecting a round trip: a
fingerprint cannot invert its own hash, and decodes only by looking the payload
up in a library.

The lower-level functions — `oc.encode`, `oc.decode`, `oc.round_trip` and their
`_many` forms — take a codec as their first argument instead of carrying
defaults, and `codec.codec` is the fitted codec to hand them. Imports are lazy,
so `import osmiumchem` costs about 60 ms and pulls in RDKit only when a name
that needs it is first used.
