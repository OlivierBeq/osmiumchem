<div align="center">

  # osmiumchem

  **How few bits does a molecule need, if you have to get it back exactly?**

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://github.com/OlivierBeq/osmiumchem/actions/workflows/ci.yml/badge.svg)](https://github.com/OlivierBeq/osmiumchem/actions/workflows/ci.yml)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
</div>

A benchmark of molecular encodings judged on two axes at once:

1. **Fidelity** — does `decode(encode(mol))` give back *the same molecule*, at
   a rate of at least **99.99%**?
2. **Frugality** — does the payload fit in **1000 bits**?

Most representations are measured on one axis or the other. Fingerprints are
compared on how well they support property prediction, SELFIES is praised for
never producing an invalid string, and latent spaces are reported in dimensions
rather than bits. This benchmark asks the question those metrics skip: *can you
get the molecule back, and what did it cost you?*

**Short answer:** order-4 arithmetic coding over canonical SMILES recovers
every one of 200 000 drug-like molecules exactly, at a median of **70 bits**, a
fourteenth of the budget.

## Contents

- [Installation](#-installation)
- [Quick start](#-quick-start) — one molecule, then the full benchmark
- [How fidelity is judged](#-how-fidelity-is-judged)
- [What the benchmark shows](#-what-the-benchmark-shows) — the findings
- [Codecs](#-codecs) · [Datasets](#-datasets) · [Documentation](#-documentation)

## 📦 Installation

```bash
pip install osmiumchem

# optional extras
pip install "osmiumchem[representations]"  # DeepSMILES
pip install "osmiumchem[neural]"           # CDDD and GROVER encoders
pip install "osmiumchem[parquet]"          # Parquet corpora, for the scripts
```

Fitted codebooks for every string and entropy codec come with it, one set per
benchmark corpus, so encoding works immediately with no corpus on disk.

## 🚀 Quick start

### One molecule, by codec name

```bash
# list the names the commands below accept
osmiumchem codecs

# encode, decode, and check the molecule survived
osmiumchem roundtrip --codec smiles+arith-o4 "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
#  OK    57 bits  Cn1c(=O)c2c(ncn2C)n(C)c1=O

# or the two halves separately; both read the same shipped codebook
osmiumchem encode --codec smiles+arith-o4 "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
osmiumchem decode --codec smiles+arith-o4 101000100101011100101101110110111110001000100100101000000
```

Caffeine, recovered exactly, in 57 bits. Nothing was downloaded and nothing was
fitted: an order-4 arithmetic coder *is* its frequency table, and a table
fitted on each benchmark corpus ships inside the package. `--corpus` picks
which one, and `osmiumchem codecs --corpora` lists them.

Input may be SMILES, SELFIES, InChI or DeepSMILES (`--in`, default `auto`), the
molecule comes back in any of those (`--out`), and `-` reads from stdin.

The verdict has three values, because losing a molecule and keeping it in too
many bits are different results:

| verdict | meaning |
| --- | --- |
| `OK` | same molecule, inside the bit budget |
| `OVER` | same molecule, above the bit budget |
| `FAIL` | molecule lost |

### From Python

Three verbs, reading the same codebooks as the commands:

```python
import osmiumchem as oc

codec = oc.load_codec("smiles+arith-o4")
result = codec.round_trip("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
result.verdict, result.bits, result.decoded.smiles
# ('ok', 57, 'Cn1c(=O)c2c(ncn2C)n(C)c1=O')

# fit on your own series, keep it, and read it back anywhere
mine = oc.fit_codec("smiles+arith-o4", my_series)
oc.save_codec(mine, "series.oc.xz")
mine = oc.load_codec("series.oc.xz")
results = mine.round_trip_many(held_out)        # batched, one result each
```

Fitting on the chemistry you intend to encode pays off, because the fitted
table *is* the code. On 100 held-out PROTACs, a codec fitted on 2500 other
PROTACs spends a median of **146 bits** against **208** for the shipped `union`
codebook, with both recovering every molecule.

[docs/USAGE.md](docs/USAGE.md) covers both interfaces, including why a
successful round trip returns the *canonical* SMILES rather than the string you
typed.

### The benchmark

The harness is in `scripts/`, not in the package, because it reads corpora from
files and the package knows nothing about where corpora live. Name your own
with `--source FILE[:COLUMN]`, repeatable:

```bash
# what the suites contain
python scripts/benchmark.py --suite all --list-codecs

# the lossless ladder, on enough molecules to certify four nines
python scripts/benchmark.py --source ~/corpora/drug_like.parquet:SMILES \
    --suite lossless --evaluate 50000

# fingerprints and latent spaces, with the injectivity ceiling
python scripts/benchmark.py --source ~/corpora/drug_like.parquet \
    --suite lossy --evaluate 50000 --csv results/lossy.csv

# the hardest corpus for a fixed budget
python scripts/benchmark.py --source ~/corpora/protacs.csv:canonical_smiles \
    --suite fast --evaluate 0 --fit 3000
```

Several sources are aggregated and deduplicated on canonical SMILES, so a
compound reached from two files cannot sit in the fitting split and the
evaluation split at once. `scripts/run_all.sh` reproduces every table in
`docs/RESULTS.md` from corpora you name.

## 🔬 How fidelity is judged

Two molecules are the same when their **canonical isomeric SMILES** agree, so
atom ordering and aromatic spelling are free, while stereochemistry, formal
charge and isotopes all count. Tautomers are *not* unified: moving a proton
gives a different molecule, and normalising that away would excuse real
information loss.

Accuracy is reported with a **Wilson confidence interval**, and a codec is
marked as meeting the target only when the interval's *lower* bound clears it.
That matters more than it sounds. A run with zero failures reaches a 95% lower
bound of just 99.96% at ten thousand molecules, and 99.98% at twenty thousand:

> **About forty thousand clean round trips are needed to certify 99.99%.**
> Smaller runs can refute a codec but never confirm one.

## 💡 What the benchmark shows

### Entropy-coded SMILES wins outright

Order-4 arithmetic coding over canonical SMILES tokens is exactly reversible
and spends a median of **70 bits** per drug-like molecule, a fourteenth of the
budget. Measured over **200 000 molecules**: 100.0000% round trips, 95% lower
bound 99.9981%.

It holds up where it is hardest, which is what makes it the default rather than
the winner on easy molecules:

| corpus | accuracy | median bits | max bits | every payload ≤ 1000 bits |
| --- | ---: | ---: | ---: | :---: |
| 200 000 drug-like | 100.0000% | 70 | — | yes |
| 7 249 PROTACs | 100.0000% | 143 | **922** | yes |
| PROTACs, fitted on drug-like | 100.0000% | 228 | — | 99.8% |
| 200 000 from the 7.7M union | 100.0000% | 74 | **3457** | no — 4 molecules |

Order 3 is a close second (80 / 180 / 252 bits) and order 2 is cheaper to fit.
Below order 2 the PROTAC tail starts overflowing the budget.

### On the union corpus, the budget breaks before the code does

Over 200 000 molecules sampled from all 7.7M, order-4 arithmetic coding over
SMILES round-trips **every single molecule exactly**, at a median of 74 bits.
Four of them will not fit 1000 bits: lipidated and disulfide-bridged cyclic
peptides and oligonucleotides, needing up to 3457 bits.

Nothing about the coder can fix that, and nothing should. The largest entry in
the corpus is an oligonucleotide of **681 heavy atoms** whose SMILES runs to
1818 characters, and a thousand bits cannot describe it. The 1000-bit ceiling
is a statement about small-molecule chemistry, and on this corpus it holds for
99.998% of what is in ChEMBL and ExCAPE-DB put together.

### SELFIES qualifies too, once its valence rules are widened

On the same 200 000 molecules, `selfies-hypervalent+arith-o4` also certifies at
100.0000%. Its median is slightly larger at 73 bits, but its tail is markedly
**tighter**: worst case 348 bits against 574 for SMILES. The extra tokens buy a
more uniform payload size, which is worth having when the budget is a hard
ceiling.

Under the *default* valence rules SELFIES does not qualify at all, because its
encoder refuses hypervalent species outright.

### Where the saving comes from

The ladder attributes each step to one cause rather than hiding them all in a
black box:

| rung | median bits | what the step buys |
| --- | ---: | --- |
| `smiles+flat` | 329 | baseline, no modelling |
| `smiles+huffman` | 173 | symbol skew |
| `smiles+arith-o0` | 156 | fractional bits per symbol |
| `smiles+arith-o2` | 95 | context, 2 tokens |
| `smiles+arith-o4` | **70** | context, 4 tokens |

Context is the biggest single win, which figures: a `(` is almost always
followed by an atom, and a `[` by an element symbol.

### Reaching for a general-purpose compressor backfires

`zlib`, `bz2` and `lzma` all produce payloads *larger* than the uncompressed
string, because a multi-byte container header dwarfs a fifty-character message.

### A static model needs an escape hatch

Fitted on a few thousand molecules, a frequency model meets an unseen token
(`[SH]`, `[/O]`) in roughly one evaluation molecule in four hundred, capping
accuracy near 99.7%. Escaping unknown tokens to literal characters restores
exact reversibility for *any* input, and is what lifts these codecs over the
99.99% bar.

### Fingerprints are not invertible, and a lookup table hides it

ECFP is a hash, so decoding one means searching a library. A library decoder is
given the answer key, so its score measures *identification*, not
reconstruction — and the `library-index` control reaches the same score with
`ceil(log2(N))` ≈ 16 bits instead of 2048. Those 2048 bits were never what
identified the molecule.

Judged on reconstruction instead, with the decoder holding only the fitting
split, every fingerprint scores **0.0000%** over 50 000 molecules.

### For fingerprints, more bits buy nothing

Their injectivity ceiling is identical at 512, 1024 and 2048 bits (99.8920%),
so the information is lost at feature extraction rather than in the folding.
Stereochemistry (+0.024 points) and counts (+0.052) help a little, and the best
configuration measured reaches 99.9580%. **None reaches 99.99%**, so for this
family the target is excluded by the representation itself, independent of
decoder or budget.

### For lossy codes, report the ceiling, not just the decoder

If two molecules share a payload, no decoder can separate them, so the fraction
of molecules with a *unique* payload bounds the accuracy of **any** decoder,
including one not yet trained. That is the rigorous answer to "could a custom
decoder fix this?"

### Learned latent spaces do not fit the budget at native precision

CDDD is 512 float32 coordinates, **16 384 bits**, sixteen times over.
GROVER-base is 3200, **102 400 bits**, a hundred times over. Quantisation is
the entry fee, not an optimisation.

CDDD ships a trained beam-search decoder, so it can be measured end to end at
each precision:

| bits/dim | payload | accuracy | fits budget |
| --- | ---: | ---: | :---: |
| float32 | 16 384 | 58.33% | no |
| 4 | 2 048 | 58.13% | no |
| 2 | 1 024 | 54.20% | no |
| **1** | **512** | **18.27%** | **yes** |

Two things stand out. Compressing the latent sixteen-fold, from 16 384 bits to
1 024, costs almost nothing: the precision was never load-bearing. But the step
to the only configuration that *fits* collapses it to 18%. And the whole column
plateaus near 58%, so CDDD's decoder never approaches 99.99% at any budget.

Meanwhile the quantised 512-bit code is **100% injective** on the evaluation
set, so the information survives the budget even though the decoder cannot use
it. Could a custom decoder do better? In principle yes, by a wide margin: the
ceiling says so and the shipped decoder is nowhere near it. Whether a decoder
that *generalises* can be learned is a question this benchmark poses rather
than settles.

### Lossless does not imply frugal

A hand-written canonical graph serialisation (`graph-bits`) reproduces every
molecule exactly, stereochemistry included, but spends a median of ~610 bits
and overflows 1000 on the tail. RDKit's native binary form is lossless at ~3000
bits, three times the budget.

## 🧪 Codecs

| family | codecs |
| --- | --- |
| `string` | canonical SMILES, Kekulé SMILES, SELFIES, DeepSMILES, InChI |
| `entropy` | flat, Huffman, arithmetic order 0–4 (over SMILES and SELFIES), zlib/bz2/lzma |
| `structural` | `graph-bits`, RDKit binary, RDKit binary + LZMA |
| `fingerprint` | ECFP4/ECFP6 at 512–2048 bits, counted ECFP, MACCS, RDKit, atom pairs |
| `latent` | CDDD (quantised, and via its own decoder), GROVER, descriptors |
| `control` | `library-index` |

[docs/CODECS.md](docs/CODECS.md) covers each family in detail.

## 📊 Corpora

**No corpus ships with this project, and no path to one appears in it.** The
results below were measured on corpora that are not ours to redistribute, so
the scripts take files you name and the package carries only the fitted tables
those files produced. Each shipped codebook records which corpus it came from
as a label — `union`, `multitask`, `warheads`, `cyp450`, `protacs`,
`zinc250k`, `hiv` — and that label is all that is left of them.

The corpora behind the published runs are ChEMBL-derived activity tables
(PROTACs, human CYP450, covalent warheads), an ExCAPE-DB reproduction, three
splits of a multitask collection, ZINC250k and MoleculeNet HIV.

For a fixed bit budget the PROTAC corpus is the real test, with a median SMILES
of 107 characters against a drug-like median of 44.

### The `union` corpus

The broadest is `union`: **7,676,241 unique molecules** aggregated from every
other source. Build your own equivalent once, then pass the result as a single
source:

```bash
python scripts/corpora.py --out ~/corpora/union.csv --workers 16 \
    --source ~/corpora/protacs.csv:canonical_smiles \
    --source ~/corpora/excape.parquet:SMILES \
    --source ~/corpora/train_set.parquet:SMILES

python scripts/benchmark.py --source ~/corpora/union.csv --label union \
    --suite lossless --evaluate 50000
```

The sources overlap heavily and each repeats a compound once per measurement,
so aggregation deduplicates on canonical SMILES rather than concatenating. One
row per molecule is what makes the split trustworthy: the same compound reached
from four sources must not be able to sit in the fitting split and the
evaluation split at once.

That aggregation took seven minutes, and a sibling `.manifest.json` records
what each source contributed:

| source | rows | distinct values read | molecules added |
| --- | ---: | ---: | ---: |
| PROTAC ChEMBL activities | 4,059 | 4,059 | 1,160 |
| human CYP450 activities | 98,846 | 98,502 | 33,585 |
| covalent warhead bioactivities | 481,031 | 481,030 | 195,458 |
| ExCAPE-DB reproduction | 388,061,449 | 126,155,633 | 2,213,907 |
| multitask finetune split | 395 | 395 | 218 |
| multitask test split | 395 | 395 | 195 |
| multitask train split | 6,033,577 | 6,033,577 | 5,231,718 |

Nothing was unparseable, and the order matters: each row's contribution is what
was *new* at that point, so the 2.2M from ExCAPE-DB and the 5.2M from the
multitask split are disjoint sets of compounds. Reading a 388-million-row
activity table is affordable because each Parquet batch is narrowed to its
distinct values in Arrow before any Python string is built.

## 📚 Documentation

- [docs/USAGE.md](docs/USAGE.md) — encoding and decoding a single molecule by
  codec name, from the command line or the Python API: codebooks, input and
  output representations, payload text forms, fitting on your own corpus.
- [docs/METHOD.md](docs/METHOD.md) — what counts as the same molecule, how bits
  are counted, why the evaluation split is large, the injectivity ceiling.
- [docs/CODECS.md](docs/CODECS.md) — every codec, what it isolates, and the
  stereochemistry pitfalls.
- [docs/RESULTS.md](docs/RESULTS.md) — measured tables, generated from run CSVs
  by `scripts/make_results.py`.
- `scripts/run_all.sh` — reproduces every table from corpora you name.
- `scripts/make_codebooks.py` — refits the codebooks that ship in the package.
- `scripts/corpora.py` — aggregates several corpora into one deduplicated file.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

[MIT](LICENSE)
