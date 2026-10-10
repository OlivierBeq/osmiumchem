# Method

How the benchmark decides what a round trip costs and whether it succeeded.

The corpora themselves are not here. They are not ours to redistribute, so
`scripts/benchmark.py` reads whichever files you name with `--source`, and the
package carries only the fitted tables they produced. Every figure below was
measured on the corpora named with it.

- [What counts as the same molecule](#what-counts-as-the-same-molecule)
- [Why the evaluation split is large](#why-the-evaluation-split-is-large)
- [How bits are counted](#how-bits-are-counted)
- [Shared state is fitted out of sample](#shared-state-is-fitted-out-of-sample)
- [The injectivity ceiling](#the-injectivity-ceiling)
- [Closed-world versus open-world decoding](#closed-world-versus-open-world-decoding)
- [Failure classification](#failure-classification)
- [Exactness of the entropy coders](#exactness-of-the-entropy-coders)

## What counts as the same molecule

Fidelity is measured against **canonical isomeric SMILES**, as produced by
RDKit after sanitisation. Two molecules are identical when those strings agree.

That choice fixes exactly what a codec is allowed to lose:

| property | must survive? | reason |
| --- | --- | --- |
| atom ordering | no | canonicalisation removes it, so codecs may reorder freely |
| aromatic vs Kekulé spelling | no | both canonicalise to the same string |
| component ordering in a salt | no | canonicalisation sorts components |
| **tetrahedral stereochemistry** | yes | enantiomers are different compounds |
| **double-bond (E/Z) stereochemistry** | yes | geometric isomers are different compounds |
| **formal charge** | yes | an ion is not its neutral form |
| **isotopic labelling** | yes | a tracer is not its unlabelled parent |
| **radical electrons** | yes | a radical is not its closed-shell form |
| hydrogen *representation* | no | compared after sanitisation moves H into counts |
| hydrogen *count* | yes | changing it changes the molecule |

Tautomers are **not** unified. Moving a proton produces a different canonical
SMILES and is scored as a failure. Normalising tautomers would excuse genuine
information loss, and it is exactly the loss that InChI's mobile-hydrogen
convention introduces.

## Why the evaluation split is large

A 99.99% target means fewer than one failure in ten thousand. Accuracy is
therefore reported with a **Wilson score interval**, and a codec is credited
only when the interval's *lower* bound clears the target.

For a run with **zero observed failures**:

| molecules | 95% lower bound | certifies 99.99%? |
| --- | --- | --- |
| 1 000 | 99.6173% | no |
| 10 000 | 99.9616% | no |
| 20 000 | 99.9808% | no |
| 30 000 | 99.9872% | no |
| **40 000** | **99.9904%** | **yes** |
| 50 000 | 99.9923% | yes |
| 200 000 | 99.9981% | yes |

So a clean thousand-molecule run is not evidence of four nines; it is only the
absence of evidence against it. Runs smaller than ~40 000 can refute a codec
but never confirm one.

## How bits are counted

A payload's cost has to be unambiguous, because the budget is a hard ceiling.

**Bit sequences** are charged their exact length. Lengths are tracked
explicitly rather than inferred, since leading zeros are data: `0101` and `101`
are different payloads.

**Strings** have no intrinsic bit length, so they are charged a flat code over
the alphabet observed while fitting: `ceil(log2(|alphabet|))` bits per token,
about 6 bits for a ~40-symbol SMILES alphabet. Charging `len(s) * 8` would
overstate the cost by a third and flatter every compressor measured against it.
The UTF-8 figure is recorded separately.

**Float vectors** are charged `bits_per_dim * dims` after quantisation, and
their unquantised cost `32 * dims` is reported alongside as `native bits`. That
conversion is what makes a latent space comparable to a bitstring at all:

| encoder | dims | native (float32) | vs 1000-bit budget |
| --- | --- | --- | --- |
| CDDD | 512 | 16 384 bits | 16× over |
| GROVER-base | 3 200 | 102 400 bits | 102× over |

For these encoders, quantisation is the entry fee rather than an optimisation.

## Shared state is fitted out of sample

Anything a codec learns — a frequency table, Huffman lengths, a quantisation
range, a lookup library — is fitted on a split **disjoint** from the evaluation
set, and is treated as a shipped artefact rather than charged per molecule.
That mirrors how a trained decoder is deployed: weights ship once, the payload
travels per molecule. Fitting on the evaluation set would report training
error.

`--fit-source` fits on *different files*, to test whether a learned code
transfers across chemical space, for example fitting on drug-like molecules and
evaluating on PROTACs.

### Why the union corpus is deduplicated, not concatenated

The `union` corpus aggregates every other source: three ChEMBL-derived activity
tables, an ExCAPE-DB reproduction and the three multitask splits. They overlap
heavily, and each repeats a compound once per measurement, with 388 million
ExCAPE-DB rows contributing 2.2 million distinct compounds.

Concatenating them would break the guarantee above in the worst way: the same
molecule, reached from two sources, could be drawn into the fitting split *and*
into the evaluation split, and every learned codec would then be scored partly
on what it was fitted on.

Aggregation therefore deduplicates on **canonical SMILES**, the same identity
test used to judge a round trip, so one compound is one row no matter how many
sources hold it and in how many spellings. Deduplicating on the raw string
would not be enough, because the sources disagree about aromatic notation and
atom order, so the same molecule appears written several ways.

The redundancy is large and measured: the multitask split contributed 5,231,718
new molecules from 6,033,577 SMILES, so some 800 thousand of its entries were
either already present from an earlier source or a second spelling of a
molecule it already held.

A reservoir sample over the aggregate is uniform by molecule rather than by
measurement, which also removes a bias concatenation would have introduced. A
compound assayed a thousand times would otherwise have been a thousand times
likelier to be benchmarked than one assayed once.

## The injectivity ceiling

A poor score from a lossy code invites blaming a weak decoder. That excuse is
removable.

If two distinct molecules produce the same payload, no decoder can tell them
apart, however large or well fitted, because the distinguishing information is
not present. The fraction of evaluation molecules whose payload is **unique**
is therefore a hard upper bound on the accuracy of *any* decoder, and it is
computable without training anything.

This is reported as `ceiling`. It answers "could a custom decoder fitted to
this latent space reach 100%?" in the *negative* direction: if the ceiling is
below 99.99%, no decoder can get there, and that settles it. Every fingerprint
in this benchmark falls into that category.

### Injectivity is necessary, not sufficient

The converse does **not** follow, and the benchmark includes a control that
makes the point unmistakable. Sixteen physicochemical descriptors — molecular
weight, logP, TPSA, ring counts — quantised to sixteen bits each come to 256
bits and are **100% injective** over a 3000-molecule evaluation set. Nobody can
reconstruct a molecule from its molecular weight and logP. A unique code is not
a decodable one.

Three limits therefore apply whenever a ceiling of 100% is read as encouraging:

1. **Uniqueness is measured on this evaluation set.** Injectivity over three
   thousand molecules says nothing about three million, where collisions
   become likelier.
2. **The witness is a lookup table.** "A decoder exists" is satisfied by
   memorising the corpus, which is not a decoder in any useful sense and
   generalises to nothing.
3. **Learnability is a separate question.** Whether a decoder that
   *generalises* can be fitted to a given latent space is not something an
   injectivity count can answer, and this benchmark does not claim to.

So a ceiling below the target is a proof of impossibility; a ceiling at the
target is an invitation, not a result.

## Closed-world versus open-world decoding

Codes like fingerprints have no inverse, so they are decoded by lookup. The
world matters, and the benchmark reports both:

| `--world` | the library holds | what the score measures |
| --- | --- | --- |
| `closed` | the evaluation molecules | **identification** |
| `open` | only the fitting split | **reconstruction** |

Closed world hands the decoder the answer key, so its only failure mode is a
true collision and its accuracy equals the injectivity ceiling. Open world
makes it produce a molecule it has never seen, which is what an application
needs.

The `library-index` control keeps closed-world numbers in proportion. Naming an
entry in a library of `N` molecules costs `ceil(log2(N))` bits, about 18 for a
quarter of a million. When a 2048-bit fingerprint and an 18-bit index score the
same, the fingerprint's 2048 bits were not what identified the molecule.

## Failure classification

Failures are grouped by cause, because a decode that returns nothing and one
that flips a stereocentre are different problems:

| class | meaning |
| --- | --- |
| `no-encoding` | the encoder refused the molecule |
| `no-output` | the decoder returned nothing |
| `unparseable` | a string came back, but it is not a molecule |
| `stereo-only` | the skeleton is right; only stereochemistry differs |
| `formula-kept` | same atoms, different connectivity — a tautomer or rearrangement |
| `formula-changed` | atoms were gained or lost |

## Exactness of the entropy coders

The arithmetic coder keeps its active interval as an exact rational over Python
big integers, subdividing it as `den *= T; lo = lo*T + size*c; size *= f`. No
rounding occurs, so there is no renormalisation and no carry propagation, the
two places fixed-width arithmetic coders usually leak. The payload is the
shortest binary fraction inside the final interval.

The cost is big-integer arithmetic that grows with message length, which is
quadratic overall. For strings of tens to a few hundred tokens that is fast,
and it buys reversibility that is provable rather than hoped for. The
arithmetic being integer-only and deterministic, the emitted bits are
reproducible across platforms.

Context models are **static**: fitted once, then frozen, so the decoder sees
exactly the distribution the encoder saw. Unknown tokens are **escaped** to
literal characters, which is what makes these codecs lossless for arbitrary
input rather than only for the fitted alphabet.
