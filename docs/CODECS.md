# Codec reference

Codecs are grouped into families. Within a family they form a *ladder*, so the
difference between two adjacent rungs isolates one cause.

| family | what it is | meets both targets? |
| --- | --- | --- |
| [`string`](#string--textual-representations) | textual representations | not on its own |
| [`entropy`](#entropy--lossless-bit-level-codecs-over-a-string) | bit-level coders over a string | **yes** |
| [`structural`](#structural--the-graph-written-directly-into-bits) | the graph written into bits | lossless, but over budget |
| [`fingerprint`](#fingerprint--hashed-structural-keys) | hashed structural keys | no, by construction |
| [`latent`](#latent--learned-continuous-embeddings) | learned embeddings | no |
| [`control`](#control) | yardsticks, not candidates | n/a |

## `string` — textual representations

| codec | lossless by design | note |
| --- | --- | --- |
| `smiles` | yes | RDKit canonical isomeric SMILES; the benchmark's reference form |
| `smiles-kekule` | yes | aromaticity written as alternating bonds |
| `selfies` | yes | SELFIES v2, default valence rules |
| `selfies-hypervalent` | yes | SELFIES v2, widened valence rules |
| `deepsmiles` | yes | ring closures and branches rewritten |
| `inchi` | **no** | a canonical *identifier*, never designed to be decoded |

Canonical SMILES is the control: it defines the identity test, so anything less
than a perfect score would indicate a harness bug rather than chemistry.

### SELFIES is robust at generating, not at encoding

SELFIES guarantees that every *string* decodes to a valid molecule. That is a
different property from encoding every *molecule*, and the gap is measurable.
Under the default valence rules the encoder **raises** on hypervalent species
such as `Nc1ccc([IH]c2ccc(N)cc2)cc1`, where `[IH]` carries two bonds against a
permitted maximum of zero.

On a drug-like corpus that costs roughly one molecule in twenty thousand. Small
— but on its own enough to hold SELFIES at 99.995%, and so below a 99.99%
*certified* target. It is a configuration choice rather than a limitation: the
`hypervalent` preset encodes those molecules losslessly and changes nothing
else, which is why both variants are reported.

Note that `selfies` keeps its valence rules in module-global state, so the
preset is installed immediately before every call and the conversion cache is
keyed by preset. Otherwise a value computed under one rule set would be served
under another.

### InChI shows that canonical does not imply invertible

Its mobile-hydrogen convention normalises amide and other tautomers, so
`C[C@H](N)C(=O)N[C@H](Cc1ccccc1)C(=O)O` comes back with the amide proton
moved. Those failures classify as `formula-kept`: the atoms are all there, the
connectivity is not.

## `entropy` — lossless bit-level codecs over a string

This is the family that meets both requirements at once. Each rung attributes
the saving to a single cause.

Every codec here is a frozen table, so the table is what has to be shared
between an encoder and a decoder. One per codec per benchmark corpus ships
inside the package; `osmiumchem codecs --corpora` lists the labels and
`docs/USAGE.md` covers fitting your own.

Measured on 20 000 drug-like molecules with a 50 000-molecule fitting split
(median payload, SMILES variants):

| codec | gain over the rung above | median bits |
| --- | --- | ---: |
| `smiles+flat` | — (baseline, no modelling) | 329 |
| `smiles+huffman` | symbol **skew** | 173 |
| `smiles+arith-o0` | **fractional** bits per symbol | 156 |
| `smiles+arith-o2` | **context**, 2 tokens | 95 |
| `smiles+arith-o3` | more context | **80** |

Context is the single biggest win. It roughly halves the payload again after
symbol skew has already halved it, which is unsurprising given how constrained
SMILES syntax is: a `(` is almost always followed by an atom, a `[` by an
element symbol, and a ring-opening digit by a bond or atom.

The ladder is available over both `smiles` and `selfies`, which land within a
few bits of each other once entropy-coded. SELFIES' extra tokens are highly
predictable, so context coding recovers almost all of what the longer strings
cost.

`smiles+zlib`, `smiles+bz2` and `smiles+lzma` are included as the obvious first
instinct, and they lose badly. Each writes a multi-byte container header, which
on a fifty-character message costs more than the message, so they produce
payloads *larger* than the uncompressed string.

### Escaping

A static model fitted on a finite corpus eventually meets a token that corpus
lacked (`[SH]`, `[/O]`, a rare element). Measured on a 4000-molecule fit split
that happens to about one evaluation molecule in four hundred: a hard cap near
99.7%, far short of the target, and more fitting data shrinks but never removes
the tail.

Every entropy codec therefore carries an escape hatch. Alongside the observed
tokens the alphabet always contains all printable ASCII, and an unknown token
is emitted as `ESC`, its own characters, `ESC`:

```
[SH]  ->  ESC  '['  'S'  'H'  ']'  ESC
```

Escapes are rare, so the average cost is negligible, and the codec becomes
lossless for *any* ASCII input. This is what lifts the family over 99.99%.

## `structural` — the graph written directly into bits

Measured on the same 20 000 drug-like molecules, with the PROTAC corpus shown
alongside because that is where a fixed budget bites:

| codec | lossless | median bits (drug-like) | fits 1000 bits | median bits (PROTACs) | fits 1000 bits |
| --- | --- | ---: | ---: | ---: | ---: |
| `graph-bits` | yes | 620 | 97.6% | 1521 | 22.2% |
| `rdkit-binary-lzma` | yes | 2016 | 0.0% | 3392 | 0.0% |
| `rdkit-binary` | yes | 3088 | 0.2% | 6544 | 0.0% |

`graph-bits` is a hand-written canonical serialisation: atom count, then per
atom an element code, hydrogen count, charge, isotope and radical flags, then
the adjacency with bond orders and stereo descriptors. It is exactly
reversible, stereochemistry included, and it still loses. It spends roughly
eight times what entropy-coded SMILES does, overflows 1000 bits on 2.4% of
drug-like molecules, and on PROTACs fits the budget only 22% of the time.

**Lossless does not imply frugal**, and writing bits by hand is no substitute
for modelling what the bits are likely to say.

### Why stereochemistry is the hard part

RDKit stores tetrahedral chirality as a tag whose meaning is relative to *the
order an atom's bonds happen to be stored in*. Permute that order and the same
tag denotes the opposite enantiomer, so copying tags across unchanged flips
about half of all stereocentres. The fix is two-sided:

1. The decoder adds bonds in a fixed order — for each atom `i` ascending, every
   neighbour `j < i` ascending — which makes every atom's final bond list sorted
   by neighbour index.
2. The encoder computes the parity of the permutation from the actual bond
   order to that sorted order, and flips the stored tag when it is odd.

Double-bond stereochemistry needs a second fix. The E/Z descriptor alone is not
enough for the SMILES writer, which emits `/` and `\` from the *directions* of
the neighbouring single bonds. `Chem.SetDoubleBondNeighborDirections` derives
them after the descriptors are restored. Without it, every double-bond
stereocentre comes back unspecified.

## `fingerprint` — hashed structural keys

| codec | bits | fits 1000-bit budget |
| --- | --- | --- |
| `ecfp4-512` | 512 | yes |
| `ecfp4-1024` | 1024 | no |
| `ecfp4-2048` | 2048 | no |
| `ecfp6-1024` | 1024 | no |
| `ecfp4-512-count` | 2048 | no |
| `ecfp4-512-chiral` | 512 | yes |
| `ecfp4-2048-chiral` | 2048 | no |
| `maccs-167` | 167 | yes |
| `rdkit-512` | 512 | yes |
| `atompair-512` | 512 | yes |
| `atompair-512-chiral` | 512 | yes |

These are hashes, and folding an unbounded space of substructures into a fixed
vector is many-to-one on purpose. They have no inverse, so they are decoded by
lookup, and the `closed`/`open` distinction decides whether the resulting score
means anything. Read the `ceiling` column, not the accuracy.

It is also why no fingerprint or latent codec ships a fitted codebook: its
fitted state is the library, which is a corpus, and the corpora stay out of the
package. They are fitted at use, from `--fit-on` or `fit_codec`.

The sizes bracket the budget. ECFP4 at 2048 bits is the field's default, so
showing where it falls against a 1000-bit ceiling is part of the point.

### No fingerprint reaches a 99.99% ceiling, and extra bits do not help

Measured over 50 000 drug-like molecules, the ceiling can be ablated to find
out *where* the information goes:

| variation | ceiling | change |
| --- | ---: | --- |
| `ecfp4-512` | 99.8920% | baseline |
| `ecfp4-1024` | 99.8920% | **none** |
| `ecfp4-2048` | 99.8920% | **none** |
| `ecfp4-512-chiral` | 99.9160% | +0.024 |
| `ecfp4-2048-chiral` | 99.9160% | +0.024 |
| `ecfp4-512-count` | 99.9440% | +0.052 |
| `atompair-512-chiral` | 99.9580% | best measured |

The headline is the first three rows: **quadrupling the fingerprint length
changes the ceiling by exactly nothing.** On drug-like molecules the bit vector
is nowhere near saturated, so folding is not what loses the information. The
loss happens earlier, at feature extraction, in *which* substructures the
fingerprint chooses to record at all. Spending more bits on a lossy feature set
buys nothing, which is why `ecfp4-2048` costs four times `ecfp4-512` for an
identical ceiling and blows the 1000-bit budget for the privilege.

Adding *kinds* of information does help, slightly. Stereochemistry is excluded
by RDKit's `includeChirality=False` default, which makes a molecule and its
epimer hash byte-identically:

```
O=C(NC1CC(C(=O)O)C12CCOCC2)OCc1ccccc1      <- different molecules
O=C(N[C@H]1CC(C(=O)O)C12CCOCC2)OCc1ccccc1  <- identical ECFP4-2048
```

Setting the flag separates such pairs at **exactly the same bit cost**, and
substructure counts help about twice as much again. But the gains are
hundredths of a percent against a target that needs 99.99%, so **no
configuration measured here reaches it**: the best, `atompair-512-chiral`,
stops at 99.9580%. For fingerprints the 99.99% target is not a decoder problem
and not a budget problem; it is ruled out by what the representation records.

Saturation *does* bite on large molecules, where the picture inverts. On
PROTACs `ecfp4-512` falls to a 74.2% ceiling and counts lift it to 95.8%,
because there many substructures really are competing for the same vector. So
length is moot when the vector is sparse and decisive when it is full, and
neither regime gets near the target.

## `latent` — learned continuous embeddings

| codec | dims | native (float32) | quantised |
| --- | --- | --- | --- |
| `cddd-q1xall` | 512 | 16 384 bits | 512 bits |
| `cddd-q2x500` | 500 of 512 | 16 384 bits | 1000 bits |
| `cddd-q8x125` | 125 of 512 | 16 384 bits | 1000 bits |
| `cddd-native-q{1,2,4}` | 512 | 16 384 bits | 512 / 1024 / 2048 bits |
| `cddd-native-f32` | 512 | 16 384 bits | — |
| `grover-q1x1000` | 1000 of 3200 | 102 400 bits | 1000 bits |
| `grover-q2x500` | 500 of 3200 | 102 400 bits | 1000 bits |
| `descriptors-q{16,32}xall` | 16 | 512 bits | 256 / 512 bits |

Neither pretrained encoder fits the budget at native precision, by a wide
margin, so quantisation is the entry fee rather than an optimisation.

`cddd-native-*` is the one codec here with a genuinely **learned decoder**.
CDDD ships a pretrained beam-search decoder, so the latent is quantised,
de-quantised to bin centres, and handed to the real decoder, which is the
end-to-end test of a neural binary latent space. All 512 dimensions are kept
regardless of budget because the decoder's input layer requires them, so the
budget is spent entirely on precision per coordinate.

GROVER has no decoder by design, being a representation learner for property
prediction. Fitting one is exactly the task the injectivity ceiling speaks to,
so GROVER is reported through that bound, which states the best any future
decoder could do.

`descriptors-*` is weak on purpose and exists as a cautionary control: a
handful of physicochemical properties cannot reconstruct a molecule however
finely quantised, yet scores 100% under closed-world lookup.

## `control`

`library-index` encodes a molecule as its index in a known library, costing
`ceil(log2(N))` bits, about 18 for a quarter of a million molecules. It is the
yardstick for every lookup-decoded score: when a 2048-bit fingerprint and an
18-bit index score the same, the fingerprint's bits were not what identified
the molecule.
