# Contributing to this repository

## Getting started
- Before contributing, make sure you have a working developping environment set up.
```bash
    pip install tox
```
Few *tox* environments are defined for easier linting and testing.

We enforce strict coding rules. :
 - To make sure you comply with coding rules use the following command:
```bash
    tox -e ruff
```
 - Pyroma checks if the installation information is sufficient
 ```bash
    tox -e pyroma
```

For the entire workflow of linting and testing
```
    tox
```

## Running the benchmark

No corpus is in this repository and no path to one is either, so every run
takes files you name with `--source FILE[:COLUMN]`, repeatable:

```bash
pip install -e ".[all]"

# one-off: aggregate your sources into one file, deduplicated on canonical
# SMILES. A single pass over a few hundred million activity rows takes about
# seven minutes and writes one row per molecule.
python scripts/corpora.py --out ~/corpora/union.csv --workers 16 \
    --source ~/corpora/protacs.csv:canonical_smiles \
    --source ~/corpora/excape.parquet:SMILES

# every table in docs/RESULTS.md, for the corpora you set
MULTITASK=~/corpora/train_set.parquet:SMILES \
PROTACS=~/corpora/protacs.csv:canonical_smiles \
UNION=~/corpora/union.csv \
    scripts/run_all.sh results
python scripts/make_results.py results docs/RESULTS.md
```

`run_all.sh` skips any run whose corpus is unset, so a partial set of files
still produces the tables it can.

## Refitting the shipped codebooks

An entropy codec *is* its frequency table, so the package ships one per codec
per corpus and an installed copy needs no data. They are regenerated per
corpus, which `tests/test_codebooks.py` checks the consistency of:

```bash
python scripts/make_codebooks.py --label protacs \
    --source ~/corpora/protacs.csv:canonical_smiles
```

Bump `FORMAT` in `src/osmiumchem/codebook.py` whenever a change makes an older
codebook decode differently, and regenerate. The package version is recorded
in each codebook but deliberately not enforced, so an ordinary release does not
invalidate the set.

Only `pytest`, `ruff` and `pyroma` are needed to develop; the neural encoders
and Parquet corpora are optional extras, and the benchmark reports a missing
one as a per-codec error rather than failing the run.

## Adding a codec

Subclass `BaseCodec` (or `StringCodec` for a textual representation) and
implement `fit`, `encode`, `decode` and `payload_bits`, then add it to a suite
in `src/osmiumchem/codecs/__init__.py`.

Two rules matter more than anything else here:

1. **Never substitute.** If a molecule cannot be encoded, return `None`. A
   silent substitution turns into a reported success for the wrong molecule,
   which is the one failure mode the benchmark must never have.
2. **Charge honestly.** `payload_bits` must return what storing the payload
   really costs. For a float vector that means `bits_per_dim * dims`, with the
   native `32 * dims` reported alongside.

New codecs should be added to `tests/test_codecs.py`; anything claiming to be
lossless is expected to round trip every molecule in `tests/hard_cases.py`,
which covers
both kinds of stereochemistry, charges, isotopes, radicals, salts and
two-digit ring closures.
