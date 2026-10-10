#!/usr/bin/env bash
# Reproduce every benchmark table in docs/RESULTS.md.
#
# The corpora are not in this repository and no path to them is either, so
# point the variables below at your own copies. Every one is optional: a run
# whose corpus is unset is skipped.
#
#   MULTITASK=~/corpora/train_set.parquet:SMILES \
#   PROTACS=~/corpora/protacs.csv:canonical_smiles \
#   scripts/run_all.sh results
#
# The evaluation splits are deliberately large: certifying 99.99% at 95%
# confidence needs roughly 40000 clean round trips, so smaller runs can only
# refute a codec, never confirm one. The neural suite is the slow one, since
# CDDD's beam-search decoder runs per molecule on CPU.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-results}"
mkdir -p "$OUT"

# Each is a --source argument: a path, optionally followed by :column.
MULTITASK="${MULTITASK:-}"
PROTACS="${PROTACS:-}"
UNION="${UNION:-}"

run() {
  local name="$1"; shift
  echo "=== $name"
  python "$HERE/benchmark.py" "$@" --csv "$OUT/$name.csv" --markdown "$OUT/$name.md" | tee "$OUT/$name.txt"
}

have() {
  if [ -z "$2" ]; then
    echo "=== skipping $1: set \$$3 to a corpus file"
    return 1
  fi
}

if have multitask "$MULTITASK" MULTITASK; then
  # The headline: the configurations that actually qualify, scored with the
  # budget as a hard requirement. 200000 molecules because ~40000 clean round
  # trips are needed before a 99.99% claim is supportable at all.
  run certify_o4        --source "$MULTITASK" --label multitask \
                        --suite lossless --evaluate 200000 --fit 100000 \
                        --strict-budget --workers 8 \
                        --only "smiles+arith-o2,smiles+arith-o4,selfies-hypervalent+arith-o4"

  # The full non-neural sweep, closed world so the fingerprint ceilings show.
  run fast_multitask_closed --source "$MULTITASK" --label multitask \
                        --suite fast --evaluate 50000 --fit 50000 \
                        --world closed --workers 8

  # Fingerprints judged as reconstruction rather than identification.
  run lossy_open        --source "$MULTITASK" --label multitask \
                        --suite lossy --evaluate 50000 --fit 50000 \
                        --world open --workers 8

  # Learned latent spaces. Smaller split: these are bounded well below the
  # target, so a few thousand molecules already settle the question.
  run neural_multitask  --source "$MULTITASK" --label multitask \
                        --suite neural --evaluate 3000 --fit 20000 --workers 8
fi

if have protacs "$PROTACS" PROTACS; then
  # Beyond-rule-of-5 molecules, where a fixed bit budget actually bites.
  run fast_protacs      --source "$PROTACS" --label protacs \
                        --suite fast --evaluate 0 --fit 3000 \
                        --world closed --workers 8

  # The same corpus with the budget enforced, which is the question as posed.
  run strict_protacs    --source "$PROTACS" --label protacs \
                        --suite lossless --evaluate 0 --fit 3000 \
                        --strict-budget --workers 8

  # Does a code fitted on one region of chemical space transfer to another?
  if [ -n "$MULTITASK" ]; then
    run transfer_protacs --source "$PROTACS" --label protacs \
                        --suite lossless --evaluate 0 \
                        --fit 100000 --fit-source "$MULTITASK" --workers 8
  fi
fi

# The union corpus: every source aggregated and deduplicated to one row per
# molecule. Build it once, which for an activity table of a few hundred million
# rows is a long pass, and afterwards it reads back as an ordinary source:
#
#   python scripts/corpora.py --out ~/corpora/union.csv --workers 16 \
#       --source ~/corpora/a.csv:smiles --source ~/corpora/b.parquet:SMILES
if have union "$UNION" UNION; then
  run union_certify     --source "$UNION" --label union \
                        --suite lossless --evaluate 200000 --fit 100000 \
                        --strict-budget --workers 8 \
                        --only "smiles+arith-o2,smiles+arith-o4,selfies-hypervalent+arith-o4"

  run union_fast        --source "$UNION" --label union \
                        --suite fast --evaluate 50000 --fit 50000 \
                        --world closed --workers 8

  run union_lossy_open  --source "$UNION" --label union \
                        --suite lossy --evaluate 50000 --fit 50000 \
                        --world open --workers 8

  # Both worlds for the latents: open is the honest reading of the code, closed
  # is the identification ceiling the earlier drug-like table reports.
  run union_neural      --source "$UNION" --label union \
                        --suite neural --evaluate 3000 --fit 20000 \
                        --world open --workers 8

  run union_neural_closed --source "$UNION" --label union \
                        --suite neural --evaluate 3000 --fit 20000 \
                        --world closed --workers 8
fi

echo "all runs written to $OUT/"
