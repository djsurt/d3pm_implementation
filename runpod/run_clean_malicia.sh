#!/usr/bin/env bash
# Decontaminated re-run of the run_01 BASELINE used by the §2.5 downstream-utility
# headline. Reproduces run_01's exact config (default preset: d_model=128, layers=4,
# T=500, max_len=256, 100 epochs, batch 128, 500 synth/family) and changes ONE thing:
# --holdout-test-frac 0.3 holds out the SAME 30% test files (seed 1234) the
# downstream classifier scores on, so the generator never trains on a test file.
# That isolates "decontamination" as the single variable vs the published numbers.
#
# Run inside tmux:  bash runpod/run_clean_malicia.sh 2>&1 | tee run_clean.log
# Afterwards (locally, fast, sklearn venv):
#   cd comparison && PYTHONPATH=lib .venv/bin/python lib/seq_classifier.py \
#       --d3pm-run ../runs/run_03_clean --out results/seq_classifier_clean.json
set -euo pipefail
cd "$(dirname "$0")/.."        # repo root
PY=${PY:-.venv/bin/python}

FAMILIES=("winwebsec" "zbot" "zeroaccess")
EPOCHS=100
BATCH=128                      # match run_01 exactly; decontamination is the ONLY changed variable
MAXLEN=256
NGEN=500
OUT=runs/run_03_clean          # keep run_01 artifacts intact; write clean ones here
# MUST match seq_classifier.py defaults so the held-out files line up exactly.
HOLDOUT_FRAC=0.3
HOLDOUT_SEED=1234

START=$(date +%s)
for fam in "${FAMILIES[@]}"; do
  echo "============================================================"
  echo "  TRAIN (clean) $fam   baseline max_len=$MAXLEN epochs=$EPOCHS  holdout=$HOLDOUT_FRAC"
  echo "============================================================"
  # No --max-files: the holdout split must index the full corpus, exactly like the
  # classifier's full-corpus test split.
  $PY -m d3pm.train --family "$fam" --epochs "$EPOCHS" --batch "$BATCH" \
      --max-len "$MAXLEN" --out-dir "$OUT/checkpoints" \
      --holdout-test-frac "$HOLDOUT_FRAC" --holdout-seed "$HOLDOUT_SEED"

  echo "------ GENERATE $fam  (n=$NGEN) ------"
  $PY -m d3pm.generate --family "$fam" --n "$NGEN" --seq-len "$MAXLEN" \
      --ckpt-dir "$OUT/checkpoints" --out-dir "$OUT/synthetic"

  echo "------ EVAL $fam (sequence metrics) ------"
  $PY -m d3pm.seq_eval --family "$fam" \
      --synthetic "$OUT/synthetic" --out-dir "$OUT/eval_results"

  echo "------ AUDIT $fam (memorization + transition KL) ------"
  $PY -m d3pm.seq_audit --family "$fam" \
      --synthetic "$OUT/synthetic" --out-dir "$OUT/eval_results"
done

echo
echo "Clean Malicia re-run done in $(( ($(date +%s) - START) / 60 )) min → $OUT/"
echo ">>> Re-run the downstream sweep against $OUT/ to get the decontaminated lift."
