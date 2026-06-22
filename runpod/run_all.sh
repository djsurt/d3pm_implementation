#!/usr/bin/env bash
# Train -> generate -> evaluate the top-3 Malicia families on the GPU,
# then bundle the outputs for download.
# Run inside tmux:  bash runpod/run_all.sh 2>&1 | tee run_all.log
set -euo pipefail
cd "$(dirname "$0")/.."        # repo root
PY=.venv/bin/python

# --- knobs (run_02 scale-up; RTX 4090 has plenty of headroom) ---------------
# Scale-up vs run_01: --scaleup preset (d_model=256, layers=6, dim_ff=1024,
# max_len=512, T=1000 ≈ 8M params) + EMA + held-out val loss w/ early stop.
FAMILIES=("winwebsec" "zbot" "zeroaccess")   # the top-3 by file count
EPOCHS=150                                     # cap; early-stop usually ends sooner
BATCH=64                                       # smaller: bigger model + 2× context
MAXLEN=512
NGEN=500                                       # synthetic sequences per family
EMA=0.999
VALFRAC=0.05
PATIENCE=15
# run_02b fixes: a short LR warmup keeps the scale-up model off the flat init
# basin that stranded zeroaccess (most mov-skewed, largest vocab), and a
# min-epochs floor stops early-stop firing mid-plateau the way it killed
# zeroaccess at epoch 32 in run_02.
WARMUP=0.03      # 3% of total steps: linear LR ramp before cosine anneal
MINEPOCHS=60     # never early-stop before here (rides out the init plateau)
# ----------------------------------------------------------------------------

START=$(date +%s)
for fam in "${FAMILIES[@]}"; do
  echo "============================================================"
  echo "  TRAIN  $fam   (scaleup epochs=$EPOCHS batch=$BATCH max_len=$MAXLEN ema=$EMA)"
  echo "============================================================"
  $PY -m d3pm.train --family "$fam" --scaleup --epochs "$EPOCHS" --batch "$BATCH" \
      --max-len "$MAXLEN" --ema "$EMA" --val-frac "$VALFRAC" --patience "$PATIENCE" \
      --warmup-frac "$WARMUP" --min-epochs "$MINEPOCHS"

  echo "------ GENERATE $fam  (n=$NGEN) ------"
  $PY -m d3pm.generate --family "$fam" --n "$NGEN" --seq-len "$MAXLEN"

  echo "------ EVAL $fam (sequence metrics) ------"
  $PY -m d3pm.seq_eval --family "$fam"

  echo "------ AUDIT $fam (memorization + transition KL) ------"
  $PY -m d3pm.seq_audit --family "$fam"
done

echo
echo "All families done in $(( ($(date +%s) - START) / 60 )) min. Packaging results..."
# COPYFILE_DISABLE + --exclude keep macOS/junk metadata out of the tarball.
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' --exclude='__pycache__' \
  -czf d3pm_results.tar.gz checkpoints synthetic eval_results run_all.log 2>/dev/null \
  || COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' --exclude='__pycache__' \
       -czf d3pm_results.tar.gz checkpoints synthetic eval_results
echo
echo ">>> Now run:   runpodctl send d3pm_results.tar.gz"
echo ">>> Then on your Mac:   runpodctl receive <code-it-prints>"
