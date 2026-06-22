#!/usr/bin/env bash
# Train -> generate -> eval -> audit D3PM on VirusShare families (data root = virusshare/).
# Mirrors run_all.sh; the ONLY structural change is `--malicia virusshare`.
# Run inside tmux:  bash runpod/run_virusshare.sh 2>&1 | tee run_virusshare.log
set -euo pipefail
cd "$(dirname "$0")/.."        # repo root
PY=${PY:-.venv/bin/python}    # override on Colab (no venv): PY=python bash ...
DATA=virusshare               # data root (family folders live here)

# --- families ---------------------------------------------------------------
# Start with a 3-family pilot to validate the pipeline end-to-end on VirusShare,
# then expand. ALL trainable families (>=~500 unique, safe to generate from):
#   Small CeeInject Renos Toga!rfn Hotbar Injector FakeRean Expiro.BK \
#   OnLineGames Allaple.A VBInject Beebone Obfuscator Vundo Vobfus
# Too sparse to TRAIN a generator (classifier-only): Lamechi.B Diplugem Systex.A Startpage Delf
FAMILIES=("Vobfus" "Vundo" "Obfuscator")     # pilot: 3 largest by unique count
# ---------------------------------------------------------------------------

EPOCHS=150; BATCH=64; MAXLEN=512; NGEN=500
EMA=0.999; VALFRAC=0.05; PATIENCE=15; WARMUP=0.03; MINEPOCHS=60

# --- PILOT ONLY: cap files per family --------------------------------------
# VirusShare files are FAR longer than Malicia's (Vobfus median ~55k opcodes/file
# vs Malicia's few-hundred-to-10k). Uncapped, the 3 pilot families yield ~790k
# 512-chunks (~14x the Malicia scaleup's ~56k) -> ~90+ GPU-hours. Capping at 300
# files/family lands near Malicia scale (~30k chunks/family, ~6h total) so we can
# validate the pipeline first. Raise/remove MAXFILES for the full corpus run.
# Note: only TRAIN intake is capped; seq_eval/seq_audit see the FULL real set.
MAXFILES=${MAXFILES:-300}

START=$(date +%s)
for fam in "${FAMILIES[@]}"; do
  echo "============================================================"
  echo "  TRAIN  $fam   (data=$DATA scaleup epochs=$EPOCHS batch=$BATCH max_len=$MAXLEN)"
  echo "============================================================"
  $PY -m d3pm.train --malicia "$DATA" --family "$fam" --scaleup --epochs "$EPOCHS" \
      --batch "$BATCH" --max-len "$MAXLEN" --max-files "$MAXFILES" --ema "$EMA" \
      --val-frac "$VALFRAC" --patience "$PATIENCE" --warmup-frac "$WARMUP" \
      --min-epochs "$MINEPOCHS"

  echo "------ GENERATE $fam  (n=$NGEN) ------"
  $PY -m d3pm.generate --family "$fam" --n "$NGEN" --seq-len "$MAXLEN"

  echo "------ EVAL $fam (sequence metrics) ------"
  $PY -m d3pm.seq_eval --malicia "$DATA" --family "$fam"

  echo "------ AUDIT $fam (memorization + transition KL) ------"
  $PY -m d3pm.seq_audit --malicia "$DATA" --family "$fam"
done

echo
echo "All families done in $(( ($(date +%s) - START) / 60 )) min. Packaging results..."
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' --exclude='__pycache__' \
  -czf d3pm_virusshare_results.tar.gz checkpoints synthetic eval_results run_virusshare.log 2>/dev/null \
  || COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' --exclude='__pycache__' \
       -czf d3pm_virusshare_results.tar.gz checkpoints synthetic eval_results
echo ">>> Now run:   runpodctl send d3pm_virusshare_results.tar.gz"
