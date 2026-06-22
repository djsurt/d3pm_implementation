#!/usr/bin/env bash
# One-time environment setup on the RunPod pod.
# Run from anywhere inside the extracted repo:  bash runpod/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."        # cd to repo root regardless of where this is called from

echo "== [1/4] Creating fresh venv (.venv) =="
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

echo "== [2/4] Installing torch (CUDA 12.4) + deps =="
# Try the exact pin we developed against; fall back to the latest cu124 torch
# if that specific wheel isn't published for cu124.
.venv/bin/pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu124 \
  || .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install numpy==2.4.6 tqdm==4.67.3 pytest==9.0.3

echo "== [3/4] CUDA sanity check =="
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print('CUDA OK ->', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

echo "== [4/4] Running test suite (expect 68 passing) =="
.venv/bin/python -m pytest tests/ -q

echo
echo "Setup complete. Next:  tmux new -s d3pm   then   bash runpod/run_all.sh 2>&1 | tee run_all.log"
