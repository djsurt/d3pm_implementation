"""Per-family output layout.

All artifacts for a single malware family live under one directory in each of
checkpoints/, synthetic/, eval_results/. File names within those dirs are by
*kind* (no family prefix), so the same code paths work for every variant.
"""

from pathlib import Path


# ── per-family output directories ───────────────────────────────────────────


def family_ckpt_dir(ckpt_root: Path, family: str) -> Path:
    p = ckpt_root / family
    p.mkdir(parents=True, exist_ok=True)
    return p


def family_synth_dir(synth_root: Path, family: str) -> Path:
    p = synth_root / family
    p.mkdir(parents=True, exist_ok=True)
    return p


def family_eval_dir(eval_root: Path, family: str) -> Path:
    p = eval_root / family
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── canonical file names within a family directory ─────────────────────────

# Continuous DDPM
DDPM_MODEL = "ddpm.pt"
DDPM_EMBEDDINGS = "ddpm_embeddings.npy"   # real W2V embeddings (DDPM space)
DDPM_LOSSES = "ddpm_losses.json"
DDPM_SYNTHETIC = "ddpm.npy"

# Discrete D3PM
D3PM_MODEL = "d3pm.pt"
D3PM_VOCAB = "d3pm_vocab.pkl"
D3PM_LOSSES = "d3pm_losses.json"
D3PM_W2V = "d3pm_w2v.pkl"
D3PM_REAL_EMBEDDINGS = "d3pm_real_embeddings.npy"
D3PM_SYNTHETIC = "d3pm.npy"
D3PM_SEQ_DIR = "d3pm_sequences"

# Eval (per-family rollup files; tsne is variant-prefixed inline as f"{variant}_tsne.png")
SEQ_EVAL = "seq_eval.json"
