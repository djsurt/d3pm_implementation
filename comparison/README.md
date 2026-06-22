# comparison/ — D3PM vs DDPM (embedding-level, the paper's 6 metrics)

Self-contained copy of the embedding-level evaluation from `../../diffusion-proj`.
Everything needed to compare this project's D3PM against the DDPM baseline lives
here — you never need to open the other folder.

## Why a separate venv

The eval stack needs `gensim`/`scikit-learn`, which have no wheels for the main
project's Python 3.14 venv. So this folder has its own **Python 3.12** venv at
`comparison/.venv`. The main `../.venv` (torch) is untouched.

## Layout

```
comparison/
  .venv/                       # py3.12: gensim, scikit-learn, matplotlib, numpy<2
  lib/
    bridge_d3pm_sequences.py   # D3PM opcode sequences -> mean-pooled W2V embeddings
    evaluate.py                # the 6 paper metrics (--variant ddpm|d3pm)
    loader.py embeddings.py paths.py   # standalone deps (no torch)
  ddpm_baseline_report.json    # FROZEN DDPM numbers — the comparison target
  checkpoints/<fam>/           # ddpm_embeddings.npy + (generated) d3pm_real_embeddings.npy, d3pm_w2v.pkl
  synthetic/<fam>/             # ddpm.npy + (generated) d3pm.npy
  results/                     # d3pm reports + t-SNE plots
```

## Run the comparison (after any D3PM run)

```bash
cd comparison
export PYTHONPATH=lib

# 1. bridge: embed real + this run's synthetic sequences into the W2V space.
#    --d3pm-run points at a run folder holding synthetic/<fam>/seq_*.txt
.venv/bin/python lib/bridge_d3pm_sequences.py --d3pm-run ../runs/run_02_scaleup

# 2. score D3PM and print it side-by-side with the frozen DDPM baseline
.venv/bin/python lib/evaluate.py --variant d3pm \
    --compare-against ddpm_baseline_report.json --out results
```

The bridge **length-matches pooling depth** to real files (groups
`ceil(avg_real_len / chunk_len)` 256-token chunks per synthetic "file", bootstrap
with replacement) so the comparison is fair. Override with `--chunks-per-file 1`
to reproduce the naive per-chunk pooling.

## Recompute the DDPM baseline (rarely needed)

The DDPM `.npy` are included, so you can regenerate the frozen report:

```bash
.venv/bin/python lib/evaluate.py --variant ddpm --out results
cp results/ddpm_full_report.json ddpm_baseline_report.json
```

## Downstream sequence-classifier augmentation (the headline — D3PM-only)

The capability DDPM cannot serve: classifying malware by its opcode *sequence*.
Multi-class family classification (n-gram TF-IDF + LinearSVC over opcode windows)
under a restricted real-data budget, real-only vs real + D3PM-synthetic, scored on
a fixed held-out real test set.

```bash
cd comparison
PYTHONPATH=lib .venv/bin/python lib/seq_classifier.py \
    --d3pm-run ../runs/run_01_baseline --out results/seq_classifier_run01.json
# quick correctness check (lighter sweep, asserts the expected lift curve):
PYTHONPATH=lib .venv/bin/python verify_seq_classifier.py
```

Result (run_01): D3PM augmentation lifts macro-F1 by **+4.1 pts** at 5 real
files/family, shrinking to ~0 at full data — utility where it matters. DDPM has no
sequence output, so it cannot participate. Re-run with `--d3pm-run
../runs/run_02_scaleup` once the scale-up lands for a before/after. See ../FINDINGS.md §2.5.

## The bottom line (see ../FINDINGS.md)

D3PM does **not** win at the embedding level under any fair pooling protocol, and
the metric is fragile (8× swings from pooling choice) and near-saturated. This is
expected — DDPM's native output *is* the mean-pooled vector. **D3PM's real
advantage is at the sequence level** (`python -m d3pm.seq_eval`), where DDPM has no
comparable output. Lead the paper there; use this comparison to show parity.
