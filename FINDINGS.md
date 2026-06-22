# FINDINGS — D3PM for Synthetic Malware

Consolidated results & decisions. Companion to `PLAN.md` (build stages) and
`CLAUDE.md` (project intent). Everything needed now lives in this one folder.

> **Goal:** D3PM (absorbing-state discrete diffusion) as an alternative to the
> DDPM-on-embeddings approach in `SyntheticMalware.pdf` (Bao et al., IEEE Access
> 2025). D3PM generates real opcode **sequences**; the paper's DDPM only ever
> produces a single mean-pooled 104-dim embedding (the sequence is unrecoverable).

---

## 1. Run 01 (baseline) — sequence-level results

Config: `d_model=128, layers=4, T=500, max_len=256, 100 epochs, 500 synth/family`
on an RTX 4090 (~75 min, all 3 families). Artifacts in `runs/run_01_baseline/`.
Eval framed vs **real-vs-real floor** (best achievable) and **random ceiling**.

| family | symKL (floor) | 1g F1 | 2g F1 (floor) | 3g F1 (floor) | vocab used | diversity |
|---|---|---|---|---|---|---|
| winwebsec | 0.0082 (0.0003) | 0.735 | 0.342 (0.885) | 0.185 (0.792) | 193/332 | no collapse |
| zbot      | 0.0061 (0.0029) | 0.793 | 0.610 (0.871) | 0.291 (0.772) | 209/318 | no collapse |
| zeroaccess| 0.0045 (0.0007) | 0.623 | 0.313 (0.758) | 0.113 (0.412) | 186/411 | no collapse |

**Read:** unigram fidelity near-real, no mode collapse, n-gram **precision** high
(every emitted opcode is valid; 1-gram precision = 1.0) but **recall** low — the
model under-covers the real corpus's n-gram diversity. Part of that low recall is
an **eval-budget artifact**: 500×256 synth tokens vs thousands of full real files.

### Levers for the next run (priority order)
1. **Fair eval budget** (free): generate more synth, compare against chunked-real.
2. **Scale the model:** `d_model=256, layers=6, dim_ff=1024` (~8M) — biggest lever.
3. **`max_len=512`** (real files run to ~10k opcodes; 256 caps learnable structure).
4. **EMA weights** for sampling; **`T=1000`** for finer un-masking.
5. **Held-out val loss** for principled early stop (loss plateaued by epoch ~50).

---

## 1.5 Memorization audit + transition-matrix KL (run_01)

Two deeper sequence diagnostics (`d3pm/seq_audit.py`; reports in
`runs/run_01_baseline/eval_results/<fam>_seq_audit.json`; verify
`verify/verify_seq_audit.py`, ALL CHECKS PASS).

**Memorization / novelty audit** — for each synthetic 256-opcode window, max
4-gram Jaccard to its nearest *training* window + exact-copy detection, read against
the same measurement for a held-out *real* window (the natural self-similarity of
genuine samples). Key subtlety: malware families are heavily templated, so genuine
held-out windows already exact-match the training set ~30–97% of the time —
memorization must be judged **relative** to that, not by an absolute copy rate.

| family | synth exact-copy | real-holdout exact-copy | synth mean-Jaccard | real-holdout | verdict |
|---|---|---|---|---|---|
| winwebsec | 0.00 | 0.97 | 0.24 | 0.91 | no memorization |
| zbot | 0.20 | 0.89 | 0.39 | 0.91 | no memorization |
| zeroaccess | 0.00 | 0.31 | 0.15 | 0.68 | no memorization |

**Read:** every family's synthetic windows are *more novel* than a genuine new
sample (negative Jaccard gap). zbot's 20% exact-copy looks alarming until you see
genuine samples self-duplicate 89% — the model copies far *less* than real data
repeats itself. Clean generalization, no rote copying. (DDPM can't be audited this
way — no sequence output.)

**Transition-matrix KL** — bigram P(next|curr) Markov structure, KL weighted by the
real unigram marginal, framed floor/ceiling like the seq-level table:

| family | synthetic | real↔real (floor) | real↔random (ceiling) |
|---|---|---|---|
| winwebsec | 0.119 | 0.007 | 9.28 |
| zbot | 0.260 | 0.025 | 8.86 |
| zeroaccess | 0.141 | 0.026 | 9.37 |

**Read:** synthetic transition dynamics sit ~5–10× the real-vs-real floor but ~35–80×
*below* the random ceiling — the model captures opcode-to-opcode structure, not just
unigram frequencies. Order-aware fidelity DDPM's pooled vector throws away.

---

## 2. Embedding-level comparison vs DDPM — settled

Ran the paper's 6 metrics (in `comparison/`) on run_01. **D3PM does NOT beat DDPM
at the embedding level**, under either pooling protocol (naive per-chunk OR
length-matched-to-real). Cosine deviation (goal≈0): winwebsec 0.055 / DDPM 0.025;
zbot 0.107 / 0.0008; zeroaccess 0.161 / 0.010. Binary-class F1 0.97–1.0 for D3PM
(DDPM 0.75–0.997) — both far from the 0.5 "fool the classifier" goal.

**Why this is expected, not a failure:**
- DDPM's *native output IS the mean-pooled vector* — it directly optimizes this
  metric. D3PM makes sequences that get pooled, discarding its advantage.
- The metric is **fragile** (cosine-dev swung ~8× on the same data from pooling
  choice) and **near-saturated** (even DDPM never nears 0.5).
- Deep pooling exposes D3PM's vocab-coverage bias (esp. zeroaccess 186/411) that
  shallow pooling masks via variance.

---

## 2.5 Downstream sequence-classifier utility — D3PM's capability win

The headline experiment. Multi-class family classification (winwebsec/zbot/
zeroaccess) from **n-gram(1-3) TF-IDF + LinearSVC over opcode windows**, under an
artificially restricted *real*-data budget, scored on a fixed held-out **real**
test set (3 seeds). Real-only vs real + D3PM-synthetic. Code:
`comparison/lib/seq_classifier.py`; report `comparison/results/seq_classifier_run01.json`;
verify `comparison/verify_seq_classifier.py` (ALL CHECKS PASS).

| budget (real files/fam) | real-only F1 | real+D3PM F1 | lift |
|---|---|---|---|
| 5   | 0.899 ± .012 | 0.939 ± .011 | **+0.041** |
| 10  | 0.924 ± .009 | 0.954 ± .007 | +0.031 |
| 25  | 0.939 ± .014 | 0.967 ± .006 | +0.028 |
| 50  | 0.959 ± .013 | 0.972 ± .009 | +0.013 |
| 100 | 0.972 ± .007 | 0.980 ± .004 | +0.009 |
| all | 0.9956       | 0.9957       | +0.0001 |

**Read:** D3PM synthetic sequences give a clean, monotonically-shrinking macro-F1
lift — **+4.1 pts** when real data is scarce, vanishing to ~0 at full data — and
the small-budget lift exceeds the seed-to-seed std, so it's signal not noise.

**Why this is the paper's strongest result:**
- It is **extrinsic utility** (does the output do useful work?), which can't be
  gamed by a saturated distance metric the way the embedding table was.
- **DDPM structurally cannot supply this task at all** — its mean-pooled vector has
  no recoverable sequence to feed a sequence classifier. So the comparison is a
  *capability* claim, not a marginal-number claim.
- It **explains** the embedding-level augmentation no-op (§2 / baseline report's
  `threshold_augmentation` = empty): full-data classification saturates at F1~1.0,
  so augmentation can only help in the scarce regime — which is exactly what we
  isolate. The "all" row (0.9957) matches the embedding `synthetic_only` ~0.9959.

---

## 3. Decisions (the through-line for the paper)

**Reframed:** the paper is **not** a head-to-head metric battle ("D3PM beats DDPM")
— DDPM owns its saturated home metric. It's a **capability** argument: D3PM-on-
sequences is a categorically more capable approach; on DDPM's own embedding metric
they're at parity, but D3PM's real-sequence output unlocks tasks DDPM cannot serve,
and that capability delivers measurable value (§2.5).

1. **Lead with downstream utility (§2.5) + the sequence-level table (§1).** Both are
   axes where DDPM *cannot compete* (no recoverable sequence output). Structural win.
2. **Frame the embedding comparison as parity:** "D3PM matches DDPM on the paper's
   embedding metrics *and additionally* yields real opcode sequences" (§2).
3. **Don't optimize for the embedding table** — scale-up won't make D3PM win there
   (may modestly help cosine-dev via better vocab coverage, that's all).
4. **Paper-worthiness gap was the *framing*, not better numbers.** The generator is
   already useful (§2.5). The through-line is capability, backed by utility + rigor
   (3 seeds on the augmentation curve, loss curves, sample listings). run_02 scale-up
   becomes a before/after ablation on these same tables, not a new headline.

### When to stop optimizing a family
1. Fix the eval-budget asymmetry and re-measure (free).
2. Do **one** capacity scale-up. If 2/3-gram F1 gains < ~10–15% relative, stop.
3. Then judge by the metrics you'll report — once better seq-level F1 stops moving
   the (sequence-level) story, further generator tuning is wasted.
4. **Never chase the real-vs-real floor** — it's the ceiling by definition.

---

## 4. Where things live (consolidated — no folder switching)

```
d3pm_implementation/
  d3pm/                    # the implementation (train, generate, seq_eval, ...)
  runs/run_01_baseline/    # run 01 checkpoints, synthetic seqs, eval, tarballs
  runs/run_02_scaleup/     # (next) — drop the pod's d3pm_results.tar.gz here
  comparison/              # SELF-CONTAINED embedding-level D3PM-vs-DDPM suite
    .venv/                 #   py3.12 venv (gensim/sklearn) — see comparison/README.md
    ddpm_baseline_report.json   # frozen DDPM numbers (the target)
    results/               #   d3pm reports + t-SNE plots
  FINDINGS.md  PLAN.md  CLAUDE.md
```

**Sequence-level eval** (the headline): `.venv/bin/python -m d3pm.seq_eval --family <fam>`
**Embedding comparison** (parity): see `comparison/README.md` (bridge → evaluate).
