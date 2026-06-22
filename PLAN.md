# D3PM (Absorbing-State) for Synthetic Malware — Plan & Progress

> **For a future Claude Code session:** this file is the single source of truth for
> what we're building, why, the math, the staged plan, and **how far we've gotten**.
> Read this top-to-bottom plus `CLAUDE.md` and you have full context to resume.
> Jump to the **Progress / Resume Here** section for current status.

---

## Goal

Build a **Discrete Denoising Diffusion Probabilistic Model (D3PM)** from scratch to
generate synthetic malware **opcode sequences** for the Malicia dataset, as a
discrete-data alternative to the continuous DDPM-on-embeddings approach in
`SyntheticMalware.pdf`.

**Why D3PM fits:** each Malicia sample is a sequence of x86 opcode mnemonics
(`malicia/<family>/*.asm.txt`, one opcode per line). This is *discrete categorical*
data with a *small* vocabulary (~196 unique opcodes per family, K ≈ 200). D3PM models
discrete tokens directly — no Word2Vec embedding, no continuous relaxation. The DDPM
baseline had to mean-pool each file into a 104-dim vector and lost the actual sequence;
D3PM generates real opcode sequences.

## Working agreement (decided with the user)

- **Absorbing-state (MASK) variant only.** Each token either stays or jumps to a special
  `[MASK]` token; the forward process ends in all-`[MASK]`. This is the paper's BERT-like
  variant and its best performer on text-like data. We do **not** implement the general
  uniform / Gaussian / embedding transition matrices. (The user will only run this on
  Malicia and, later, VirusShare opcodes.)
- **Fresh, heavily-documented rebuild.** The user is **new to PyTorch** and wants to
  understand every line and **verify correctness at each stage**. Every stage ends in a
  concrete runnable check (pytest + a `verify/` script with expected output stated).
- **Build incrementally** — do not start a stage until the previous stage's verification
  passes. Explain each new PyTorch concept inline as it first appears.
- **Sequence-level evaluation first.** The DDPM comparison is a later, separate effort.

## Reference implementations (read, don't copy)

A complete but terse absorbing D3PM already exists in the sibling project
`../diffusion-proj`. Consult it as a north star; we re-implement fresh and documented.
- `../diffusion-proj/mdiff/data/loader.py` — `load_family_opcodes()`
- `../diffusion-proj/mdiff/data/vocab.py` — `Vocabulary`, `OpcodeChunkedDataset`
- `../diffusion-proj/mdiff/models/d3pm/model.py` — terse absorbing D3PM (architecture, loss, sampling)
- `../diffusion-proj/mdiff/seq_evaluate.py` — sequence metrics (reuse in Stage 7)
- `../diffusion-proj/mdiff/evaluate.py` — embedding/classifier DDPM-vs-D3PM suite (later)

---

## The D3PM math we're implementing (absorbing state)

All from the D3PM paper (Austin et al., NeurIPS 2021); `D3PM.pdf` is in this folder.
Equation numbers refer to that paper.

- **Vocabulary layout:** ids `0..K-1` = opcodes, `K` = `[MASK]`, `K+1` = `[PAD]`,
  `vocab_size = K+2`. `[MASK]` is the absorbing state; `[PAD]` only pads batches and is
  excluded from the loss.
- **Noise schedule (§3.2, A.7):** `β_t = 1/(T−t+1)`. Closed-form cumulative "stay
  original" probability `ᾱ_t = (T−t)/T`. So at step `t`, each non-PAD token is replaced
  by `[MASK]` independently with probability `1 − ᾱ_t = t/T`. At `t=T` everything is MASK.
- **Forward sample `q(x_t | x_0)`:** mask each token i.i.d. with prob `t/T`. One line.
- **Reverse process (x₀-parameterization, Eq. 4):** a neural net predicts a distribution
  `p̃_θ(x_0 | x_t)` over the *clean* token at every position; the reverse step is built
  from that prediction plus the known forward posterior.
- **Loss (Eq. 12 + Eq. 5):** for *this* schedule the variational bound reduces exactly to
  a **(1/t)-reweighted cross-entropy on masked positions**:
  `L_vb = E_t[ (1/t) · Σ_{masked i} −log p̃_θ([x_0]_i | x_t) ]`.
  The hybrid objective adds a plain (unweighted) cross-entropy auxiliary term:
  `L_λ = L_vb + λ·L_CE`, with `λ = 0.01` (paper §6, text). Reduction is exact because
  unmasked tokens contribute zero KL and mask-stay transitions are θ-independent constants.
- **Absorbing posterior (used in sampling):** given `x_t` is masked, a token un-masks to
  its true value with prob `1/t` and stays masked with prob `(t−1)/t`.
- **Sampling:** start from all-`[MASK]` at `t=T`; at each step predict `p̃_θ(x_0|x_t)`,
  combine with the posterior to get `p_θ(x_{t-1}|x_t)`, sample only currently-masked
  positions, repeat to `t=1`; final step forbids MASK/PAD and renormalizes.

---

## Data facts (Malicia)

- One folder per family under `malicia/`; inside, `*.asm.txt` files, one opcode per line.
- Big families: `winwebsec` (4360 files), `zbot` (2136), `zeroaccess` (1311). Tiny:
  `cridex` (74), `cleaman` (32), `ramnit` (4).
- ~196 unique opcodes per family → K ≈ 200; vocab_size ≈ 198–200.
- File lengths vary widely (hundreds to ~10k opcodes) → we **chunk** to `max_len`.
- `cleaman` (32 files) is the fast smoke-test family used in verification.

---

## Project structure

```
d3pm_implementation/
  .venv/                 # local virtualenv (torch 2.12, runs on Apple "mps")
  requirements.txt
  PLAN.md                # this file
  CLAUDE.md
  D3PM.pdf  SyntheticMalware.pdf
  malicia/<family>/*.asm.txt
  d3pm/
    __init__.py
    config.py            # device() + Config presets (small/default)       [DONE]
    data.py              # load_family_opcodes, Vocabulary, OpcodeChunkedDataset  [DONE]
    diffusion.py         # AbsorbingDiffusion: schedule, q_sample, loss, sample   [Stages 2,4,6]
    denoiser.py          # TransformerDenoiser (predicts x0)                [Stage 3]
    train.py             # training loop + checkpointing CLI                [Stage 5]
    generate.py          # sampling CLI                                     [Stage 6]
    seq_eval.py          # sequence-level metrics                           [Stage 7]
  tests/
    test_data.py         # [DONE, 6 pass]
    test_schedule.py  test_diffusion.py                                     [Stages 2,4]
  verify/
    verify_data.py       # [DONE, PASS]
    verify_forward.py  verify_loss.py  verify_sample.py                     [Stages 2,4,6]
```

---

## Stages

Each stage: implement → pytest unit tests for the math → a `verify/` script printing
human-readable diagnostics with stated expected output → only then move on.

### Stage 0 — Environment & skeleton ✅ DONE
- `.venv` (never global installs). Installed torch 2.12.0, numpy, tqdm, pytest. MPS available.
- `d3pm/__init__.py`, `d3pm/config.py` (`device()`, `Config.small()/.default()`).
- **Verify:** `import torch` + `device()` → prints `mps`. PASS.

### Stage 1 — Data pipeline ✅ DONE
- `d3pm/data.py`: `load_family_opcodes`, `Vocabulary` (MASK=K, PAD=K+1, encode/decode/
  save/load), `OpcodeChunkedDataset` (non-overlapping max_len chunks, drop tail < min_chunk,
  returns `{tokens, pad_mask}`).
- **Verify:** `tests/test_data.py` (6 pass); `verify/verify_data.py --family cleaman`
  → vocab 198, 272 chunks, batch (8,256) int64, PAD-suffix invariant holds. PASS.

### Stage 2 — Forward (noising) process ✅ DONE
- `d3pm/diffusion.py`: `AbsorbingDiffusion(nn.Module)` with `alpha_bar` buffer of shape
  `(T+1,)`, `alpha_bar[t] = (T−t)/T` (closed form of `β_t = 1/(T−t+1)`, per paper §3.2 /
  App. A.4.1 / App. A.7). `q_sample(x0, t, pad_mask=None)`: independently masks each non-PAD
  token with prob `1 − ᾱ_t = t/T` (broadcasts a per-example `(B,1)` mask probability across
  the `L` axis); PAD positions are excluded.
- **Verify:** `tests/test_schedule.py` (15 pass) — schedule endpoints/monotonicity/closed
  form/buffer-not-param/state-dict roundtrip, q_sample shape/dtype, input validation,
  `t=0` identity, `t=T` fully masks non-PAD, PAD preserved at every t, empirical
  masked-fraction ≈ `t/T`, per-example t broadcasting.
  `verify/verify_forward.py --family cleaman` PASS: ᾱ_0=1.0 / ᾱ_T=0.0 / monotone /
  matches `(T-t)/T`; on a 64×128 cleaman batch the observed mask fraction at
  `t=25/50/75/100` was `0.243/0.497/0.748/1.000` vs expected `0.25/0.50/0.75/1.00`
  (all PAD intact); a 30-token sequence visibly dissolves from real opcodes at `t=0` to
  all `[MASK]` at `t=T`.

### Stage 3 — Denoiser network ✅ DONE
- `d3pm/denoiser.py` `TransformerDenoiser`: token emb (`padding_idx=pad`) + learned
  positional emb + sinusoidal time emb (→SiLU MLP) added to every position; pre-LN
  `nn.TransformerEncoder` (`batch_first=True`, `norm_first=True`, final LayerNorm,
  `enable_nested_tensor=False`); `src_key_padding_mask` from `x_t == pad_idx`; linear head
  → logits `(B, L, vocab)`. Also `sinusoidal_time_embedding(t, dim)` helper and
  `num_parameters()`. Derived from `D3PM.pdf` §3.3 (x₀-param) + standard transformer refs.
- **Verify:** `tests/test_denoiser.py` (17 pass) — time-emb shape/values/smoothness,
  forward shape/finite, accepts L<max_len, rejects L>max_len / bad t / non-2D, time
  conditioning changes output, eval determinism, PAD row zero, **trailing PAD doesn't
  change real-position logits** (padded-vs-unpadded), nhead-divides-d_model guard.
  `verify/verify_denoiser.py --family cleaman` PASS: output `(8,128,198)` finite; params
  small=117k / default=1.0M (plan's "~2M" was a loose estimate; d_model=128/layers=4 → ~1M);
  `max|logits(t=1)−logits(t=T)| = 2.03` (time conditioning works), same-t diff = 0
  (deterministic in eval), PAD-isolation diff = 8e-7.

### Stage 4 — Training loss ✅ DONE
- `d3pm/diffusion.py` `AbsorbingDiffusion.loss(model, x0, pad_mask, lambda_ce=0.01)`
  (takes the denoiser as an arg so the diffusion module stays parameter-free; the plan's
  `loss(x0, pad_mask)` was shorthand). Sample `t ~ U{1..T}` per example; `x_t = q_sample`;
  `logits = model(x_t, t)`; `nll = −log_softmax(logits).gather(x0)`; score positions that
  are **masked AND not PAD**; `L_vb = Σ_scored (1/t)·nll / N_nonpad`,
  `L_CE = Σ_scored nll / N_nonpad`; return dict `{loss = L_vb + λ·L_CE, vb, ce}`.
  **Key normalization choice:** divide by the (t-independent) non-PAD token count, *not*
  the masked count — the `1/t` weight is what balances each timestep's contribution
  (≈`t/T` of tokens masked × `1/t` ≈ constant), and dividing by a t-dependent count would
  destroy that balance. Docstring documents the Eq. 4–5 / App. A.3 reduction (unmasked →
  zero KL; masked → `(1/t)·(−log p̃_θ(x0))` + θ-independent const).
- **Verify:** `tests/test_diffusion.py` (8 pass) — finite scalar dict; `backward()` fills
  grads; uses a `StubDenoiser` oracle to prove the loss scores **only masked** positions
  (perfect→~0; wrong-on-unmasked→~0; wrong-on-PAD→~0; wrong-on-masked→loss>1); works with
  `pad_mask=None`. `verify/verify_loss.py --family cleaman` PASS: 34/34 param groups get
  non-zero grad; `ce` starts ~4.2 ≈ ln(198); **overfit one 4×128 batch 400 steps** →
  first-step loss 0.10 → final 20-step avg 0.005 (vb 0.058→0.004), <25% of start.

### Stage 5 — Training loop ✅ DONE
- `d3pm/train.py`: `train(...)` helper + CLI (`--family --malicia --out-dir --max-files
  --epochs --batch --lr --T --max-len --min-chunk --small --seed`). Builds vocab+dataset+
  `DataLoader`(shuffle, drop_last)+model+`Adam`+`CosineAnnealingLR` (stepped per batch),
  trains (`model.train()`, zero_grad/backward/`clip_grad_norm_(1.0)`/step/sched.step),
  tracks epoch-mean loss, saves `checkpoints/<family>/{d3pm.pt, vocab.pkl, losses.json}`.
  `d3pm.pt` stores `model_state` + `model_config` + `diffusion_config` + `lambda_ce` so the
  net can be rebuilt exactly. `load_checkpoint(family, ...)` reverses it → `(model,
  diffusion, vocab, meta)`, model in `.eval()` — Stage 6's sampler builds on this.
- **Concepts:** `DataLoader`/batching, Adam, optimizer/zero_grad/backward/step cycle, grad
  clipping, cosine LR scheduler, `state_dict` save/load, `model.train()/.eval()`.
- **Verify:** `tests/test_train.py` (5 pass) — tiny CPU run returns finite epoch-loss list,
  writes the 3 files, `load_checkpoint` reload matches every weight bit-for-bit + identical
  eval logits + correct config. `verify/verify_train.py --family cleaman --max-files 50
  --epochs 20` PASS on MPS: 32 files → 537 chunks, vocab 198, 117k params; epoch loss
  0.0362→0.0259 (last < 0.85·first; loss is small because L_vb is (1/t)-reweighted and
  normalized by non-PAD count — the *trend* is the signal); checkpoint roundtrip: 34/34
  state-dict tensors match (max weight diff 0.0), eval logits deterministic.

### Stage 6 — Sampling (reverse process)  ✅ DONE
- `d3pm/diffusion.py` `AbsorbingDiffusion.sample(model, n, seq_len, device, return_trajectory=False)`
  under `@torch.no_grad()`: start all-`[MASK]`; for `t=T..1` predict `p̃_θ(x_0|x_t)` (softmax
  of logits, with `[MASK]`/`[PAD]` logits set to `-inf` **every** step so a written candidate is
  always a real opcode — a strict superset of the plan's "only at t=1"), sample one candidate
  per position with `torch.multinomial`, and un-mask each *currently-masked* position with a
  Bernoulli(`1/t`) coin (derived: `P(unmask | masked@t) = ᾱ_{t-1}·β_t / (1−ᾱ_t) = 1/t`); at
  `t=1` the prob is 1 so all remaining positions un-mask → output guaranteed MASK-free.
  `d3pm/generate.py` `generate(...)` helper + CLI (`--family --n --seq-len --out-dir --ckpt-dir
  --seed`): `load_checkpoint` → `sample` → `vocab.decode` → write `synthetic/<family>/seq_*.txt`
  (one opcode/line, matching the real format). `seq_len` defaults to the model's trained `max_len`.
- **Verify:** `tests/test_sample.py` (6 pass) — shape/dtype; output clean (no MASK/PAD); oracle
  fixed-token denoiser fills the whole output; forbidden MASK/PAD still yields clean output;
  trajectory is `T+1` snapshots, starts all-MASK, ends == sample, mask-count monotonically
  non-increasing (absorbing); deterministic under a seed. `verify/verify_sample.py --family
  cleaman` PASS (reuses the Stage-5 checkpoint): 16×128 samples, **0** MASK/PAD ids, all len 128,
  **top-10 opcode overlap 9/10** with real (mov/push/add/inc/dec/cmp/jz/jb/call), decoded previews
  look like real opcode streams.

### Stage 7 — Sequence-level evaluation ✅ DONE
- `d3pm/seq_eval.py`: (1) `opcode_freq_kl` — smoothed unigram KL (both directions +
  symmetric, with vocab-overlap stats); (2) `ngram_overlap` — corpus-level
  precision/recall/F1 over n-gram **type sets** for n=1,2,3; (3) `edit_distance_stats` —
  sampled token-level Levenshtein for real↔synth, real↔real (floor), synth↔synth
  (diversity / mode-collapse detector). `evaluate(...)` wraps all three and frames every
  number with two baselines: `baseline_real_vs_real` (split-halves **upper bound**) and
  `baseline_real_vs_random` (uniform opcodes, length-matched, **lower bound**); prints a
  model-vs-baselines table. `load_synthetic` reads `synthetic/<family>/seq_*.txt`. CLI
  (`--family --malicia --synthetic --out-dir --max-files --n-pairs --seed`) writes
  `eval_results/<family>_seq_eval.json`.
- **Verify:** `tests/test_seq_eval.py` (11 pass) — KL ≥0 / zero-for-identical / ordered /
  finite under disjoint vocab; n-gram perfect-for-identical / unigram-blind-to-order /
  hand-computed P&R; edit dist zero-for-identical / grows-with-difference; split-halves
  partitions; random_like length+vocab match; full-report shape + model-beats-random.
  `verify/verify_seq_eval.py --family cleaman` PASS (Stage-5 ckpt, 16 samples): symmetric
  KL model **0.169** vs real-vs-real floor 0.032 vs random 2.45; 2-gram F1 **0.220** vs
  random 0.074; 3-gram F1 **0.073** vs random 0.001; 1-gram F1 0.653 vs real-floor 0.886;
  synth↔synth edit dist 99.2 (no collapse). All 5 baseline-relative checks PASS.
- **Note:** numbers above are from the tiny cleaman smoke-test checkpoint; coverage/F1 will
  rise with a bigger model + more/longer samples (scale-up at the bottom of this file).

### Later (out of scope now)
DDPM-vs-D3PM comparison via the embedding/classifier suite in
`../diffusion-proj/mdiff/evaluate.py` (real-vs-synth F1, t-SNE, fidelity). After generation
quality is solid.

---

## End-to-end verification (after all stages)

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m d3pm.train    --family cleaman --max-files 32 --epochs 30 --T 250 --max-len 128
.venv/bin/python -m d3pm.generate --family cleaman --n 50
.venv/bin/python -m d3pm.seq_eval --family cleaman
```
Success = tests pass, training loss falls, generated files have no MASK/PAD and a plausible
opcode mix, unigram KL is low with real-vs-real as the reference floor. Then scale up
(`zbot`/`winwebsec`, `T=500`, `max_len=256+`, more epochs), ideally on a GPU.

---

## Progress / Resume Here

| Stage | Status | Artifacts |
|------|--------|-----------|
| 0 — Environment & skeleton | ✅ done, verified | `.venv`, `requirements.txt`, `d3pm/config.py` |
| 1 — Data pipeline | ✅ done, verified | `d3pm/data.py`, `tests/test_data.py` (6 pass), `verify/verify_data.py` (PASS) |
| 2 — Forward (noising) process | ✅ done, verified | `d3pm/diffusion.py` (`AbsorbingDiffusion`), `tests/test_schedule.py` (15 pass), `verify/verify_forward.py` (PASS) |
| 3 — Denoiser network | ✅ done, verified | `d3pm/denoiser.py`, `tests/test_denoiser.py` (17 pass), `verify/verify_denoiser.py` (PASS) |
| 4 — Training loss | ✅ done, verified | `d3pm/diffusion.py` (`loss`), `tests/test_diffusion.py` (8 pass), `verify/verify_loss.py` (PASS, overfits to ~0) |
| 5 — Training loop | ✅ done, verified | `d3pm/train.py` (`train` + `load_checkpoint` + CLI), `tests/test_train.py` (5 pass), `verify/verify_train.py` (PASS: loss falls, checkpoint roundtrips exactly) |
| 6 — Sampling | ✅ done, verified | `d3pm/diffusion.py` (`sample`), `d3pm/generate.py` (`generate` + CLI), `tests/test_sample.py` (6 pass), `verify/verify_sample.py` (PASS: clean output, 9/10 top-opcode overlap) |
| 7 — Sequence evaluation | ✅ done, verified | `d3pm/seq_eval.py` (`opcode_freq_kl`/`ngram_overlap`/`edit_distance_stats`/`evaluate` + CLI), `tests/test_seq_eval.py` (11 pass), `verify/verify_seq_eval.py` (PASS: KL 0.17 vs floor 0.03/random 2.45; n-gram F1 ≫ random) |
| **All core stages complete** | 🎉 | 68 tests pass. Next: scale up (zbot/winwebsec, larger T/max_len/epochs) then the DDPM comparison. |

**Setup for a fresh machine/session:**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q                       # confirm Stage 1 still green
.venv/bin/python -m verify.verify_data --family cleaman    # confirm data pipeline
```

**To resume:** start Stage 7 — implement **sequence-level evaluation** in
`d3pm/seq_eval.py` (adapt `../diffusion-proj/mdiff/seq_evaluate.py` for ideas, but
keep it fresh and documented). Generation already works: a trained checkpoint can be
sampled with `d3pm/generate.py` (writes `synthetic/<family>/seq_*.txt`), and in-process
via `model, diffusion, vocab, meta = load_checkpoint(family)` then
`diffusion.sample(model, n, seq_len, dev)`. Build the metrics comparing real
(`load_family_opcodes("malicia", family)`) vs. synthetic:

1. **Opcode-frequency KL** real‖synth (unigram distributions over the shared opcode set,
   with smoothing for unseen opcodes). Lower = closer.
2. **n-gram precision/recall/F1** for n = 1, 2, 3 (overlap of the n-gram sets/multisets).
3. **Edit-distance distribution** (e.g. Levenshtein) between synthetic and nearest real.

**Verify (`verify/verify_seq_eval.py`):** a well-trained model → low unigram KL and
non-trivial 2/3-gram overlap. **Contextualize every metric with baselines:** real-vs-real
(split the real set in half) as the achievable *upper bound*, and synthetic-vs-shuffled (or
synthetic-vs-uniform-random) as the *lower bound*, so the numbers are interpretable rather
than absolute. Then the end-to-end run at the bottom of this file (train → generate →
seq_eval on cleaman) should pass, after which scale up to `zbot`/`winwebsec` with larger
`T`/`max_len`/epochs.
