# Paper Outline — D3PM for Synthetic Malware (journal length)

> Target: full journal paper. Through-line decided in `FINDINGS.md §3`: a **capability**
> argument, not a metric battle. D3PM generates real opcode *sequences*; DDPM-on-embeddings
> only emits one mean-pooled vector. On DDPM's own embedding metric they're at parity; D3PM's
> sequence output unlocks tasks DDPM structurally cannot serve, and that delivers measurable
> downstream value. VirusShare is added as the generalization study (§ Results, second dataset).

## Working title
*Discrete Denoising Diffusion for Synthetic Malware: Generating Real Opcode Sequences
as a Capable Alternative to Embedding-Space DDPM*

## Section plan (→ artifact that already backs each claim)

1. **Introduction**
   - Problem: synthetic malware for data augmentation / scarcity. The DDPM-on-embeddings
     baseline (`SyntheticMalware.pdf`, Bao et al., IEEE Access 2025).
   - Gap: embedding DDPM throws away the sequence; output is a single 104-dim vector,
     unrecoverable to opcodes → can't feed any sequence model, can't be audited for copying.
   - Contributions (4): (a) a from-scratch, fully-derived absorbing-state D3PM for opcode
     sequences; (b) a sequence-level eval suite framed by real-vs-real / random baselines;
     (c) a memorization + transition-structure audit; (d) demonstrated downstream utility
     and DDPM parity, replicated on **two** datasets (Malicia + VirusShare).

2. **Background & Related Work**
   - DDPM / continuous diffusion; the Bao et al. embedding approach (summarize honestly).
   - D3PM (Austin et al. 2021) — absorbing-state variant; cite `D3PM.pdf`.
   - Malware representation: opcode sequences, n-gram models, prior synthetic-malware work.

3. **Method** — derive, don't hand-wave (you have it all in `PLAN.md`)
   - Data representation: opcode-per-line, vocab layout (opcodes / MASK / PAD), chunking.
   - Forward process: `β_t=1/(T−t+1)`, closed form `ᾱ_t=(T−t)/T`, i.i.d. masking.
   - x₀-parameterized reverse process; absorbing posterior `1/t` un-mask.
   - Loss: `(1/t)`-reweighted CE on masked positions + λ CE hybrid; the Eq.4–5 reduction
     and the *normalize-by-non-PAD-count* argument (your key derivation).
   - Architecture: transformer denoiser (token + positional + sinusoidal-time embeddings).
   - Sampling algorithm (forbid MASK/PAD logits every step).
   - **Figure:** forward dissolution `t=0→T` (you have this in `verify_forward`).

4. **Experimental Setup**
   - Datasets: Malicia (pre-disassembled, 3 families) + VirusShare (your new pipeline —
     disassembly + AVClass labeling; describe it as a methodological contribution).
   - Configs (baseline run_01 vs scale-up run_02); hardware (RTX 4090), training budget.
   - Metrics: unigram symKL, n-gram P/R/F1 (n=1,2,3), edit-distance diversity,
     memorization Jaccard, transition-matrix KL, embedding suite (6 metrics), downstream
     classifier macro-F1. **Every metric framed by real-vs-real floor + random ceiling.**

5. **Results**
   - 5.1 Sequence-level fidelity (Malicia) — `FINDINGS.md §1` table.
   - 5.2 Scale-up ablation — run_01 vs run_02 (`results_scaleup/`).
   - 5.3 Memorization + transition audit — `FINDINGS.md §1.5`.
   - 5.4 DDPM embedding parity — `FINDINGS.md §2` (frame as parity, not loss).
   - 5.5 **Downstream utility (headline)** — `FINDINGS.md §2.5`, the +4.1pt low-data lift,
     3 seeds, monotone decay to ~0 at full data. This is the section that carries the paper.
   - 5.6 **VirusShare generalization** — second dataset, new families (Obfuscator/Vobfus/
     Vundo). The claim is "the fidelity + non-memorization result is dataset-independent."
     **Done** (`FINDINGS.md §2.6`): seq-level fidelity, memorization audit, transition KL,
     convergence — all replicate run_01 (zero exact-copies across all 3 families).
     **Still to run for the full capability claim:** the downstream classifier utility
     curve (§5.5 analogue) on VirusShare — this is the headline experiment and has *not*
     been run on these families yet. DDPM parity (§5.4): Bao et al. *did* train on VirusShare
     (top-3 Malicia + 22 largest VS families), but our reproducible DDPM baseline
     (`comparison/ddpm_baseline_report.json`) only covers the 3 Malicia families — there are
     NO local DDPM numbers for Obfuscator/Vobfus/Vundo. To add an embedding-parity panel here
     we'd either lift per-family numbers from the paper (if it reports VS families individually
     — likely aggregated, verify against `SyntheticMalware.pdf`) or train our own DDPM-on-
     embeddings on these 3 families via `comparison/lib`. Optional; the cleaner §5.6 framing is
     seq-level fidelity + audit + downstream utility.

6. **Discussion**
   - Why parity-not-win is the honest and correct framing (DDPM optimizes its home metric).
   - Capability vs metric: extrinsic utility can't be gamed by a saturated distance metric.
   - Limitations: vocab under-coverage / low n-gram recall; eval-budget asymmetry; chunking
     caps long-range structure; single architecture family.

7. **Conclusion + Future Work**
   - Cross-architecture (PE/ELF/APK opcode sets), control over family conditioning,
     longer-context models.

## How to start (recommended order)
1. **Method + Setup first** — fully done in code/PLAN; lowest-friction, gets momentum.
2. **Results 5.1–5.5** — every number already exists in `FINDINGS.md`; transcribe + make
   the tables/figures. Decide final figures: forward dissolution, augmentation curve
   (utility), seq-level bar charts vs floor/ceiling, t-SNE (from `comparison/results/`).
3. **Intro + Related Work** — write last, once the story's framing is concrete on the page.
4. **VirusShare results (5.6)** — fill in as Phase 2 (`VIRUSSHARE_PLAN.md`) completes.

Draft in LaTeX (IEEEtran or journal template). Keep `FINDINGS.md` as the source of truth
for numbers; never hand-copy a number you can't point back to a JSON artifact.
