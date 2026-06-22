# VirusShare Generalization — Staged Plan (Phase 2)

> Goal: replicate the **full** Malicia result set (seq-level fidelity, memorization +
> transition audits, DDPM embedding parity, downstream-classifier utility) on VirusShare,
> to show the capability claim is dataset-independent (paper §5.6).
>
> **Key difference from Malicia:** Malicia arrived *pre-disassembled* (`<family>/*.asm.txt`,
> one opcode per line). VirusShare ships **raw binaries with no labels**. So Phase 2 is
> mostly a new *preprocessing pipeline* that produces the exact same on-disk shape the
> existing `d3pm/data.py` already consumes. Once the data looks like Malicia, train /
> generate / seq_eval / seq_audit / comparison all re-run essentially unchanged.

---

## Stage V0 — Access & raw download
- Register for VirusShare access (historically invite/registration-gated). Pull one or two
  hash-chunk torrents — you do NOT need all of it; you need enough samples for ~3–5 families
  comparable to Malicia (winwebsec had 4360 files; aim for a few thousand/family).
- Decide platform scope: **PE/Windows only** keeps the opcode vocabulary comparable to
  Malicia (x86). Mixing ELF/APK changes the vocab and muddies the cross-dataset claim — keep
  it PE for the generalization argument; note multi-arch as future work.
- **Output:** `virusshare/raw/<sha256>` raw files.
- **Verify:** count files, confirm PE magic (`MZ`) on a sample with `pefile`/`LIEF`.

## Stage V1 — Family labeling
- VirusShare gives no family labels. Resolve them: run **AVClass / AVClass2** over VirusTotal
  reports for each hash (or VirusShare-bundled AV detections if available).
- Keep only samples with a confident single-family label; drop singletons/unknowns.
- Pick families with enough labeled samples to train (≥ ~500, ideally a few thousand).
- **Output:** `virusshare/labels.csv` (sha256 → family).
- **Verify:** family histogram; chosen families have sufficient counts; spot-check labels.

## Stage V2 — Disassembly → opcode sequences  ← the real new engineering
- Turn each PE into an opcode-per-line file matching Malicia's format.
  - Parse the `.text` (executable) section with `pefile`/`LIEF`.
  - Disassemble with **Capstone** (x86, 32/64 per the PE header). Linear sweep is simplest
    and reproducible; recursive/`angr` is higher fidelity but slower — start linear.
  - Emit **mnemonics only**, one per line (drop operands), to match `*.asm.txt`.
- Handle packed/obfuscated samples: many VirusShare PEs are packed → tiny/garbage `.text`.
  Filter by min opcode count and entropy; optionally note unpacking as a limitation.
- **Output:** `virusshare/<family>/<sha256>.asm.txt` — identical shape to `malicia/`.
- **Verify:** opcode vocab size per family (expect ~150–250 like Malicia), length
  distribution, that `d3pm.data.load_family_opcodes("virusshare", fam)` loads them. Add a
  `verify/verify_virusshare_data.py` mirroring `verify_data.py`.

## Stage V3 — Plug into existing pipeline (minimal code change)
- `d3pm/train.py`, `generate.py`, `seq_eval.py`, `seq_audit.py` already take a dataset-root
  arg (`--malicia`). Either pass `--malicia virusshare` or add a clearer `--data-root` flag.
- Per chosen family: `train` → `generate` (500+ synth) → `seq_eval` → `seq_audit`.
- **Verify:** same green checks as Malicia (no MASK/PAD in output, KL near real-vs-real
  floor, no mode collapse, novelty gap negative in the audit).

## Stage V4 — DDPM baseline on VirusShare (for the parity comparison)
- ⚠ The frozen `comparison/ddpm_baseline_report.json` is **Malicia-specific**. To claim DDPM
  parity on VirusShare you must re-run the DDPM/Word2Vec embedding baseline on VirusShare
  opcodes too (same protocol as the Bao et al. pipeline used for Malicia).
- Then re-run the embedding comparison suite in `comparison/` against the new baseline.
- **Verify:** D3PM ≈ DDPM on the 6 embedding metrics, as on Malicia (expect parity, with
  vocab-coverage effects on cosine-dev).

## Stage V5 — Downstream utility on VirusShare (the headline, generalized)
- Re-run `comparison/lib/seq_classifier.py`: multi-class family classification (the new
  VirusShare families) from n-gram TF-IDF + LinearSVC, restricted real-data budget, fixed
  held-out real test set, 3 seeds, real-only vs real+D3PM-synthetic.
- **Success = a positive low-data macro-F1 lift that shrinks to ~0 at full data**, mirroring
  Malicia's +4.1pt curve. This is the result that makes §5.6 worth its own section.

## Stage V6 — Write-up
- Fill paper §5.6 with VirusShare versions of every Malicia table/figure. Lead the cross-
  dataset paragraph with: "the downstream-utility lift reproduces on an independent corpus."

---

## Experiment matrix (what actually gets run)
| family (×3–5) | train ckpt | 500+ synth | seq_eval | seq_audit | embed-compare | downstream util |
|---|---|---|---|---|---|---|
| VS fam A | V3 | V3 | V3 | V3 | V4 | V5 |
| VS fam B | … | … | … | … | … | … |

GPU budget: Malicia scale-up was ~408 min for 3 families on the pod. Budget similarly per
VirusShare family; the disassembly/labeling (V1–V2) is CPU and can be slow on many samples.

## Biggest risks / gotchas
1. **Access latency** — VirusShare registration can take time; start V0 immediately.
2. **Packing** — packed PEs yield useless opcode streams; expect to discard a chunk of
   samples and document the filter. This is the single most likely thing to dent quality.
3. **Disassembly fidelity** — linear sweep mis-decodes data-in-code; acceptable for a
   statistical opcode model, but note it.
4. **DDPM re-baseline (V4)** is real work, not a re-run — don't forget it's required for the
   parity claim on the new dataset.
5. **Keep PE-only** to preserve a comparable vocabulary; multi-arch is future work.
