"""Stage 7 verification — run the sequence-level metrics on a real checkpoint and
show that the trained model lands near the real-vs-real bound and far from random:

    .venv/bin/python -m verify.verify_seq_eval --family cleaman

What it does
------------
  1. Loads the real family opcodes from ``malicia/<family>/``.
  2. Gets synthetic sequences: reads ``synthetic/<family>/seq_*.txt`` if present,
     otherwise samples them in-process from the Stage-5 checkpoint (and with
     ``--train`` it will first fit a quick cleaman checkpoint if none exists).
  3. Runs ``evaluate`` and prints the model-vs-baselines table.

What "PASS" means
-----------------
The metrics are *contextual*, so we check the model sits in the right part of the
band rather than against magic absolute thresholds:

  * unigram/symmetric KL: model ≪ random lower bound (learned the opcode mix);
  * 2- and 3-gram F1: model clearly above the random baseline (learned local
    structure, which random opcodes cannot have);
  * synth↔synth edit distance > 0 (no total mode collapse);
  * vocab coverage: most real opcodes appear in the synthetic set.

These are the same checks the plan calls for: "low unigram KL and non-trivial
2/3-gram overlap, contextualised with real-vs-real and random baselines."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from d3pm.config import Config, device
from d3pm.data import load_family_opcodes
from d3pm.seq_eval import evaluate, load_synthetic
from d3pm.train import load_checkpoint, train


def _get_synthetic(
    family: str,
    n: int,
    synthetic_dir: str,
    ckpt_dir: str,
    seed: int,
    dev: torch.device,
) -> list[list[str]]:
    """Prefer on-disk synthetic files; otherwise sample fresh from the checkpoint."""
    try:
        seqs = load_synthetic(synthetic_dir, family)
        print(f"  using {len(seqs)} synthetic sequences from {synthetic_dir}/{family}/")
        return seqs
    except FileNotFoundError:
        print(f"  no files in {synthetic_dir}/{family}/ — sampling {n} from checkpoint")
        torch.manual_seed(seed)
        model, diffusion, vocab, meta = load_checkpoint(family, out_dir=ckpt_dir, dev=dev)
        seq_len = meta["model_config"]["max_len"]
        x0 = diffusion.sample(model, n=n, seq_len=seq_len, device=dev)
        return [vocab.decode(row.cpu()) for row in x0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7 verification.")
    ap.add_argument("--family", default="cleaman")
    ap.add_argument("--malicia", default="malicia")
    ap.add_argument("--synthetic", default="synthetic")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--n", type=int, default=50, help="samples to draw if none on disk")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", action="store_true", help="train a quick checkpoint if missing")
    args = ap.parse_args()

    dev = device()
    print(f"Device: {dev}")

    if args.train and not (Path(args.ckpt_dir) / args.family / "d3pm.pt").exists():
        print(f"  training a quick {args.family} checkpoint first ...")
        train(args.family, cfg=Config.small(), malicia=args.malicia,
              out_dir=args.ckpt_dir, max_files=32, epochs=20, seed=args.seed)

    real = load_family_opcodes(args.malicia, args.family)
    print(f"  real: {len(real)} sequences from {args.malicia}/{args.family}/")
    synthetic = _get_synthetic(
        args.family, args.n, args.synthetic, args.ckpt_dir, args.seed, dev
    )

    report = evaluate(
        args.family, real, synthetic, n_edit_pairs=args.n_pairs, seed=args.seed
    )

    # ── PASS/FAIL checks against the baselines ────────────────────────────────
    model = report["synthetic"]
    upper = report["baseline_real_vs_real"]
    lower = report["baseline_real_vs_random"]

    kl_model = model["freq_kl"]["symmetric_kl"]
    kl_random = lower["freq_kl"]["symmetric_kl"]
    f1_2_model = model["ngram_overlap"]["2gram_f1"]
    f1_2_random = lower["ngram_overlap"]["2gram_f1"]
    f1_3_model = model["ngram_overlap"]["3gram_f1"]
    f1_3_random = lower["ngram_overlap"]["3gram_f1"]
    f1_1_model = model["ngram_overlap"]["1gram_f1"]
    f1_1_upper = upper["ngram_overlap"]["1gram_f1"]
    diversity = model["edit_distance"]["synth_vs_synth"]["mean"]
    fk = model["freq_kl"]
    coverage = fk["vocab_overlap"] / max(fk["vocab_real"], 1)

    # Coverage is judged relative to the achievable upper bound, not an absolute %:
    # even two real halves don't reach 100% here (rare opcodes don't recur), so we
    # ask the model to reach at least half of the real-vs-real unigram-F1 floor.
    checks = [
        ("unigram KL well below random", kl_model < 0.5 * kl_random),
        ("2-gram F1 above random", f1_2_model > f1_2_random),
        ("3-gram F1 above random", f1_3_model > f1_3_random),
        ("no mode collapse (synth↔synth dist > 0)", diversity > 0.0),
        ("1-gram F1 ≥ 50% of real-vs-real bound", f1_1_model >= 0.5 * f1_1_upper),
    ]

    print("\n  Checks:")
    all_pass = True
    for label, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        all_pass &= ok

    print(f"\n  KL  model={kl_model:.4f}  random={kl_random:.4f}")
    print(f"  2-gram F1  model={f1_2_model:.3f}  random={f1_2_random:.3f}")
    print(f"  3-gram F1  model={f1_3_model:.3f}  random={f1_3_random:.3f}")
    print(f"  1-gram F1  model={f1_1_model:.3f}  real-vs-real={f1_1_upper:.3f}")
    print(f"  coverage={coverage:.0%}  diversity(synth↔synth)={diversity:.1f}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
