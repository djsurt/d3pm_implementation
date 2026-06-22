"""
Verify the memorization/novelty audit + transition-matrix KL (d3pm/seq_audit.py).

Run (main torch venv):

    .venv/bin/python -m verify.verify_seq_audit
    .venv/bin/python -m verify.verify_seq_audit --family zbot

Uses run_01's generated sequences by default (the only sizeable synthetic set on
disk). EXPECTED OUTPUT (winwebsec, run_01):
  * NOVELTY: exact-copy rate ~0, mean Jaccard gap strongly NEGATIVE (synth more
    novel than a genuine held-out sample), verdict = NO memorization.
  * TRANSITION KL: synthetic symmetric KL sits well below the real-vs-random ceiling
    and above the real-vs-real floor — i.e. structure learned, far from random.
All CHECK lines should print PASS.
"""

from __future__ import annotations

import argparse

from d3pm.data import load_family_opcodes
from d3pm.seq_audit import audit
from d3pm.seq_eval import load_synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="winwebsec")
    ap.add_argument("--malicia", default="malicia")
    ap.add_argument("--synthetic", default="runs/run_01_baseline/synthetic")
    args = ap.parse_args()

    real = load_family_opcodes(args.malicia, args.family)
    synthetic = load_synthetic(args.synthetic, args.family)
    report = audit(args.family, real, synthetic)

    nov = report["novelty"]
    tk = report["transition_kl"]
    syn_kl = tk["synthetic"]["transition_symmetric_kl"]
    floor = tk["baseline_real_vs_real"]["transition_symmetric_kl"]
    ceil = tk["baseline_real_vs_random"]["transition_symmetric_kl"]

    checks = [
        ("audit reports NO memorization", not nov["memorizing"]),
        (
            "synth not closer to train than genuine sample (gap <= 0.1)",
            nov["mean_jaccard_gap"] <= 0.1,
        ),
        (
            "exact-copy rate not above real-holdout baseline",
            nov["synthetic_vs_train"]["exact_copy_rate"]
            <= nov["real_holdout_vs_train"]["exact_copy_rate"],
        ),
        ("transition KL far below random ceiling (< ceil/5)", syn_kl < ceil / 5),
        ("transition KL above real-vs-real floor", syn_kl > floor),
        ("transition KL closer to floor than ceiling", abs(syn_kl - floor) < abs(syn_kl - ceil)),
    ]
    print("\nChecks:")
    allpass = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allpass = allpass and ok
    print(f"\n{'ALL CHECKS PASS' if allpass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
