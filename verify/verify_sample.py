"""Stage 6 verification — sample from a trained checkpoint and prove the reverse
process produces clean, plausible opcode sequences:

    .venv/bin/python -m verify.verify_sample --family cleaman

What it checks
--------------
  1. CLEAN OUTPUT. After a full reverse chain (all-[MASK] at t=T → t=1) no
     generated token id is [MASK] or [PAD]. This is the core correctness claim:
     the absorbing reverse posterior un-masks every position by the final step
     (prob 1/t = 1 at t=1) and we forbid the model from ever proposing MASK/PAD.

  2. PLAUSIBLE LENGTH. We generate at the model's max_len; since decode only drops
     MASK/PAD (and there are none), every sequence should be exactly seq_len long.

  3. REALISTIC OPCODE MIX. The synthetic unigram distribution should resemble the
     real family's: the workhorse opcodes (mov/push/call/...) dominate both. We
     print the top opcodes side by side and check there is meaningful overlap in
     the top-10. A model that hadn't learned anything would produce a near-uniform
     mix instead.

If no checkpoint exists yet, pass --train to fit a quick cleaman one first.
A couple of decoded samples are printed so you can eyeball them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from d3pm.config import Config, device
from d3pm.data import load_family_opcodes
from d3pm.generate import generate
from d3pm.train import load_checkpoint, train


def _top_counter(seqs: list[list[str]], k: int = 10) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for s in seqs:
        c.update(s)
    return c.most_common(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicia", default="malicia")
    ap.add_argument("--family", default="cleaman")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out-dir", default="synthetic")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true",
                    help="train a quick checkpoint first (if none exists)")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    dev = device()

    # ── 0. ensure a checkpoint exists ────────────────────────────────────────
    ckpt_file = Path(args.ckpt_dir) / args.family / "d3pm.pt"
    if args.train or not ckpt_file.exists():
        print(f"[0] training a quick {args.family} checkpoint ({args.epochs} epochs)…")
        cfg = Config.small()
        cfg.T = 250
        cfg.max_len = 128
        train(
            args.family,
            malicia_dir=args.malicia,
            out_dir=args.ckpt_dir,
            max_files=50,
            epochs=args.epochs,
            batch_size=16,
            cfg=cfg,
            seed=args.seed,
        )
    else:
        print(f"[0] reusing existing checkpoint at {ckpt_file}")

    # ── 1. sample ────────────────────────────────────────────────────────────
    print("\n[1] sampling")
    synth = generate(
        args.family,
        n=args.n,
        seq_len=args.seq_len,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        dev=dev,
    )

    # ── 2. clean-output check: re-sample raw ids to inspect MASK/PAD directly ─
    # generate() already drops MASK/PAD in decode; to *prove* none survived the
    # reverse chain we re-run sample() and inspect the raw id tensor.
    model, diffusion, vocab, meta = load_checkpoint(args.family, out_dir=args.ckpt_dir, dev=dev)
    seq_len = args.seq_len or meta["model_config"]["max_len"]
    torch.manual_seed(args.seed + 1)
    x0 = diffusion.sample(model, n=args.n, seq_len=seq_len, device=dev)
    n_mask = (x0 == vocab.mask_idx).sum().item()
    n_pad = (x0 == vocab.pad_idx).sum().item()
    clean = n_mask == 0 and n_pad == 0
    print("\n[2] clean output (raw ids)")
    print(f"  generated shape : {tuple(x0.shape)}")
    print(f"  [MASK] ids left : {n_mask}")
    print(f"  [PAD]  ids left : {n_pad}")
    print(f"  all clean       : {clean}")

    # ── 3. length plausibility ───────────────────────────────────────────────
    lengths = [len(s) for s in synth]
    len_ok = all(l == seq_len for l in lengths)
    print("\n[3] sequence lengths")
    print(f"  expected (=seq_len): {seq_len}")
    print(f"  min/avg/max        : {min(lengths)}/{sum(lengths) / len(lengths):.0f}/{max(lengths)}")
    print(f"  all == seq_len     : {len_ok}")

    # ── 4. opcode mix vs. real ───────────────────────────────────────────────
    real = load_family_opcodes(args.malicia, args.family)
    real_top = _top_counter(real, 10)
    synth_top = _top_counter(synth, 10)
    real_set = {op for op, _ in real_top}
    synth_set = {op for op, _ in synth_top}
    overlap = real_set & synth_set
    overlap_ok = len(overlap) >= 5  # at least half the top-10 shared

    print("\n[4] top-10 opcodes (real vs. synthetic)")
    print(f"  {'real':<20}{'synthetic':<20}")
    for i in range(10):
        r = f"{real_top[i][0]} ({real_top[i][1]})" if i < len(real_top) else ""
        s = f"{synth_top[i][0]} ({synth_top[i][1]})" if i < len(synth_top) else ""
        print(f"  {r:<20}{s:<20}")
    print(f"  top-10 overlap     : {len(overlap)}/10  -> {sorted(overlap)}")
    print(f"  overlap >= 5       : {overlap_ok}")

    # ── 5. eyeball a couple of samples ───────────────────────────────────────
    print("\n[5] sample previews (first 30 opcodes)")
    for i in range(min(2, len(synth))):
        preview = " ".join(synth[i][:30])
        print(f"  seq_{i}: {preview} …")

    ok = clean and len_ok and overlap_ok
    print("\nVERDICT:", "PASS ✅" if ok else "FAIL ❌")


if __name__ == "__main__":
    main()
