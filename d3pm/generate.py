"""
Stage 6 — generation CLI: turn a trained checkpoint into synthetic opcode files.

Where this fits
---------------
Stage 5 trained the denoiser and saved a checkpoint; Stage 6 added the reverse
(sampling) process to ``AbsorbingDiffusion.sample`` (see ``diffusion.py``). This
file is the thin command-line wrapper that ties them together:

  1. ``load_checkpoint(family)`` rebuilds the model + diffusion + vocab and puts
     the model in ``.eval()`` on the compute device (from ``train.py``).
  2. ``diffusion.sample(model, n, seq_len)`` runs the all-``[MASK]`` → clean
     reverse chain and returns ``(n, seq_len)`` token ids.
  3. ``vocab.decode`` maps each row back to opcode strings (dropping any MASK/PAD,
     though after a full reverse run there should be none) and we write one file
     per sequence under ``synthetic/<family>/seq_*.txt``, one opcode per line —
     the exact same on-disk format as the real ``malicia/<family>/*.asm.txt``.

Run it:

    .venv/bin/python -m d3pm.generate --family cleaman --n 50 --seq-len 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from d3pm.config import device
from d3pm.train import load_checkpoint


def generate(
    family: str,
    *,
    n: int = 50,
    seq_len: int | None = None,
    out_dir: str = "synthetic",
    ckpt_dir: str = "checkpoints",
    seed: int | None = None,
    dev: torch.device | None = None,
    verbose: bool = True,
) -> list[list[str]]:
    """Generate ``n`` synthetic opcode sequences for ``family`` and write them out.

    Returns the decoded sequences (list of opcode-string lists). ``seq_len``
    defaults to the model's training chunk length (``max_len``), which is the
    length the denoiser saw during training and is capped by its positional table.
    """
    dev = dev or device()
    if seed is not None:
        torch.manual_seed(seed)

    model, diffusion, vocab, meta = load_checkpoint(family, out_dir=ckpt_dir, dev=dev)

    # Default to the model's trained chunk length (its positional table caps this).
    max_len = meta["model_config"]["max_len"]
    seq_len = seq_len or max_len
    if seq_len > max_len:
        raise ValueError(
            f"seq_len={seq_len} exceeds the model's max_len={max_len}; "
            "the positional table cannot index longer sequences"
        )

    if verbose:
        print(f"Device: {dev}")
        print(f"  family={family}  n={n}  seq_len={seq_len}  T={diffusion.T}  "
              f"vocab={vocab.size}")

    # The full reverse chain: all-[MASK] at t=T → clean opcodes at t=1.
    x0 = diffusion.sample(model, n=n, seq_len=seq_len, device=dev)   # (n, seq_len)
    sequences = [vocab.decode(row.cpu()) for row in x0]

    # Write one file per sequence, matching the real data's one-opcode-per-line form.
    family_dir = Path(out_dir) / family
    family_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(n - 1)) if n > 1 else 1
    for i, ops in enumerate(sequences):
        (family_dir / f"seq_{i:0{width}d}.txt").write_text("\n".join(ops) + "\n")

    if verbose:
        lengths = [len(s) for s in sequences]
        avg = sum(lengths) / max(1, len(lengths))
        print(f"  wrote {len(sequences)} files → {family_dir}/  "
              f"(lengths min/avg/max = {min(lengths)}/{avg:.0f}/{max(lengths)})")

    return sequences


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate synthetic opcode sequences from a trained D3PM checkpoint."
    )
    ap.add_argument("--family", required=True, help="family whose checkpoint to sample")
    ap.add_argument("--n", type=int, default=50, help="number of sequences to generate")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="length of each sequence (default: model's trained max_len)")
    ap.add_argument("--out-dir", default="synthetic", help="where to write synthetic/<family>/")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="root dir of checkpoints/<family>/")
    ap.add_argument("--seed", type=int, default=None, help="optional RNG seed for reproducibility")
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()
    generate(
        args.family,
        n=args.n,
        seq_len=args.seq_len,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
