"""Stage 4 verification — sanity-check the hybrid loss, then run the canonical
correctness test for any diffusion model: OVERFIT ONE TINY BATCH and watch the
loss collapse toward zero.

    .venv/bin/python -m verify.verify_loss --family cleaman

If the loss, gradients, and optimization are all wired correctly, a model with
enough capacity can memorize a single small batch: the denoiser learns to map
every partially-masked view of those few sequences back to the originals, so the
masked-position cross-entropy → ~0. If anything is mis-wired (wrong masking,
gather on the wrong axis, scoring the wrong positions, a detached graph) the loss
will plateau well above zero.

What to look for:
  * Initial loss is finite; the *unweighted* `ce` diagnostic starts near
    ln(vocab) ≈ ln(198) ≈ 5.3 (a fresh model guessing ~uniformly). The total
    `loss` is much smaller because `vb` is (1/t)-reweighted and normalized by the
    non-PAD token count — that is expected, the signal here is the *drop*.
  * backward() fills gradients (reported non-None / non-zero count).
  * Over a few hundred steps on one batch the loss falls by orders of magnitude
    toward ~0. (It need not hit exactly 0 — masked tokens at large t with little
    context are genuinely ambiguous — but it should drop dramatically.)
"""

from __future__ import annotations

import argparse

import torch

from d3pm.config import Config, device
from d3pm.data import OpcodeChunkedDataset, Vocabulary, load_family_opcodes
from d3pm.denoiser import TransformerDenoiser
from d3pm.diffusion import AbsorbingDiffusion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicia", default="malicia")
    ap.add_argument("--family", default="cleaman")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    cfg = Config.small()
    print(f"Device: {dev}")

    # ── data: grab one tiny batch ───────────────────────────────────────────
    seqs = load_family_opcodes(args.malicia, args.family, max_files=args.max_files)
    vocab = Vocabulary.from_sequences(seqs)
    ds = OpcodeChunkedDataset(seqs, vocab, max_len=cfg.max_len)
    B = min(args.batch, len(ds))
    x0 = torch.stack([ds[i]["tokens"] for i in range(B)]).to(dev)
    pad_mask = x0 == vocab.pad_idx
    print(f"  family={args.family}  vocab={vocab.size}  overfit batch = {B}×{cfg.max_len}  "
          f"T={cfg.T}")

    diffusion = AbsorbingDiffusion(cfg.T, vocab.mask_idx, vocab.pad_idx).to(dev)
    model = TransformerDenoiser(
        vocab_size=vocab.size, max_len=cfg.max_len, pad_idx=vocab.pad_idx,
        d_model=cfg.d_model, nhead=cfg.nhead, layers=cfg.layers,
        dim_ff=cfg.dim_ff, dropout=0.0,   # no dropout: we WANT to overfit
    ).to(dev)

    # ── 1. one forward/backward: finite loss + gradients fill ───────────────
    model.train()
    out = diffusion.loss(model, x0, pad_mask, lambda_ce=cfg.lambda_ce)
    out["loss"].backward()
    n_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_params = sum(1 for _ in model.parameters())
    print("\n[1] single step")
    print(f"  loss={out['loss'].item():.4f}  vb={out['vb'].item():.4f}  "
          f"ce={out['ce'].item():.4f}  (finite: {bool(torch.isfinite(out['loss']))})")
    print(f"  params receiving non-zero grad: {n_grad}/{n_params}")
    print(f"  ln(vocab) reference (uniform guess) ≈ {torch.log(torch.tensor(float(vocab.size))).item():.3f}")
    model.zero_grad()

    # ── 2. overfit the single batch ─────────────────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"\n[2] overfitting one batch for {args.steps} steps (lr={args.lr}) ...")
    first = None
    last = None
    for step in range(args.steps):
        opt.zero_grad()
        out = diffusion.loss(model, x0, pad_mask, lambda_ce=cfg.lambda_ce)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = out["loss"].item()
        last = out["loss"].item()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"   step {step:>4d}  loss={out['loss'].item():.5f}  "
                  f"vb={out['vb'].item():.5f}  ce={out['ce'].item():.5f}")

    # The loss is stochastic (a fresh random t each step), so compare a short
    # running average at the end against the start rather than a single value.
    model.eval()
    with torch.no_grad():
        torch.manual_seed(args.seed + 1)
        tail = torch.tensor(
            [diffusion.loss(model, x0, pad_mask, lambda_ce=cfg.lambda_ce)["loss"].item()
             for _ in range(20)]
        ).mean().item()
    print(f"\n  first-step loss : {first:.5f}")
    print(f"  final 20-step avg: {tail:.5f}")
    dropped = tail < first * 0.25
    print(f"  dropped to <25% of start : {dropped}")

    ok = bool(torch.isfinite(out["loss"])) and n_grad > 0 and dropped
    print("\nVERDICT:", "PASS ✅" if ok else "FAIL ❌")


if __name__ == "__main__":
    main()
