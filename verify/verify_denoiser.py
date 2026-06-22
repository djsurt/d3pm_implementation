"""Stage 3 verification — build the TransformerDenoiser, run a real noised
malicia batch through it, and check the three things that must hold before we
can train it:

    .venv/bin/python -m verify.verify_denoiser --family cleaman

What to look for:
  * Output shape is (B, L, vocab_size) and every logit is finite (no NaN/Inf).
  * Parameter count is in the expected ballpark (~2M at the default config,
    much smaller at the `small` preset used here).
  * Time conditioning actually works: feeding the SAME x_t with DIFFERENT t
    produces DIFFERENT logits. If t were ignored the outputs would be identical.
  * PAD invariance sanity: changing tokens *after* a PAD boundary does not change
    the logits at the real (non-PAD) positions (PAD is masked out of attention).
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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    cfg = Config.small()
    print(f"Device: {dev}")
    print(f"Config (small): T={cfg.T} max_len={cfg.max_len} d_model={cfg.d_model} "
          f"nhead={cfg.nhead} layers={cfg.layers} dim_ff={cfg.dim_ff}")

    # ── load data + build a noised batch x_t ────────────────────────────────
    print(f"\nLoading family '{args.family}' from {args.malicia}/ ...")
    seqs = load_family_opcodes(args.malicia, args.family, max_files=args.max_files)
    vocab = Vocabulary.from_sequences(seqs)
    ds = OpcodeChunkedDataset(seqs, vocab, max_len=cfg.max_len)
    B = min(args.batch, len(ds))
    x0 = torch.stack([ds[i]["tokens"] for i in range(B)]).to(dev)
    pad_mask = x0 == vocab.pad_idx
    print(f"  vocab.size = {vocab.size}  dataset = {len(ds)} chunks  batch = {B}")

    diffusion = AbsorbingDiffusion(cfg.T, vocab.mask_idx, vocab.pad_idx).to(dev)
    t = torch.randint(1, cfg.T + 1, (B,), device=dev)
    x_t = diffusion.q_sample(x0, t, pad_mask=pad_mask)

    model = TransformerDenoiser(
        vocab_size=vocab.size,
        max_len=cfg.max_len,
        pad_idx=vocab.pad_idx,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        layers=cfg.layers,
        dim_ff=cfg.dim_ff,
    ).to(dev)
    model.eval()  # disable dropout so the time-conditioning check is deterministic

    # ── 1. shape + finiteness ───────────────────────────────────────────────
    with torch.no_grad():
        logits = model(x_t, t)
    shape_ok = tuple(logits.shape) == (B, cfg.max_len, vocab.size)
    finite_ok = bool(torch.isfinite(logits).all().item())
    print("\n[1] forward pass")
    print(f"  output shape : {tuple(logits.shape)}  (expect ({B}, {cfg.max_len}, {vocab.size}))")
    print(f"  all finite   : {finite_ok}")

    # ── 2. parameter count ──────────────────────────────────────────────────
    n_params = model.num_parameters()
    # The `small` preset is intentionally tiny; the ~2M figure in the plan is for
    # the default config. We just sanity-check the count is positive & printed,
    # and report the default-config size too for the plan's stated expectation.
    default = Config.default()
    big = TransformerDenoiser(
        vocab_size=vocab.size, max_len=default.max_len, pad_idx=vocab.pad_idx,
        d_model=default.d_model, nhead=default.nhead, layers=default.layers,
        dim_ff=default.dim_ff,
    )
    n_params_default = big.num_parameters()
    params_ok = n_params > 0
    print("\n[2] parameter count")
    print(f"  small preset   : {n_params:,}")
    print(f"  default preset : {n_params_default:,}  (plan expects ~2M)")

    # ── 3. time conditioning changes the output ─────────────────────────────
    # Same noised input, two different timesteps. If t is truly wired in, the
    # logits must differ. We compare t and a clearly different t' for each row.
    with torch.no_grad():
        t_a = torch.full((B,), 1, device=dev)
        t_b = torch.full((B,), cfg.T, device=dev)
        out_a = model(x_t, t_a)
        out_b = model(x_t, t_b)
        # also confirm SAME t gives identical output (determinism in eval mode)
        out_a2 = model(x_t, t_a)
    max_diff_tt = (out_a - out_b).abs().max().item()
    max_diff_same = (out_a - out_a2).abs().max().item()
    time_ok = max_diff_tt > 1e-4 and max_diff_same < 1e-5
    print("\n[3] time conditioning")
    print(f"  max|logits(t=1) - logits(t=T)| : {max_diff_tt:.6f}  (expect > 0)")
    print(f"  max|logits(t=1) - logits(t=1)| : {max_diff_same:.6e}  (expect ~0)")

    # ── 4. PAD positions don't leak into real ones ──────────────────────────
    # Trailing PAD must not change the logits at real positions. Take the first
    # `r` real tokens of the batch, run them unpadded, then run them padded out
    # to max_len, and confirm the real-position logits match. (PAD is zeroed in
    # the embedding and masked out of attention, so it cannot influence reals.)
    r = max(1, cfg.max_len - 6)
    real_tokens = x_t[:, :r].clone()
    # ensure the "real" slice has no incidental PAD so the comparison is clean
    real_tokens[real_tokens == vocab.pad_idx] = vocab.mask_idx
    padded = torch.full((B, cfg.max_len), vocab.pad_idx, device=dev)
    padded[:, :r] = real_tokens
    with torch.no_grad():
        out_short = model(real_tokens, t)
        out_long = model(padded, t)
    diff_real = (out_short - out_long[:, :r]).abs().max().item()
    pad_ok = diff_real < 1e-4
    print("\n[4] PAD isolation (padded vs unpadded)")
    print(f"  max change at real positions when PAD appended : {diff_real:.6e} "
          f"(expect ~0)")

    ok = shape_ok and finite_ok and params_ok and time_ok and pad_ok
    print("\nVERDICT:", "PASS ✅" if ok else "FAIL ❌")


if __name__ == "__main__":
    main()
