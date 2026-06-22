"""Stage 2 verification — run the forward noising process on a real malicia
family and print human-readable diagnostics.

    .venv/bin/python -m verify.verify_forward --family cleaman

What to look for:
  * Schedule sanity: ᾱ_0 = 1, ᾱ_T = 0, strictly monotone, matches (T-t)/T.
  * Empirical masked fraction (over non-PAD positions) ≈ t / T at t = T/4, T/2, 3T/4.
  * Endpoints exact: at t = 0 nothing is masked; at t = T every non-PAD is masked.
  * PAD positions never change to [MASK].
  * Watching one short sequence at t = 0, T/4, T/2, 3T/4, T: real opcodes
    *gradually* get replaced by [MASK] until the whole sequence is [MASK].
"""

from __future__ import annotations

import argparse

import torch

from d3pm.config import Config, device
from d3pm.data import OpcodeChunkedDataset, Vocabulary, load_family_opcodes
from d3pm.diffusion import AbsorbingDiffusion


def _decode_with_specials(ids: torch.Tensor, vocab: Vocabulary) -> str:
    """Render a token-id sequence including [MASK]/[PAD] (vocab.decode hides them).

    We need the special tokens here precisely *because* the whole point of the
    diagnostic is to watch [MASK] take over.
    """
    out: list[str] = []
    for i in ids.tolist():
        if i == vocab.mask_idx:
            out.append("[MASK]")
        elif i == vocab.pad_idx:
            out.append("[PAD]")
        else:
            # Vocabulary keeps its idx -> token list as a private attribute; using
            # it here is fine for a diagnostic script (no public method exposes
            # the mapping back to a string while keeping specials).
            out.append(vocab._idx2tok[i])
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicia", default="malicia")
    ap.add_argument("--family", default="cleaman")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=Config.small().max_len)
    ap.add_argument("--T", type=int, default=Config.small().T)
    ap.add_argument("--display-len", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    print(f"Device: {dev}")

    # ── load data + build the diffusion process ─────────────────────────────
    print(f"\nLoading family '{args.family}' from {args.malicia}/ ...")
    seqs = load_family_opcodes(args.malicia, args.family, max_files=args.max_files)
    vocab = Vocabulary.from_sequences(seqs)
    ds = OpcodeChunkedDataset(seqs, vocab, max_len=args.max_len)
    print(f"  vocab.size = {vocab.size}  mask_idx = {vocab.mask_idx}  pad_idx = {vocab.pad_idx}")
    print(f"  dataset    = {len(ds)} chunks of length {args.max_len}    T = {args.T}")

    diffusion = AbsorbingDiffusion(
        T=args.T, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx
    ).to(dev)

    # ── 1. schedule sanity ──────────────────────────────────────────────────
    ab = diffusion.alpha_bar
    expected = (args.T - torch.arange(args.T + 1, device=ab.device, dtype=ab.dtype)) / args.T
    schedule_ok = (
        tuple(ab.shape) == (args.T + 1,)
        and ab[0].item() == 1.0
        and ab[args.T].item() == 0.0
        and bool((ab[1:] < ab[:-1]).all().item())
        and torch.allclose(ab, expected)
    )
    print("\n[1] schedule (ᾱ_t)")
    print(f"  shape                  : {tuple(ab.shape)}   (expect ({args.T + 1},))")
    print(f"  ᾱ_0                    : {ab[0].item():.6f}  (expect 1.0)")
    print(f"  ᾱ_{args.T}                  : {ab[args.T].item():.6f}  (expect 0.0)")
    print(f"  strictly monotone dec. : {bool((ab[1:] < ab[:-1]).all().item())}")
    print(f"  matches (T-t)/T        : {bool(torch.allclose(ab, expected))}")

    # ── 2. empirical masked fraction on a real batch ────────────────────────
    B_big = min(64, len(ds))
    xs = torch.stack([ds[i]["tokens"] for i in range(B_big)]).to(dev)
    pad_mask = xs == vocab.pad_idx
    n_nonpad = int((~pad_mask).sum().item())
    n_pad = int(pad_mask.sum().item())

    print(f"\n[2] empirical masked fraction (B={B_big}, L={args.max_len}, "
          f"non-PAD={n_nonpad}, PAD={n_pad})")
    print(f"   {'t':>5}  {'expected t/T':>14}  {'observed':>10}  {'|diff|':>8}  "
          f"{'PAD intact?':>11}")
    frac_ok = True
    pad_ok = True
    for t_val in [0, args.T // 4, args.T // 2, 3 * args.T // 4, args.T]:
        t = torch.full((B_big,), t_val, dtype=torch.long, device=dev)
        xt = diffusion.q_sample(xs, t, pad_mask=pad_mask)

        pads_stayed = bool((xt[pad_mask] == vocab.pad_idx).all().item())
        non_pad_xt = xt[~pad_mask]
        frac = (non_pad_xt == vocab.mask_idx).float().mean().item()
        expected_frac = t_val / args.T
        diff = abs(frac - expected_frac)
        print(f"   {t_val:>5d}  {expected_frac:>14.4f}  {frac:>10.4f}  {diff:>8.4f}  "
              f"{str(pads_stayed):>11}")

        if not pads_stayed:
            pad_ok = False
        # Endpoints are deterministic, so they must be exact. Interior steps
        # are random — tolerate 0.03 (≈ 7× the standard error for this N).
        if t_val in (0, args.T):
            if diff != 0.0:
                frac_ok = False
        else:
            if diff > 0.03:
                frac_ok = False

    # ── 3. watch one sequence dissolve ──────────────────────────────────────
    # Pick a chunk that has no PAD so the visualization is uncluttered.
    no_pad_idx = next(
        (i for i in range(len(ds)) if not (ds[i]["tokens"] == vocab.pad_idx).any()),
        0,
    )
    x_one = ds[no_pad_idx]["tokens"][: args.display_len].unsqueeze(0).to(dev)
    one_pad_mask = x_one == vocab.pad_idx

    print(f"\n[3] one sequence at increasing t (chunk #{no_pad_idx}, "
          f"first {args.display_len} tokens)")
    for t_val in [0, args.T // 4, args.T // 2, 3 * args.T // 4, args.T]:
        # Reseed per t so reruns are reproducible AND the rows here are
        # independent draws (not a single chain across t).
        torch.manual_seed(args.seed + t_val)
        t = torch.tensor([t_val], dtype=torch.long, device=dev)
        xt = diffusion.q_sample(x_one, t, pad_mask=one_pad_mask)
        print(f"   t={t_val:>4d}: {_decode_with_specials(xt[0].cpu(), vocab)}")

    print(
        "\nVERDICT:",
        "PASS ✅" if (schedule_ok and frac_ok and pad_ok) else "FAIL ❌",
    )


if __name__ == "__main__":
    main()
