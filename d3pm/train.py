"""
Stage 5 — the training loop: turn the pieces from Stages 1–4 into a trained model.

Where this fits
---------------
Everything needed to *learn* now exists and is verified:

  * Stage 1 (`data.py`)      — read a family's opcode files, build a `Vocabulary`,
                               chunk them into fixed-length integer tensors.
  * Stage 2/4 (`diffusion.py`) — `AbsorbingDiffusion`: the forward noising process
                               and the hybrid training loss.
  * Stage 3 (`denoiser.py`)  — `TransformerDenoiser`: the network that predicts the
                               clean tokens (the only thing with parameters to train).

This file wires them together and runs the optimization that actually fits the
denoiser's weights. The output is a checkpoint we can reload in Stage 6 to sample
new opcode sequences.

The optimization loop, conceptually
------------------------------------
Training a neural net is a loop over the data, repeated for several *epochs* (one
epoch = one full pass over the dataset). For each mini-batch we:

  1. ``optimizer.zero_grad()`` — clear the gradients left over from the previous
     step. PyTorch *accumulates* gradients into ``param.grad`` by default, so we
     must reset them or steps would add together.
  2. ``out = diffusion.loss(model, tokens, pad_mask)`` — a forward pass: noise the
     batch, predict clean tokens, compute the scalar loss. This also records the
     computation graph autograd needs.
  3. ``out["loss"].backward()`` — backpropagation: walk that graph backwards and
     fill every ``param.grad`` with ∂loss/∂param.
  4. ``clip_grad_norm_(..., 1.0)`` — *gradient clipping*: if the gradient vector is
     longer than 1.0, scale it down. This caps the occasional huge gradient that
     would otherwise throw the weights off; standard for training transformers.
  5. ``optimizer.step()`` — the Adam optimizer nudges every weight a little in the
     direction that reduces the loss, using the gradients from step 3.
  6. ``scheduler.step()`` — shrink the learning rate slightly (cosine schedule), so
     we take big steps early and fine, careful steps near the end.

We track the mean loss per epoch; on a healthy run it falls steadily.

The checkpoint
--------------
We save three files under ``checkpoints/<family>/``:

  * ``d3pm.pt``    — the trained ``model.state_dict()`` (every weight tensor) plus
                     the config needed to *rebuild the exact same architecture* and
                     diffusion schedule at generation time. A ``state_dict`` is just
                     an ordered dict {parameter name -> tensor}; ``torch.save`` /
                     ``torch.load`` serialize it.
  * ``vocab.pkl``  — the `Vocabulary`, so generation decodes ids with the *same*
                     opcode<->id mapping we trained on (`Vocabulary.save`).
  * ``losses.json``— the per-epoch loss history, for plotting / sanity checks.

``load_checkpoint`` below reverses all of this: it reads the configs, rebuilds the
model + diffusion, loads the weights, and returns everything ready to sample. Stage
6's `generate.py` will call it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import copy

import torch
from torch.utils.data import DataLoader, random_split

from d3pm.config import Config, device
from d3pm.data import OpcodeChunkedDataset, Vocabulary, load_family_opcodes
from d3pm.denoiser import TransformerDenoiser
from d3pm.diffusion import AbsorbingDiffusion


# ── new PyTorch concepts used in this file ───────────────────────────────────
#
# torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
#       Wraps a `Dataset` (our `OpcodeChunkedDataset`) and hands out *batches*.
#       Iterating it yields, each step, a dict whose tensors have a new leading
#       batch axis: `tokens` becomes (batch_size, max_len). With `shuffle=True`
#       it reshuffles the order every epoch (so the model doesn't memorize an
#       order). Because every item is already the same length, the default
#       "stack them" collation just works — no custom collate function needed.
#
# torch.optim.Adam(params, lr)
#       The optimizer. It holds the parameters and, on `.step()`, updates each
#       using its gradient plus per-parameter adaptive moments (Adam). `lr` is
#       the base learning rate (step size).
#
# torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max)
#       Adjusts the optimizer's learning rate over time following a cosine curve
#       from the base lr down to ~0 across `T_max` scheduler steps. We step it
#       once per batch, so T_max = total number of batches across all epochs.
#
# torch.nn.utils.clip_grad_norm_(params, max_norm)
#       Rescales gradients in-place so their global L2 norm is at most `max_norm`.
#
# model.train() / model.eval()
#       Toggle training-mode behavior (dropout active vs. disabled, etc.). We
#       train in `.train()`; verification reloads and compares in `.eval()`.


class _EMA:
    """Exponential moving average of model weights.

    Diffusion models sample better from a slowly-averaged copy of the weights
    than from the raw (still-jittering) training weights. We keep a *shadow* copy
    that, after each optimizer step, is nudged toward the live weights:
    ``shadow = decay·shadow + (1−decay)·live``. Float tensors are averaged;
    integer buffers (if any) are copied verbatim. At save time we hand the shadow
    to the checkpoint so generation uses the smoothed weights transparently.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> dict:
        return self.shadow


@torch.no_grad()
def _mean_loss(diffusion, model, loader, dev, lambda_ce) -> float:
    """Average loss over a loader without updating weights (held-out val signal).

    The loss samples a random t and masking each call, so this is a noisy but
    unbiased estimate of the same objective the training loop minimizes — good
    enough as an early-stop trend. We toggle ``eval()`` so dropout is off, then
    restore ``train()``.
    """
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        tokens = batch["tokens"].to(dev)
        pad_mask = batch["pad_mask"].to(dev)
        out = diffusion.loss(model, tokens, pad_mask, lambda_ce=lambda_ce)
        total += out["loss"].item() * tokens.shape[0]
        n += tokens.shape[0]
    if was_training:
        model.train()
    return total / max(1, n)


def train(
    family: str,
    *,
    malicia_dir: str = "malicia",
    out_dir: str = "checkpoints",
    max_files: int | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    cfg: Config | None = None,
    lr: float = 1e-3,
    min_chunk: int = 32,
    seed: int = 0,
    ema_decay: float | None = None,
    val_frac: float = 0.0,
    patience: int = 0,
    min_epochs: int = 0,
    warmup_frac: float = 0.0,
    dev: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train an absorbing-state D3PM on one malware family and save a checkpoint.

    Builds the vocab + dataset + dataloader + model + diffusion, runs the
    optimization loop, and writes ``checkpoints/<family>/{d3pm.pt, vocab.pkl,
    losses.json}``. Returns a small summary dict (also handy for tests).

    Args mirror the CLI flags; `cfg` carries the architecture/diffusion sizes
    (defaults to `Config.default()`), so `T`/`max_len`/`d_model`/... live there.

    Optional run_02 knobs:
      ema_decay  if set (e.g. 0.999), keep an EMA of the weights and save *that*
                 as the checkpoint (generation samples from the smoothed weights).
      val_frac   fraction of chunks held out to report a per-epoch val loss.
      patience   if >0 (and val_frac>0), stop early when val loss hasn't improved
                 for `patience` consecutive epochs.
      min_epochs floor for early stop: never stop before this many epochs even if
                 patience is exhausted. Guards against killing a family during its
                 initial loss plateau — zeroaccess (mov-heavy, large vocab) sits on
                 a flat basin for tens of epochs before the loss breaks through, and
                 plain patience fired mid-plateau in run_02.
      warmup_frac fraction of total steps spent linearly ramping the LR from ~0 up
                 to `lr` before cosine annealing begins. A cold start at the full
                 lr=1e-3 was enough to strand the scale-up model on that flat basin;
                 a short warmup lets it find the descent direction first.
    """
    cfg = cfg or Config.default()
    dev = dev or device()
    torch.manual_seed(seed)

    # ── 1. data → vocab → chunked dataset → (train/val) dataloaders ───────────
    sequences = load_family_opcodes(malicia_dir, family, max_files=max_files)
    if not sequences:
        raise ValueError(f"no opcode files found for family '{family}' under {malicia_dir}")
    vocab = Vocabulary.from_sequences(sequences)
    dataset = OpcodeChunkedDataset(sequences, vocab, max_len=cfg.max_len, min_chunk=min_chunk)
    if len(dataset) == 0:
        raise ValueError(
            f"family '{family}' produced 0 chunks (every file shorter than "
            f"min_chunk={min_chunk}); lower --max-len or --min-chunk"
        )

    # Hold out a val split only if it leaves a non-empty train set (tiny smoke
    # datasets fall back to train-only). `random_split` is seeded for repeatable
    # partitions.
    n_val = int(len(dataset) * val_frac)
    if val_frac > 0 and 0 < n_val < len(dataset):
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=gen)
    else:
        train_ds, val_ds = dataset, None

    # `drop_last=True` keeps every batch the same size (a ragged final batch is
    # fine for the math but makes step counts/averages cleaner); if the dataset
    # is smaller than one batch we keep the single short batch instead.
    drop_last = len(train_ds) > batch_size
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
    )

    # ── 2. model + diffusion, both moved onto the compute device ─────────────
    diffusion = AbsorbingDiffusion(cfg.T, vocab.mask_idx, vocab.pad_idx).to(dev)
    model = TransformerDenoiser(
        vocab_size=vocab.size,
        max_len=cfg.max_len,
        pad_idx=vocab.pad_idx,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        layers=cfg.layers,
        dim_ff=cfg.dim_ff,
    ).to(dev)

    # ── 3. optimizer + cosine LR schedule + (optional) EMA ───────────────────
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    # Optional linear LR warmup, then cosine anneal over the remaining steps.
    # `SequentialLR` chains the two and switches at the warmup milestone. With
    # warmup_frac=0 this collapses to the original plain cosine schedule.
    warmup_steps = int(warmup_frac * total_steps)
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - warmup_steps)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    ema = _EMA(model, ema_decay) if ema_decay else None

    if verbose:
        print(f"Device: {dev}")
        print(f"  family={family}  files={len(sequences)}  chunks={len(dataset)}  "
              f"vocab={vocab.size}")
        print(f"  train/val chunks={len(train_ds)}/{len(val_ds) if val_ds else 0}  "
              f"batches/epoch={steps_per_epoch}  epochs={epochs}  batch={batch_size}  "
              f"T={cfg.T}  max_len={cfg.max_len}")
        print(f"  model params={model.num_parameters():,}  base_lr={lr}  "
              f"warmup={warmup_steps or 'off'}  ema={ema_decay or 'off'}  "
              f"patience={patience or 'off'}  min_epochs={min_epochs or 'off'}")

    # ── 4. the optimization loop ─────────────────────────────────────────────
    model.train()
    epoch_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_ema_state: dict | None = None
    epochs_since_improve = 0
    for epoch in range(epochs):
        running, n = 0.0, 0
        for batch in loader:
            tokens = batch["tokens"].to(dev)        # (B, max_len) ids
            pad_mask = batch["pad_mask"].to(dev)    # (B, max_len) True at PAD

            opt.zero_grad()
            out = diffusion.loss(model, tokens, pad_mask, lambda_ce=cfg.lambda_ce)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            running += out["loss"].item() * tokens.shape[0]
            n += tokens.shape[0]

        mean_loss = running / max(1, n)
        epoch_losses.append(mean_loss)

        val_loss = _mean_loss(diffusion, model, val_loader, dev, cfg.lambda_ce) \
            if val_loader is not None else None
        if val_loss is not None:
            val_losses.append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_since_improve = 0
                # snapshot the weights we'd actually deploy at this best point
                best_ema_state = copy.deepcopy(ema.state_dict() if ema else model.state_dict())
            else:
                epochs_since_improve += 1

        if verbose:
            cur_lr = scheduler.get_last_lr()[0]
            vmsg = f"  val_loss={val_loss:.5f}" if val_loss is not None else ""
            print(f"  epoch {epoch + 1:>3d}/{epochs}  mean_loss={mean_loss:.5f}{vmsg}  "
                  f"lr={cur_lr:.2e}")

        if (patience and val_loader is not None and epochs_since_improve >= patience
                and (epoch + 1) >= min_epochs):
            if verbose:
                print(f"  early stop: val loss flat for {patience} epochs "
                      f"(best={best_val:.5f}); restoring best weights")
            break

    # ── 5. save the checkpoint ───────────────────────────────────────────────
    family_dir = Path(out_dir) / family
    family_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "vocab_size": vocab.size,
        "max_len": cfg.max_len,
        "pad_idx": vocab.pad_idx,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "layers": cfg.layers,
        "dim_ff": cfg.dim_ff,
    }
    diffusion_config = {"T": cfg.T, "mask_idx": vocab.mask_idx, "pad_idx": vocab.pad_idx}

    # Which weights to ship: the best-val snapshot if we tracked one, else the
    # final EMA (if enabled), else the raw final weights. Generation loads this
    # via `load_checkpoint` → it samples from whatever we save here.
    if best_ema_state is not None:
        save_state = best_ema_state
    elif ema is not None:
        save_state = ema.state_dict()
    else:
        save_state = model.state_dict()

    ckpt_path = family_dir / "d3pm.pt"
    torch.save(
        {
            "model_state": save_state,
            "model_config": model_config,
            "diffusion_config": diffusion_config,
            "lambda_ce": cfg.lambda_ce,
            "ema_decay": ema_decay,
        },
        ckpt_path,
    )
    vocab.save(family_dir / "vocab.pkl")
    (family_dir / "losses.json").write_text(
        json.dumps(
            {
                "family": family,
                "epochs": epochs,
                "epochs_run": len(epoch_losses),
                "batch_size": batch_size,
                "lr": lr,
                "ema_decay": ema_decay,
                "best_val_loss": best_val if val_losses else None,
                "epoch_losses": epoch_losses,
                "val_losses": val_losses,
            },
            indent=2,
        )
    )

    if verbose:
        print(f"\nSaved checkpoint → {family_dir}/ (d3pm.pt, vocab.pkl, losses.json)")

    return {
        "checkpoint_dir": str(family_dir),
        "epoch_losses": epoch_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val if val_losses else None,
        "vocab_size": vocab.size,
        "num_chunks": len(dataset),
    }


def load_checkpoint(
    family: str,
    *,
    out_dir: str = "checkpoints",
    dev: torch.device | None = None,
) -> tuple[TransformerDenoiser, AbsorbingDiffusion, Vocabulary, dict]:
    """Reload a trained checkpoint: rebuild the model + diffusion and load weights.

    The inverse of the save block in `train`. Reconstructs `TransformerDenoiser`
    and `AbsorbingDiffusion` from the stored configs (so the architecture matches
    exactly), loads the saved weights, moves everything onto `dev`, and puts the
    model in eval mode. Stage 6's sampler builds on this.

    Returns ``(model, diffusion, vocab, meta)`` where `meta` is the saved dict
    (configs + lambda_ce).
    """
    dev = dev or device()
    family_dir = Path(out_dir) / family
    ckpt = torch.load(family_dir / "d3pm.pt", map_location=dev)
    vocab = Vocabulary.load(family_dir / "vocab.pkl")

    mc = ckpt["model_config"]
    model = TransformerDenoiser(
        vocab_size=mc["vocab_size"],
        max_len=mc["max_len"],
        pad_idx=mc["pad_idx"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        layers=mc["layers"],
        dim_ff=mc["dim_ff"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dc = ckpt["diffusion_config"]
    diffusion = AbsorbingDiffusion(dc["T"], dc["mask_idx"], dc["pad_idx"]).to(dev)

    return model, diffusion, vocab, ckpt


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train an absorbing-state D3PM on a malware family.")
    ap.add_argument("--family", required=True, help="malware family folder under --malicia")
    ap.add_argument("--malicia", default="malicia", help="root dir holding family folders")
    ap.add_argument("--out-dir", default="checkpoints", help="where to write checkpoints/<family>/")
    ap.add_argument("--max-files", type=int, default=None, help="cap files read (fast runs)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32, help="mini-batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="base (Adam) learning rate")
    ap.add_argument("--T", type=int, default=None, help="diffusion timesteps (default: Config)")
    ap.add_argument("--max-len", type=int, default=None, help="chunk length (default: Config)")
    ap.add_argument("--min-chunk", type=int, default=32, help="drop tail chunks shorter than this")
    ap.add_argument("--small", action="store_true", help="use the small (fast) Config preset")
    ap.add_argument("--scaleup", action="store_true", help="use the scale-up Config preset (run_02)")
    ap.add_argument("--ema", type=float, default=None, metavar="DECAY",
                    help="EMA decay (e.g. 0.999); sample from the averaged weights")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="fraction of chunks held out for a per-epoch val loss")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after this many epochs of no val improvement (needs --val-frac)")
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="never early-stop before this many epochs (rides out the init plateau)")
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="fraction of total steps to linearly warm up the LR before cosine")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.scaleup:
        cfg = Config.scaleup()
    elif args.small:
        cfg = Config.small()
    else:
        cfg = Config.default()
    # Let explicit flags override the chosen preset's T / max_len.
    if args.T is not None:
        cfg.T = args.T
    if args.max_len is not None:
        cfg.max_len = args.max_len

    train(
        args.family,
        malicia_dir=args.malicia,
        out_dir=args.out_dir,
        max_files=args.max_files,
        epochs=args.epochs,
        batch_size=args.batch,
        cfg=cfg,
        lr=args.lr,
        min_chunk=args.min_chunk,
        ema_decay=args.ema,
        val_frac=args.val_frac,
        patience=args.patience,
        min_epochs=args.min_epochs,
        warmup_frac=args.warmup_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
