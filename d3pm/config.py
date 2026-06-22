"""
Stage 0 — configuration: where to run, and the default model/diffusion sizes.

Two things live here:

1. `device()` — picks the hardware PyTorch will compute on. A PyTorch *tensor*
   (the n-dimensional array type everything is built from) lives either in CPU
   RAM or on an accelerator. On your Mac the accelerator is "MPS" (Metal
   Performance Shaders = Apple-Silicon GPU). On an NVIDIA machine it's "cuda".
   We prefer the fastest available and fall back to plain CPU.

2. `Config` — one place for every hyperparameter, with two presets:
     - `Config.small()`  tiny + fast, for local iteration and verification.
     - `Config.default()` the values we'd use for a real training run.
   Keeping these in a dataclass (instead of scattering magic numbers) means a
   stage's verification script and the training CLI agree on the same shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def device() -> torch.device:
    """Return the best available compute device: cuda > mps > cpu.

    `torch.device` is just a handle naming where tensors/models should live.
    You move data onto it with `.to(device)`; ops then run on that hardware.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    """All knobs for the diffusion process and the denoiser network.

    Diffusion knobs:
      T        number of diffusion timesteps (how many noising steps 0 -> all-MASK)
      max_len  sequence length the model sees (long files are split into chunks)

    Denoiser (transformer) knobs:
      d_model  width of the model (embedding/hidden size)
      nhead    number of self-attention heads (must divide d_model)
      layers   number of transformer encoder layers (depth)
      dim_ff   width of the feed-forward sublayer inside each layer

    Loss knob:
      lambda_ce  weight of the auxiliary cross-entropy term in L = L_vb + λ·L_CE
    """

    T: int = 500
    max_len: int = 256
    d_model: int = 128
    nhead: int = 4
    layers: int = 4
    dim_ff: int = 512
    lambda_ce: float = 0.01

    @classmethod
    def small(cls) -> "Config":
        """Tiny + fast preset for verification and local sanity checks."""
        return cls(T=100, max_len=128, d_model=64, nhead=4, layers=2, dim_ff=128)

    @classmethod
    def default(cls) -> "Config":
        """Sensible defaults for a real (small-scale) training run."""
        return cls()

    @classmethod
    def scaleup(cls) -> "Config":
        """Larger preset for run_02 — the capacity scale-up.

        run_01 (the `default` preset, ~1M params) plateaued in loss by ~epoch 50
        and under-covered the real n-gram diversity: it was capacity-bound, not
        time-bound. This preset roughly 8×'s the model (d_model 128→256, layers
        4→6, dim_ff 512→1024 ≈ 8M params), doubles the context (max_len 256→512;
        real files run to ~10k opcodes), and uses a finer diffusion schedule
        (T 500→1000, fewer simultaneous un-maskings per reverse step). `nhead=8`
        keeps `d_model/nhead = 32`. Pair with EMA + a held-out val split at the
        training-loop level (see `train.py`).
        """
        return cls(T=1000, max_len=512, d_model=256, nhead=8, layers=6, dim_ff=1024)
