"""Stage 6 unit tests — the reverse (sampling) process.

These pin down the contract of ``AbsorbingDiffusion.sample`` implied by the
absorbing reverse posterior (paper §3.1, App. A.2.2 / A.3):

  * output has shape (n, seq_len), dtype long;
  * the output is CLEAN — no [MASK] or [PAD] id survives the full chain (every
    position un-masks by t=1, and MASK/PAD are forbidden as predicted tokens);
  * with an oracle denoiser that always predicts one fixed opcode, the whole
    output is that opcode (the model fully controls the un-masked values);
  * the forward process is absorbing in reverse too: the count of [MASK] tokens
    is monotonically NON-INCREASING along the trajectory and the trajectory
    starts all-[MASK] and ends == the returned sample;
  * even a denoiser that *tries* to emit [MASK]/[PAD] cannot — those are masked
    out, so the output is still clean.

A StubDenoiser lets us drive the un-masking deterministically without training.
"""

import torch
from torch import nn

from d3pm.diffusion import AbsorbingDiffusion


VOCAB = 12
MASK = 10
PAD = 11
T = 25
BIG = 30.0


def _diff() -> AbsorbingDiffusion:
    return AbsorbingDiffusion(T=T, mask_idx=MASK, pad_idx=PAD)


class FixedDenoiser(nn.Module):
    """Predicts a single fixed token id at every position (near one-hot logits).

    With ``target`` a real opcode id this lets us assert the sampler writes
    exactly that token. With ``target`` set to MASK/PAD it lets us check the
    sampler's -inf forbidding still yields clean output.
    """

    def __init__(self, target: int) -> None:
        super().__init__()
        self.target = target

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x_t.shape
        logits = torch.zeros(B, L, VOCAB)
        logits[..., self.target] = BIG
        return logits


def test_sample_shape_and_dtype():
    diff = _diff()
    x0 = diff.sample(FixedDenoiser(3), n=4, seq_len=16)
    assert x0.shape == (4, 16)
    assert x0.dtype == torch.long


def test_sample_output_is_clean():
    diff = _diff()
    torch.manual_seed(0)
    x0 = diff.sample(FixedDenoiser(3), n=8, seq_len=32)
    assert (x0 == MASK).sum().item() == 0
    assert (x0 == PAD).sum().item() == 0


def test_oracle_fixed_token_fills_output():
    # A denoiser that always predicts token 5 → every position must end as 5.
    diff = _diff()
    torch.manual_seed(1)
    x0 = diff.sample(FixedDenoiser(5), n=4, seq_len=20)
    assert (x0 == 5).all()


def test_forbidden_tokens_still_clean():
    # Even if the net puts all its mass on MASK/PAD, the -inf forbidding means
    # neither can be written; the output is some real opcode, never MASK/PAD.
    diff = _diff()
    torch.manual_seed(2)
    for bad in (MASK, PAD):
        x0 = diff.sample(FixedDenoiser(bad), n=4, seq_len=16)
        assert (x0 == MASK).sum().item() == 0
        assert (x0 == PAD).sum().item() == 0


def test_trajectory_is_absorbing_and_consistent():
    diff = _diff()
    torch.manual_seed(3)
    x0, traj = diff.sample(FixedDenoiser(7), n=4, seq_len=24, return_trajectory=True)
    # T steps → T+1 snapshots (t=T down through t=0).
    assert len(traj) == T + 1
    # starts fully masked, ends == returned sample.
    assert (traj[0] == MASK).all()
    assert torch.equal(traj[-1], x0)
    # mask count never increases (absorbing): once un-masked, stays un-masked.
    mask_counts = [(s == MASK).sum().item() for s in traj]
    assert mask_counts[0] == 4 * 24
    assert mask_counts[-1] == 0
    assert all(b <= a for a, b in zip(mask_counts, mask_counts[1:]))


def test_sample_is_deterministic_under_seed():
    diff = _diff()
    torch.manual_seed(42)
    a = diff.sample(FixedDenoiser(2), n=4, seq_len=16)
    torch.manual_seed(42)
    b = diff.sample(FixedDenoiser(2), n=4, seq_len=16)
    assert torch.equal(a, b)
