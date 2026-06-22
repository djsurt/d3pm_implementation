"""Stage 4 unit tests — the hybrid D3PM training loss.

These pin down the loss contract the paper's reduction (§3.3, Eq. 4–5; App. A.3)
implies:

  * loss(model, x0, pad_mask) returns finite scalar diagnostics {loss, vb, ce};
  * backprop through ["loss"] fills gradients on the model's parameters;
  * the loss scores ONLY masked positions: a model that is perfect on masked
    positions gives ~0 loss even if it is wrong on unmasked or PAD positions;
  * conversely, being wrong on masked positions makes the loss clearly positive;
  * PAD is never scored.

A small fixed T keeps the masking dense so "some positions are masked" holds with
overwhelming probability and the tests are not flaky.
"""

import pytest
import torch
from torch import nn

from d3pm.diffusion import AbsorbingDiffusion


VOCAB = 12
MASK = 10
PAD = 11
T = 10
B, L = 8, 64
BIG = 30.0  # logit magnitude that makes softmax effectively one-hot


def _diff() -> AbsorbingDiffusion:
    return AbsorbingDiffusion(T=T, mask_idx=MASK, pad_idx=PAD)


class StubDenoiser(nn.Module):
    """A non-trained 'oracle' denoiser used to probe *which* positions the loss
    scores. It emits near-one-hot logits at the true ``x0`` everywhere, except it
    can deliberately sabotage a chosen class of positions (point them at a wrong
    token) so we can see whether that class contributes to the loss.

    sabotage ∈ {"none", "masked", "unmasked", "pad"} — chosen relative to ``x_t``.
    """

    def __init__(self, x0: torch.Tensor, sabotage: str = "none") -> None:
        super().__init__()
        self.x0 = x0
        self.sabotage = sabotage
        # one real parameter so the gradient test has something to fill.
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        Bx, Lx = x_t.shape
        logits = torch.zeros(Bx, Lx, VOCAB)
        # perfect prediction: BIG logit on the true token at every position.
        target = self.x0.clone()
        if self.sabotage == "masked":
            where = x_t == MASK
        elif self.sabotage == "unmasked":
            where = (x_t != MASK) & (x_t != PAD)
        elif self.sabotage == "pad":
            where = x_t == PAD
        else:
            where = torch.zeros_like(x_t, dtype=torch.bool)
        # at sabotaged positions point to a deliberately WRONG token.
        target = torch.where(where, (self.x0 + 1) % VOCAB, target)
        logits.scatter_(-1, target.unsqueeze(-1), BIG)
        return logits * self.scale


def _make_batch():
    torch.manual_seed(0)
    x0 = torch.randint(0, MASK, (B, L))            # only real opcode ids 0..9
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -8:] = True                         # last 8 positions are PAD
    x0[pad_mask] = PAD
    return x0, pad_mask


# ── basic contract ───────────────────────────────────────────────────────────


def test_loss_returns_finite_scalar_dict():
    x0, pad_mask = _make_batch()
    out = _diff().loss(StubDenoiser(x0), x0, pad_mask)
    assert set(out) == {"loss", "vb", "ce"}
    for k, v in out.items():
        assert v.dim() == 0 and torch.isfinite(v), k


def test_loss_backward_fills_gradients():
    torch.manual_seed(1)
    x0, pad_mask = _make_batch()
    # a tiny REAL denoiser so there are many params to receive gradient.
    from d3pm.denoiser import TransformerDenoiser
    model = TransformerDenoiser(
        vocab_size=VOCAB, max_len=L, pad_idx=PAD,
        d_model=32, nhead=4, layers=2, dim_ff=64, dropout=0.0,
    )
    out = _diff().loss(model, x0, pad_mask)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_loss_rejects_non_2d_x0():
    with pytest.raises(ValueError):
        _diff().loss(StubDenoiser(torch.zeros(1, 1, dtype=torch.long)),
                     torch.zeros(1, 1, 1, dtype=torch.long))


# ── which positions are scored ───────────────────────────────────────────────


def test_perfect_prediction_gives_zero_loss():
    x0, pad_mask = _make_batch()
    torch.manual_seed(2)
    out = _diff().loss(StubDenoiser(x0, sabotage="none"), x0, pad_mask)
    assert out["loss"].item() < 1e-4
    assert out["vb"].item() < 1e-4
    assert out["ce"].item() < 1e-4


def test_wrong_on_unmasked_does_not_matter():
    # unmasked positions contribute zero KL → sabotaging them keeps loss ~0.
    x0, pad_mask = _make_batch()
    torch.manual_seed(2)
    out = _diff().loss(StubDenoiser(x0, sabotage="unmasked"), x0, pad_mask)
    assert out["loss"].item() < 1e-4


def test_wrong_on_pad_does_not_matter():
    # PAD is never scored → sabotaging PAD predictions keeps loss ~0.
    x0, pad_mask = _make_batch()
    torch.manual_seed(2)
    out = _diff().loss(StubDenoiser(x0, sabotage="pad"), x0, pad_mask)
    assert out["loss"].item() < 1e-4


def test_wrong_on_masked_makes_loss_positive():
    # masked positions ARE scored → sabotaging them must blow the loss up.
    x0, pad_mask = _make_batch()
    torch.manual_seed(2)
    out = _diff().loss(StubDenoiser(x0, sabotage="masked"), x0, pad_mask)
    assert out["loss"].item() > 1.0


def test_works_without_pad_mask():
    torch.manual_seed(3)
    x0 = torch.randint(0, MASK, (B, L))
    out = _diff().loss(StubDenoiser(x0, sabotage="none"), x0, pad_mask=None)
    assert out["loss"].item() < 1e-4
