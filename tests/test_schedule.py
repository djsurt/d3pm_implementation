"""Stage 2 unit tests — schedule + q_sample invariants.

These pin down the math of the absorbing-state forward process exactly as the
D3PM paper (§3.1, §3.2, App. A.4.1, App. A.7) specifies it:

  * ᾱ_t = (T - t) / T  on the grid t = 0, ..., T;
  * ᾱ_0 = 1 (clean), ᾱ_T = 0 (fully masked), strictly monotone in between;
  * q(x_t | x_0) masks each non-PAD token independently with prob 1 - ᾱ_t;
  * PAD tokens are never touched;
  * t = 0 is the identity, t = T fully masks all non-PAD tokens.
"""

import math

import pytest
import torch

from d3pm.diffusion import AbsorbingDiffusion


# Tiny constants — keep the tests fast (everything stays on CPU and tiny).
T = 100
MASK = 10
PAD = 11
B, L = 4, 8


def _make() -> AbsorbingDiffusion:
    return AbsorbingDiffusion(T=T, mask_idx=MASK, pad_idx=PAD)


# ── schedule (ᾱ_t buffer) ────────────────────────────────────────────────────


def test_alpha_bar_shape_and_endpoints():
    d = _make()
    assert d.alpha_bar.shape == (T + 1,)
    assert d.alpha_bar.dtype == torch.float32
    assert d.alpha_bar[0].item() == 1.0     # clean data
    assert d.alpha_bar[T].item() == 0.0     # fully masked


def test_alpha_bar_strictly_monotone_decreasing():
    d = _make()
    diffs = d.alpha_bar[1:] - d.alpha_bar[:-1]
    assert (diffs < 0).all().item()
    # Equally spaced by -1/T (the schedule is perfectly linear).
    assert torch.allclose(diffs, torch.full_like(diffs, -1.0 / T))


def test_alpha_bar_matches_closed_form():
    d = _make()
    expected = (T - torch.arange(T + 1, dtype=torch.float32)) / T
    assert torch.allclose(d.alpha_bar, expected)


def test_alpha_bar_is_a_buffer_not_a_parameter():
    """ᾱ_t is fixed schedule state — it must NOT show up in .parameters()."""
    d = _make()
    assert list(d.parameters()) == []                  # no learnable params
    assert "alpha_bar" in dict(d.named_buffers())      # but registered as buffer


def test_constructor_validates_inputs():
    with pytest.raises(ValueError):
        AbsorbingDiffusion(T=0, mask_idx=MASK, pad_idx=PAD)
    with pytest.raises(ValueError):
        AbsorbingDiffusion(T=T, mask_idx=5, pad_idx=5)  # mask_idx == pad_idx


# ── q_sample structural checks ───────────────────────────────────────────────


def test_q_sample_output_shape_and_dtype_match_x0():
    torch.manual_seed(0)
    d = _make()
    x0 = torch.randint(0, MASK, (B, L), dtype=torch.long)  # avoid MASK in x0
    t = torch.randint(0, T + 1, (B,), dtype=torch.long)
    xt = d.q_sample(x0, t)
    assert xt.shape == x0.shape
    assert xt.dtype == x0.dtype


def test_q_sample_each_position_is_original_or_mask():
    """Every output token equals either the original x0 token or the MASK id —
    never some unrelated value (no off-by-one in the indexing/where logic)."""
    torch.manual_seed(0)
    d = _make()
    x0 = torch.randint(0, MASK, (B, L), dtype=torch.long)
    t = torch.randint(0, T + 1, (B,), dtype=torch.long)
    xt = d.q_sample(x0, t)
    assert ((xt == x0) | (xt == MASK)).all().item()


def test_q_sample_rejects_t_out_of_range():
    d = _make()
    x0 = torch.zeros((B, L), dtype=torch.long)
    with pytest.raises(ValueError):
        d.q_sample(x0, torch.tensor([-1, 0, 0, 0], dtype=torch.long))
    with pytest.raises(ValueError):
        d.q_sample(x0, torch.tensor([0, 0, 0, T + 1], dtype=torch.long))


def test_q_sample_rejects_wrong_batch_size():
    d = _make()
    x0 = torch.zeros((B, L), dtype=torch.long)
    with pytest.raises(ValueError):
        d.q_sample(x0, torch.zeros(B + 1, dtype=torch.long))


# ── q_sample endpoints (deterministic) ──────────────────────────────────────


def test_q_sample_t_zero_is_identity():
    """At t = 0 mask_prob = 0, so x_t must equal x_0 exactly."""
    torch.manual_seed(0)
    d = _make()
    x0 = torch.randint(0, MASK, (B, L), dtype=torch.long)
    t = torch.zeros(B, dtype=torch.long)
    xt = d.q_sample(x0, t)
    assert torch.equal(xt, x0)


def test_q_sample_t_T_masks_every_nonpad_position():
    """At t = T mask_prob = 1, so every non-PAD position becomes MASK."""
    torch.manual_seed(0)
    d = _make()
    x0 = torch.randint(0, MASK, (B, L), dtype=torch.long)
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -2:] = True
    x0[pad_mask] = PAD

    t = torch.full((B,), T, dtype=torch.long)
    xt = d.q_sample(x0, t, pad_mask=pad_mask)
    assert (xt[~pad_mask] == MASK).all().item()
    assert (xt[pad_mask] == PAD).all().item()


# ── q_sample stochastic invariants ──────────────────────────────────────────


def test_q_sample_preserves_pad_at_every_t():
    """PAD positions must stay PAD regardless of t."""
    torch.manual_seed(0)
    d = _make()
    x0 = torch.randint(0, MASK, (B, L), dtype=torch.long)
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -3:] = True
    x0[pad_mask] = PAD

    for t_val in [1, T // 4, T // 2, 3 * T // 4, T - 1, T]:
        t = torch.full((B,), t_val, dtype=torch.long)
        xt = d.q_sample(x0, t, pad_mask=pad_mask)
        assert (xt[pad_mask] == PAD).all().item(), f"PAD got modified at t={t_val}"


def test_q_sample_empirical_masked_fraction_matches_t_over_T():
    """Fraction of non-PAD positions that end up masked ≈ t/T.

    The standard error of an empirical mean of N Bernoulli(p) draws is
    sqrt(p(1-p)/N). With N = 256 * 64 ≈ 16k and p ≤ 1, se ≤ 0.004, so a 0.02
    tolerance leaves a very wide margin.
    """
    torch.manual_seed(0)
    d = _make()
    big_B, big_L = 256, 64
    x0 = torch.randint(0, MASK, (big_B, big_L), dtype=torch.long)
    for t_val in [T // 4, T // 2, 3 * T // 4]:
        t = torch.full((big_B,), t_val, dtype=torch.long)
        xt = d.q_sample(x0, t)
        frac = (xt == MASK).float().mean().item()
        expected = t_val / T
        assert math.isclose(frac, expected, abs_tol=0.02), (
            f"t={t_val}: observed masked fraction {frac:.4f}, expected {expected:.4f}"
        )


def test_q_sample_handles_per_example_t():
    """Each row uses its own t — the (B, 1) broadcast must work correctly."""
    torch.manual_seed(0)
    d = _make()
    big_B, big_L = 100, 200
    x0 = torch.randint(0, MASK, (big_B, big_L), dtype=torch.long)
    # First half: t = 0 (identity); second half: t = T (fully masked).
    t = torch.cat([
        torch.zeros(big_B // 2, dtype=torch.long),
        torch.full((big_B // 2,), T, dtype=torch.long),
    ])
    xt = d.q_sample(x0, t)
    assert torch.equal(xt[: big_B // 2], x0[: big_B // 2])
    assert (xt[big_B // 2 :] == MASK).all().item()


def test_q_sample_state_dict_roundtrips_the_buffer():
    """Saving + reloading the module preserves the schedule."""
    d = _make()
    sd = d.state_dict()
    assert "alpha_bar" in sd
    d2 = AbsorbingDiffusion(T=T, mask_idx=MASK, pad_idx=PAD)
    d2.load_state_dict(sd)
    assert torch.equal(d.alpha_bar, d2.alpha_bar)
