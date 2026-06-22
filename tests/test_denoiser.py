"""Stage 3 unit tests — TransformerDenoiser + sinusoidal time embedding.

These pin down the contract the rest of the project relies on:

  * forward(x_t, t) -> finite logits of shape (B, L, vocab_size);
  * the sinusoidal time embedding has the right shape/values and is smooth;
  * the timestep t actually conditions the output (it is not ignored);
  * [PAD] is isolated: its embedding row is zero/frozen and it is masked out of
    self-attention, so PAD content never changes real positions' logits;
  * input validation rejects the obvious shape mistakes.
"""

import math

import pytest
import torch

from d3pm.denoiser import TransformerDenoiser, sinusoidal_time_embedding


VOCAB = 12
MAX_LEN = 16
PAD = 11
MASK = 10
B, L = 4, 16


def _make(**kw) -> TransformerDenoiser:
    args = dict(
        vocab_size=VOCAB, max_len=MAX_LEN, pad_idx=PAD,
        d_model=32, nhead=4, layers=2, dim_ff=64, dropout=0.0,
    )
    args.update(kw)
    return TransformerDenoiser(**args)


# ── sinusoidal time embedding ────────────────────────────────────────────────


def test_time_embedding_shape_and_finite():
    t = torch.arange(5)
    emb = sinusoidal_time_embedding(t, dim=16)
    assert emb.shape == (5, 16)
    assert torch.isfinite(emb).all()


def test_time_embedding_values_first_freq():
    # First sin/cos pair uses frequency 1, so column 0 is sin(t), column half is cos(t).
    dim = 8
    half = dim // 2
    t = torch.tensor([0, 1, 2, 3])
    emb = sinusoidal_time_embedding(t, dim=dim)
    assert torch.allclose(emb[:, 0], torch.sin(t.float()), atol=1e-6)
    assert torch.allclose(emb[:, half], torch.cos(t.float()), atol=1e-6)


def test_time_embedding_t0_is_sin0_cos0():
    emb = sinusoidal_time_embedding(torch.tensor([0]), dim=8)
    half = 4
    assert torch.allclose(emb[0, :half], torch.zeros(half), atol=1e-6)   # sin 0 = 0
    assert torch.allclose(emb[0, half:], torch.ones(half), atol=1e-6)    # cos 0 = 1


def test_time_embedding_odd_dim_raises():
    with pytest.raises(ValueError):
        sinusoidal_time_embedding(torch.tensor([1]), dim=7)


def test_time_embedding_distinct_for_distinct_t():
    emb = sinusoidal_time_embedding(torch.tensor([1, 50, 99]), dim=16)
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[1], emb[2])


# ── forward pass contract ────────────────────────────────────────────────────


def test_forward_shape_and_finite():
    model = _make().eval()
    x_t = torch.randint(0, VOCAB, (B, L))
    t = torch.randint(1, 101, (B,))
    out = model(x_t, t)
    assert out.shape == (B, L, VOCAB)
    assert torch.isfinite(out).all()


def test_forward_accepts_shorter_than_max_len():
    model = _make().eval()
    short = L - 5
    x_t = torch.randint(0, VOCAB, (B, short))
    t = torch.randint(1, 101, (B,))
    out = model(x_t, t)
    assert out.shape == (B, short, VOCAB)


def test_forward_rejects_too_long():
    model = _make().eval()
    x_t = torch.randint(0, VOCAB, (B, MAX_LEN + 1))
    t = torch.randint(1, 101, (B,))
    with pytest.raises(ValueError):
        model(x_t, t)


def test_forward_rejects_bad_t_shape():
    model = _make().eval()
    x_t = torch.randint(0, VOCAB, (B, L))
    with pytest.raises(ValueError):
        model(x_t, torch.randint(1, 101, (B + 1,)))


def test_forward_rejects_non_2d_x():
    model = _make().eval()
    with pytest.raises(ValueError):
        model(torch.randint(0, VOCAB, (B, L, 1)), torch.randint(1, 101, (B,)))


# ── time conditioning ────────────────────────────────────────────────────────


def test_time_actually_conditions_output():
    torch.manual_seed(0)
    model = _make().eval()
    x_t = torch.randint(0, VOCAB, (B, L))
    out_a = model(x_t, torch.full((B,), 1))
    out_b = model(x_t, torch.full((B,), 100))
    assert (out_a - out_b).abs().max().item() > 1e-4


def test_same_input_is_deterministic_in_eval():
    model = _make().eval()
    x_t = torch.randint(0, VOCAB, (B, L))
    t = torch.full((B,), 7)
    assert torch.allclose(model(x_t, t), model(x_t, t))


# ── PAD isolation ────────────────────────────────────────────────────────────


def test_pad_embedding_row_is_zero():
    model = _make()
    assert torch.allclose(model.token_emb.weight[PAD], torch.zeros(model.d_model))


def test_pad_does_not_affect_real_positions():
    # Trailing PAD must not change the logits at the real positions: a sequence
    # of real length `r` should give the same logits whether or not it is padded
    # out to the full length L. (PAD is zeroed in the embedding AND masked out of
    # self-attention, so it cannot influence real tokens.)
    torch.manual_seed(0)
    model = _make().eval()
    r = L - 6
    real_tokens = torch.randint(0, VOCAB - 2, (B, r))   # avoid MASK/PAD ids
    t = torch.full((B,), 5)

    # unpadded: length exactly r
    out_short = model(real_tokens, t)                   # (B, r, VOCAB)

    # padded: same r real tokens, then PAD to length L
    padded = torch.full((B, L), PAD)
    padded[:, :r] = real_tokens
    out_long = model(padded, t)                          # (B, L, VOCAB)

    assert (out_short - out_long[:, :r]).abs().max().item() < 1e-4


# ── construction guards ──────────────────────────────────────────────────────


def test_nhead_must_divide_d_model():
    with pytest.raises(ValueError):
        _make(d_model=30, nhead=4)


def test_param_count_positive():
    assert _make().num_parameters() > 0


def test_buffers_move_and_are_not_parameters():
    model = _make()
    names = {n for n, _ in model.named_parameters()}
    assert "positions" not in names           # it's a buffer, not a parameter
    assert model.positions.shape == (MAX_LEN,)
