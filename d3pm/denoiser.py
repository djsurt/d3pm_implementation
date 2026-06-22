"""
Stage 3 — the *reverse*-process neural network (the "denoiser").

Where this fits
---------------
Stage 2 (`diffusion.py`) gave us the parameter-free **forward** process: given a
clean sequence ``x_0`` and a timestep ``t`` it produces a corrupted ``x_t`` by
replacing some tokens with ``[MASK]``. The **reverse** process has to undo that,
and *that* is what we train. This file is the trainable half.

The x₀-parameterization (D3PM paper §3.3, Eq. 3)
-----------------------------------------------
Instead of predicting the reverse transition ``p_θ(x_{t-1} | x_t)`` directly, the
paper parameterizes the network to predict a distribution over the **clean**
token at every position:

    p̃_θ(x_0 | x_t)        — a categorical over the whole vocabulary, per position.

The actual reverse step is then *reconstructed* (in Stage 6) by combining this
prediction with the known forward posterior ``q(x_{t-1} | x_t, x_0)``. The big
practical win: the network's job is just "look at a partially-masked sequence and
guess the original token at each position" — exactly a BERT-style masked-token
classifier, conditioned additionally on *how noisy* the input is (the timestep
``t``). For the absorbing schedule that conditioning matters because ``t`` tells
the model what fraction of the sequence is masked.

So this module is a function

    f_θ : (x_t : (B, L) ids,  t : (B,) timesteps)  ->  logits : (B, L, vocab_size)

where ``softmax(logits, dim=-1)`` is ``p̃_θ(x_0 | x_t)``. We output raw *logits*
(not probabilities) because the Stage-4 loss is a cross-entropy, and
``F.cross_entropy`` wants logits for numerical stability.

Architecture (a small Transformer encoder)
-------------------------------------------
A sequence model that lets every position attend to every other position is a
natural fit; the order of opcodes matters and masked positions need to be filled
in from surrounding context. We use a standard pre-LN Transformer encoder:

    token id ──► token embedding ┐
    position ──► positional emb ──┼─(sum)─► encoder layers ─► LayerNorm ─► linear head ─► logits
    timestep ──► sinusoidal+MLP ──┘                 ▲
                                                 self-attention with
                                                 src_key_padding_mask (PAD ignored)

Three things get **added** at every position before the encoder:
  * **token embedding** — what opcode is (currently) here. ``[MASK]`` and
    ``[PAD]`` have their own rows; ``padding_idx`` zeros PAD's embedding and
    freezes its gradient so padding contributes no signal.
  * **learned positional embedding** — where in the sequence we are (a plain
    ``nn.Embedding`` indexed by position 0..L-1; learned, like BERT).
  * **time embedding** — *one* vector per example derived from ``t`` via a fixed
    sinusoidal code passed through a small MLP, broadcast across all L positions.
    This is how the single scalar ``t`` conditions the whole network.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ── new PyTorch concepts used in this file ───────────────────────────────────
#
# nn.Embedding(num, dim, padding_idx=...)
#       A lookup table: a trainable (num, dim) matrix. Indexing it with a
#       LongTensor of ids returns the corresponding rows, so a batch of ids
#       (B, L) becomes embeddings (B, L, dim). `padding_idx` pins that row to
#       zeros and keeps its gradient at zero — perfect for [PAD].
#
# nn.TransformerEncoderLayer / nn.TransformerEncoder
#       PyTorch's built-in multi-head self-attention + feed-forward block, and a
#       stack of N of them. `batch_first=True` makes them accept (B, L, d_model)
#       (the layout the rest of our code uses). `norm_first=True` selects the
#       *pre-LN* variant (LayerNorm before each sublayer), which trains more
#       stably for small models than the original post-LN design.
#
# src_key_padding_mask
#       A (B, L) BoolTensor passed to the encoder; `True` marks positions that
#       every query should *ignore* in attention. We pass the PAD mask so padding
#       never contaminates real tokens' representations.
#
# Sinusoidal timestep embedding
#       The same fixed (non-learned) frequency code used by DDPM/Transformers to
#       turn an integer into a smooth vector: for dimension index k,
#           emb[2k]   = sin(t / 10000^(2k/d))
#           emb[2k+1] = cos(t / 10000^(2k/d)).
#       Nearby timesteps get nearby codes, and an MLP on top lets the model shape
#       the conditioning however it needs.
#
# register_buffer (seen in Stage 2)
#       We cache the per-position index vector [0,1,...,L-1] as a buffer so it
#       moves with `.to(device)` and isn't mistaken for a parameter.


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timesteps ``t`` (shape ``(B,)``) to sinusoidal codes ``(B, dim)``.

    Standard Transformer/DDPM positional-style encoding, but indexed by the
    diffusion timestep rather than sequence position::

        emb[i, 2k]   = sin( t_i / 10000^(2k/dim) )
        emb[i, 2k+1] = cos( t_i / 10000^(2k/dim) )

    The frequencies are geometrically spaced from 1 down to 1/10000, so the code
    is smooth in ``t`` (close timesteps → close vectors) yet distinguishes every
    timestep across many scales. ``dim`` is assumed even.

    Args:
        t:    LongTensor (or float) of shape ``(B,)`` — one timestep per example.
        dim:  width of the produced embedding (even).

    Returns:
        FloatTensor of shape ``(B, dim)`` on the same device as ``t``.
    """
    if dim % 2 != 0:
        raise ValueError(f"time-embedding dim must be even, got {dim}")

    device = t.device
    half = dim // 2
    # frequencies: 10000^(-2k/dim) for k = 0..half-1, i.e. exp(-(2k/dim)·ln 10000).
    # Computed in float so the lookup is precise regardless of t's dtype.
    exponent = torch.arange(half, dtype=torch.float32, device=device) / half
    freqs = torch.exp(-math.log(10000.0) * exponent)        # (half,)

    # outer product t ⊗ freqs -> (B, half), then interleave sin/cos into (B, dim).
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)      # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class TransformerDenoiser(nn.Module):
    """Predicts ``p̃_θ(x_0 | x_t)`` as per-position logits over the vocabulary.

    A pre-LN Transformer encoder conditioned on the diffusion timestep. This is
    the only learnable module in the project; ``AbsorbingDiffusion`` (Stage 2/4/6)
    holds the math and calls into this for its clean-token predictions.

    Args:
        vocab_size:  number of token ids (opcodes + [MASK] + [PAD]).
        max_len:     maximum sequence length (size of the positional table).
        pad_idx:     id of [PAD]; its embedding row is zeroed and frozen, and it
                     is masked out of self-attention.
        d_model:     model/embedding width.
        nhead:       number of self-attention heads (must divide ``d_model``).
        layers:      number of Transformer encoder layers.
        dim_ff:      width of each layer's feed-forward sublayer.
        dropout:     dropout probability inside the encoder layers.

    Forward:
        ``forward(x_t, t)`` with ``x_t : (B, L)`` LongTensor of ids and
        ``t : (B,)`` LongTensor of timesteps → logits ``(B, L, vocab_size)``.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.pad_idx = int(pad_idx)
        self.d_model = int(d_model)

        # ── embeddings ──────────────────────────────────────────────────────
        # Token table: one row per vocab id. padding_idx zeros PAD's row and
        # keeps its gradient at zero, so padding injects no signal anywhere.
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Learned absolute positions, BERT-style: a (max_len, d_model) table
        # indexed by position 0..L-1.
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Time conditioning: sinusoidal(t) -> small MLP -> (d_model,) vector per
        # example, later broadcast across all L positions. The MLP lets the model
        # learn a useful nonlinear function of the (fixed) sinusoidal code.
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),                      # smooth activation common in diffusion nets
            nn.Linear(dim_ff, d_model),
        )

        # ── transformer encoder (pre-LN) ────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,     # tensors are (B, L, d_model)
            norm_first=True,      # pre-LN: LayerNorm before attention/FF (stabler)
        )
        # A final LayerNorm after the stack is standard for pre-LN transformers
        # (the residual stream is otherwise never normalized at the very end).
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
            # The nested-tensor fast path is incompatible with pre-LN layers;
            # disabling it avoids a noisy warning and changes nothing numerically.
            enable_nested_tensor=False,
        )

        # ── output head: hidden state -> vocabulary logits ──────────────────
        self.head = nn.Linear(d_model, vocab_size)

        # Cache positions [0, 1, ..., max_len-1] as a buffer so it rides along
        # with .to(device) and isn't treated as a parameter.
        self.register_buffer(
            "positions", torch.arange(max_len), persistent=False
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Small, well-scaled init for the embeddings (the encoder/Linear layers
        already get PyTorch's sensible defaults). Normal(0, 0.02) is the usual
        BERT/GPT embedding init; PAD's row is then re-zeroed."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[self.pad_idx].zero_()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean-token logits for a noised batch.

        Args:
            x_t: LongTensor ``(B, L)`` of token ids (the corrupted sequence).
            t:   LongTensor ``(B,)`` of diffusion timesteps, one per example.

        Returns:
            FloatTensor ``(B, L, vocab_size)`` — unnormalized logits; apply
            ``softmax(dim=-1)`` to read them as ``p̃_θ(x_0 | x_t)``.
        """
        if x_t.dim() != 2:
            raise ValueError(f"x_t must be 2D (B, L), got shape {tuple(x_t.shape)}")
        B, L = x_t.shape
        if L > self.max_len:
            raise ValueError(
                f"sequence length {L} exceeds max_len {self.max_len}; "
                "the positional table cannot index it"
            )
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(
                f"t must have shape ({B},) to match the batch axis of x_t, "
                f"got {tuple(t.shape)}"
            )

        # ── 1. embeddings, summed at every position ─────────────────────────
        tok = self.token_emb(x_t)                       # (B, L, d_model)
        pos = self.pos_emb(self.positions[:L])          # (L, d_model)
        # sinusoidal(t) -> MLP -> (B, d_model); unsqueeze to (B, 1, d_model) so it
        # broadcasts identically across all L positions of each example.
        time = self.time_mlp(sinusoidal_time_embedding(t, self.d_model))  # (B, d_model)
        h = tok + pos.unsqueeze(0) + time.unsqueeze(1)  # (B, L, d_model)

        # ── 2. self-attention, ignoring PAD positions ──────────────────────
        # src_key_padding_mask[i, j] = True  ->  position j is ignored as an
        # attention *key* for every query in example i. This keeps PAD from
        # leaking into real tokens' representations.
        pad_mask = x_t == self.pad_idx                  # (B, L) bool
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # ── 3. project to vocabulary logits ────────────────────────────────
        return self.head(h)                             # (B, L, vocab_size)

    def num_parameters(self) -> int:
        """Total number of trainable parameters (handy for the Stage-3 check)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
