"""
Stage 2 — the *forward* (noising) part of the absorbing-state D3PM.

Where this fits
---------------
A diffusion model is two coupled Markov chains over a sequence of latent states
``x_0, x_1, ..., x_T``:

  * the **forward** process ``q(x_t | x_{t-1})`` corrupts the data step by step
    until it reaches a known stationary distribution at ``t = T``; it has no
    learnable parameters,
  * the **reverse** process ``p_θ(x_{t-1} | x_t)`` is a neural network trained
    to undo one corruption step.

This file owns the forward side (Stage 2) and now the training loss (Stage 4).
The reverse-side neural net lives in Stage 3 (``denoiser.py``); sampling (the
learned reverse process) comes in Stage 6.

The absorbing-state variant (paper §3.1, App. A.2.2)
----------------------------------------------------
For categorical tokens with vocabulary {0, ..., K-1} plus a special ``[MASK]``
token (id ``mask_idx``), the per-step transition matrix is

    Q_t :  with prob (1 - β_t) stay the same; with prob β_t jump to [MASK].

``[MASK]`` is *absorbing*: once a token is masked it stays masked forever. The
stationary distribution at ``t = T`` is therefore all-``[MASK]``.

Closed-form ``t``-step marginal (App. A.4.1):

    q(x_t | x_0)  =  Cat( x_t ;  p = ᾱ_t · onehot(x_0) + (1 − ᾱ_t) · onehot([MASK]) )

i.e. each token of ``x_0`` independently *stays* with probability ``ᾱ_t`` or
*has already been replaced* by ``[MASK]`` with probability ``1 − ᾱ_t``.
For this paper's schedule

    β_t = 1 / (T − t + 1),   t = 1, ..., T            (§3.2, App. A.7)

the cumulative "stay original" probability simplifies (it is a telescoping
product) to a perfectly linear schedule:

    ᾱ_t  =  ∏_{s=1..t} (1 − β_s)  =  (T − t) / T,    t = 0, ..., T

with ``ᾱ_0 = 1`` (clean data) and ``ᾱ_T = 0`` (everything is ``[MASK]``).
We precompute and cache ``ᾱ_t`` for every ``t`` in a *buffer* (see ``__init__``
below).

PAD is *not* part of the diffusion process
------------------------------------------
``[PAD]`` only exists to make every chunk the same length so we can stack them
into a tensor; it does not carry signal. We therefore exclude PAD positions
from masking — they stay as PAD throughout, and the loss (Stage 4) will skip
them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# ── new PyTorch concept used in this file ────────────────────────────────────
#
# nn.Module  — the base class for anything PyTorch should treat as part of the
#              model: trainable layers, but also fixed *state* (like our noise
#              schedule). Subclassing it gives us three things we use here:
#
#                * a place to register **buffers**: tensors that are part of
#                  the module's state but are *not* parameters. They move with
#                  ``.to(device)`` and are saved by ``torch.save`` along with
#                  the model, but autograd ignores them and no optimizer will
#                  update them. Perfect for a precomputed noise schedule.
#                * automatic device tracking: once ``diffusion.to("mps")`` is
#                  called, ``self.alpha_bar`` lives on the GPU too.
#                * a uniform API with the denoiser (Stage 3) so the training
#                  loop can ``.to(device)`` both with the same call.
#
# torch.where(cond, a, b)   — elementwise "if cond then a else b" on tensors.
# torch.rand(shape, ...)    — independent uniforms in [0, 1).
# Broadcasting              — a (B, 1) tensor compared with a (B, L) tensor is
#                             automatically expanded along the length axis so
#                             each row uses its own per-example value.


class AbsorbingDiffusion(nn.Module):
    """Forward (noising) process of an absorbing-state D3PM.

    No learnable parameters live here; the only state is the precomputed
    schedule ``alpha_bar``. The Stage 4 training loss and the Stage 6 sampler
    will live on this same class so that the schedule and the reverse process
    stay in lockstep.

    Args:
        T:         number of diffusion timesteps (``β_t`` is defined for
                   ``t = 1..T``; ``t = 0`` denotes the clean data).
        mask_idx:  vocabulary id of ``[MASK]`` — the absorbing state.
        pad_idx:   vocabulary id of ``[PAD]`` — excluded from masking.

    Buffers:
        alpha_bar  shape ``(T + 1,)``, ``alpha_bar[t] = (T - t) / T``.
                   ``alpha_bar[0] = 1`` (clean), ``alpha_bar[T] = 0`` (fully masked).
    """

    def __init__(self, T: int, mask_idx: int, pad_idx: int) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if mask_idx == pad_idx:
            raise ValueError("mask_idx and pad_idx must differ")

        self.T = int(T)
        self.mask_idx = int(mask_idx)
        self.pad_idx = int(pad_idx)

        # Precompute ᾱ_t for t = 0, 1, ..., T as a length-(T+1) float tensor.
        #   t        : 0   1   2   ...  T-1   T
        #   T - t    : T   T-1 T-2 ...  1     0
        #   ᾱ_t      : 1   ...               0   (perfectly linear by §3.2)
        # `torch.arange(T + 1, dtype=torch.float32)` is [0, 1, ..., T].
        t_grid = torch.arange(self.T + 1, dtype=torch.float32)
        alpha_bar = (self.T - t_grid) / self.T

        # register_buffer makes `alpha_bar` a non-parameter attribute of the
        # module: it is included in `state_dict()`, moves with `.to(device)`,
        # but autograd never tracks it and the optimizer never sees it. This
        # is the standard way to attach precomputed constants to a Module.
        self.register_buffer("alpha_bar", alpha_bar)

    # ── q(x_t | x_0) — one-shot forward noising ──────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``x_t ~ q(x_t | x_0)`` for the absorbing-state schedule.

        Per the closed-form marginal (App. A.4.1) each non-PAD token of ``x_0``
        is *independently*:

            * kept as-is        with probability  ``ᾱ_t``  =  (T - t) / T,
            * replaced by MASK  with probability  ``1 - ᾱ_t`` = t / T.

        PAD tokens (where ``pad_mask`` is True) are never masked.

        Args:
            x0:       LongTensor of shape ``(B, L)`` — clean token ids.
            t:        LongTensor of shape ``(B,)`` — one timestep per example.
                      Each entry must lie in ``[0, T]``. ``t = 0`` is identity;
                      ``t = T`` is fully-masked.
            pad_mask: BoolTensor of shape ``(B, L)`` or ``None``. ``True`` marks
                      positions to leave untouched (PAD). If ``None``, every
                      position is eligible for masking.

        Returns:
            LongTensor of shape ``(B, L)``, same dtype/device as ``x0``. Every
            position is either ``x0[i, j]`` or ``mask_idx`` (or PAD, if masked).
        """
        if x0.dim() != 2:
            raise ValueError(f"x0 must be 2D (B, L), got shape {tuple(x0.shape)}")
        if t.dim() != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError(
                f"t must have shape ({x0.shape[0]},) to match the batch axis of x0, "
                f"got {tuple(t.shape)}"
            )
        if (t < 0).any() or (t > self.T).any():
            raise ValueError(f"every entry of t must lie in [0, {self.T}]")
        if pad_mask is not None:
            if pad_mask.shape != x0.shape:
                raise ValueError(
                    f"pad_mask must have the same shape as x0, got "
                    f"{tuple(pad_mask.shape)} vs {tuple(x0.shape)}"
                )
            if pad_mask.dtype != torch.bool:
                raise ValueError(f"pad_mask must be a BoolTensor, got dtype {pad_mask.dtype}")

        # Move `t` onto the schedule's device for the index lookup below. After
        # `module.to(device)`, `self.alpha_bar` lives on that device; if the
        # caller passed `t` on CPU we want to silently fix that here.
        t = t.to(self.alpha_bar.device)

        # ── 1. per-example "stay original" probability ──────────────────────
        # Index a 1-D buffer by a 1-D LongTensor -> shape (B,). Then add a
        # length-1 axis so it broadcasts across the sequence axis L below.
        alpha_t = self.alpha_bar[t]                       # (B,)
        mask_prob = (1.0 - alpha_t).unsqueeze(1)          # (B, 1)

        # ── 2. independent Bernoulli per (example, position) ────────────────
        # rand[i, j] ~ Uniform[0, 1); mask if rand < mask_prob[i, 0]. The
        # comparison broadcasts the (B, 1) probability across the L axis, so
        # each example uses its own t. We don't use `torch.rand_like(x0)`
        # because x0 is int64; `rand` needs to be float.
        rand = torch.rand(x0.shape, device=x0.device)     # (B, L) float in [0, 1)
        should_mask = rand < mask_prob                    # (B, L) bool

        # ── 3. exclude PAD positions ────────────────────────────────────────
        if pad_mask is not None:
            should_mask = should_mask & ~pad_mask          # PAD never masked

        # ── 4. write [MASK] where should_mask is True, else keep x0 ─────────
        # `torch.full_like(x0, mask_idx)` is a tensor of the same shape/dtype/
        # device as x0, filled with `mask_idx`. `torch.where(cond, a, b)` then
        # picks elementwise: `a` where cond is True, `b` where False.
        mask_tensor = torch.full_like(x0, self.mask_idx)
        return torch.where(should_mask, mask_tensor, x0)

    # ── training loss (Stage 4) ──────────────────────────────────────────────

    def loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        lambda_ce: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        """Hybrid D3PM training loss ``L_λ = L_vb + λ·L_CE`` for one batch.

        Design note
        -----------
        ``AbsorbingDiffusion`` deliberately owns no parameters (see Stage 2), so
        the denoiser is passed in as ``model`` rather than stored here. This keeps
        the diffusion *math* separate from the *network*. (The Stage-4 plan wrote
        the signature as ``loss(x0, pad_mask)`` for brevity; we add ``model``.)

        The variational-bound reduction (paper §3.3, Eq. 4–5; App. A.3)
        ---------------------------------------------------------------
        We parameterize the reverse process by predicting the *clean* token,
        ``p̃_θ(x_0 | x_t)`` (the x₀-parameterization). For the **absorbing**
        forward process the posterior ``q(x_{t-1} | x_t, x_0)`` is degenerate:

          * a position that is **not masked** in ``x_t`` stays put — it contributes
            **zero** KL (q and p_θ both leave it unchanged), so unmasked positions
            drop out of the loss entirely;
          * a position that **is masked** in ``x_t`` either un-masks to its true
            value ``x_0`` with probability ``1/t`` or stays masked with probability
            ``(t-1)/t``. The KL of this two-outcome posterior against the model
            reduces (the stay-masked branch is θ-independent) to

                ``(1/t) · ( −log p̃_θ([x_0]_i | x_t) )``   (+ a const. independent of θ).

        Summing over the masked positions and taking the expectation over
        ``t ~ U{1..T}`` gives the per-batch ``L_vb`` we implement. The hybrid
        objective (paper §6) adds a plain denoising cross-entropy over the same
        masked positions, ``L_CE``, weighted by ``λ = 0.01``.

        The ``1/t`` weight is exactly what *balances* the schedule: at step ``t``
        roughly ``(t/T)`` of the sequence is masked, so the ``(1/t)`` factor cancels
        the growing count of masked tokens and every timestep contributes about
        equally to the bound. For that balance to survive we normalize by the
        (``t``-independent) number of **non-PAD** tokens in the batch — *not* by the
        number of masked tokens (which scales with ``t`` and would undo the
        balancing).

        Args:
            model:     the denoiser; called as ``model(x_t, t) -> (B, L, vocab)``
                       logits for ``p̃_θ(x_0 | x_t)``.
            x0:        LongTensor ``(B, L)`` — clean token ids.
            pad_mask:  BoolTensor ``(B, L)`` or ``None``; ``True`` marks PAD, which
                       is excluded from both noising and scoring.
            lambda_ce: weight λ of the auxiliary cross-entropy term.

        Returns:
            dict with scalar tensors ``{"loss", "vb", "ce"}``; backprop through
            ``["loss"]``. ``["vb"]`` and ``["ce"]`` are detached-value diagnostics
            (still on the graph, but you should only call ``.backward()`` on
            ``["loss"]``).
        """
        if x0.dim() != 2:
            raise ValueError(f"x0 must be 2D (B, L), got shape {tuple(x0.shape)}")

        B = x0.shape[0]
        device = x0.device

        # ── 1. sample one timestep per example, t ~ U{1, ..., T} ────────────
        # randint's high bound is exclusive, so (1, T+1) yields 1..T inclusive.
        t = torch.randint(1, self.T + 1, (B,), device=device)

        # ── 2. corrupt: x_t ~ q(x_t | x_0) ──────────────────────────────────
        x_t = self.q_sample(x0, t, pad_mask=pad_mask)

        # ── 3. predict clean-token logits p̃_θ(x_0 | x_t) ───────────────────
        logits = model(x_t, t)                              # (B, L, vocab)

        # ── 4. per-position negative log-likelihood of the TRUE token ───────
        # log_softmax then gather the log-prob assigned to the real x_0 id at
        # each position. `gather(dim=-1, index=x0[...,None])` picks, for every
        # (b, l), the entry logp[b, l, x0[b, l]].
        logp = F.log_softmax(logits, dim=-1)                # (B, L, vocab)
        logp_true = logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
        nll = -logp_true                                    # (B, L), ≥ 0

        # ── 5. score only positions that are MASKED in x_t and not PAD ──────
        # Unmasked positions contribute zero KL (degenerate posterior); PAD
        # carries no signal. A position is "masked" exactly when q_sample wrote
        # mask_idx there.
        scored = x_t == self.mask_idx                       # (B, L) bool
        if pad_mask is not None:
            scored = scored & ~pad_mask                     # belt-and-suspenders
            n_norm = (~pad_mask).sum().clamp(min=1)         # non-PAD token count
        else:
            n_norm = torch.tensor(float(x0.numel()), device=device)
        scored_f = scored.float()

        # ── 6. assemble L_vb and L_CE ───────────────────────────────────────
        # 1/t weight broadcast across the length axis (per-example t).
        inv_t = (1.0 / t.float()).unsqueeze(1)              # (B, 1)
        l_vb = (scored_f * inv_t * nll).sum() / n_norm      # reweighted CE
        l_ce = (scored_f * nll).sum() / n_norm              # plain CE (aux.)

        total = l_vb + lambda_ce * l_ce
        return {"loss": total, "vb": l_vb, "ce": l_ce}

    # ── sampling: the learned reverse process (Stage 6) ──────────────────────

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        n: int,
        seq_len: int,
        device: torch.device | str | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate ``n`` opcode sequences by running the reverse chain ``t=T→1``.

        The reverse (denoising) process is the *learned* inverse of ``q_sample``:
        starting from the schedule's stationary state (an all-``[MASK]`` sequence
        at ``t = T``) we repeatedly ask the denoiser to predict the clean tokens
        and un-mask a few positions per step, until at ``t = 1`` every position
        carries a real opcode.

        The absorbing reverse posterior (paper §3.1, App. A.2.2 / A.3)
        --------------------------------------------------------------
        Because the forward process is *absorbing* (a token, once masked, stays
        masked) the reverse step only ever touches positions that are **currently
        masked** — un-masked positions are already settled and are frozen. For a
        position that is masked at step ``t``, the forward posterior
        ``q(x_{t-1} | x_t, x_0)`` un-masks it to its clean value with probability

            P(un-mask at t-1 | masked at t)
              = P(masked at t ∧ un-masked at t-1) / P(masked at t)
              = (ᾱ_{t-1} · β_t) / (1 − ᾱ_t)
              = ( (T-t+1)/T · 1/(T-t+1) ) / (t/T)
              = (1/T) / (t/T)  =  1/t,

        and leaves it masked with probability ``(t-1)/t``. We don't know the true
        ``x_0`` at sampling time, so when a position does un-mask we draw its token
        from the model's prediction ``p̃_θ(x_0 | x_t)`` (softmax of the logits).

        So each step is: predict clean-token logits; sample one candidate token per
        position; for each *still-masked* position flip a Bernoulli(``1/t``) coin and
        write the candidate where it lands heads. At the last step (``t = 1``) the
        probability is ``1/1 = 1``, so every remaining masked position un-masks and
        the output is guaranteed free of ``[MASK]``.

        We additionally forbid the model from ever proposing ``[MASK]`` or ``[PAD]``
        as a clean token (their logits are set to ``-inf`` before the softmax, every
        step). ``p̃_θ(x_0 | ·)`` is a distribution over *clean* opcodes, which never
        contains those two special tokens, so this only removes mass the model
        should already place near zero — and it makes every written token provably a
        real opcode (no MASK/PAD can survive to the output).

        Args:
            model:    the trained denoiser, called as ``model(x_t, t) -> logits``.
                      Put in ``.eval()`` here; this runs under ``@torch.no_grad()``
                      so no autograd graph is built (faster, less memory).
            n:        number of sequences to generate (batch size).
            seq_len:  length of each generated sequence (≤ the model's ``max_len``).
            device:   where to run; defaults to the schedule buffer's device.
            return_trajectory: if True, also return the list of intermediate ``x_t``
                      tensors (one per step, ``t=T..0``) for visualizing the
                      iterative refinement.

        Returns:
            LongTensor ``(n, seq_len)`` of generated token ids (no MASK/PAD), or
            ``(x_0, trajectory)`` if ``return_trajectory`` is set.
        """
        device = torch.device(device) if device is not None else self.alpha_bar.device
        model.eval()

        # x_T: every position is the absorbing [MASK] token.
        x = torch.full((n, seq_len), self.mask_idx, dtype=torch.long, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        # Walk the chain backwards: t = T, T-1, ..., 1.
        for t in range(self.T, 0, -1):
            t_batch = torch.full((n,), t, dtype=torch.long, device=device)

            # p̃_θ(x_0 | x_t): predicted distribution over the *clean* token at
            # every position. Forbid MASK/PAD so a sampled candidate is always a
            # real opcode (see docstring).
            logits = model(x, t_batch)                      # (n, L, vocab)
            logits[..., self.mask_idx] = float("-inf")
            logits[..., self.pad_idx] = float("-inf")
            probs = F.softmax(logits, dim=-1)               # (n, L, vocab)

            # torch.multinomial samples one category per row given a probability
            # row; it wants a 2-D (rows, vocab) input, so flatten the (n, L) grid
            # of positions into rows, draw one token each, then reshape back.
            candidates = torch.multinomial(
                probs.reshape(-1, probs.shape[-1]), num_samples=1
            ).reshape(n, seq_len)                            # (n, L) ids

            # Only currently-masked positions are eligible to change (absorbing).
            is_masked = x == self.mask_idx                  # (n, L) bool
            if t == 1:
                # Final step: un-mask everything that's left (prob 1/t = 1).
                unmask = is_masked
            else:
                # Bernoulli(1/t) per position: heads -> un-mask to the candidate.
                coin = torch.rand((n, seq_len), device=device) < (1.0 / t)
                unmask = is_masked & coin

            x = torch.where(unmask, candidates, x)
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x
