"""Verification of the core D3PM math against independent brute-force references.

The implementation in ``d3pm/diffusion.py`` never materializes the transition
matrices ``Q_t`` / ``Q̄_t`` nor an explicit posterior ``q(x_{t-1} | x_t, x_0)``.
Instead it uses the closed-form shortcuts the absorbing schedule allows:

    ᾱ_t = (T - t) / T              (scalar "stay original" prob, the buffer)
    P(un-mask at t-1 | masked at t, x_0) = 1/t       (used by sample())
    L_vb masked-position term      = (1/t) · (−log p̃_θ(x_0))

This script rebuilds the math the *honest* way — explicit (K+1)×(K+1) absorbing
transition matrices from the paper's β_t = 1/(T−t+1), and a full Bayes
enumeration over all states — and checks the code's shortcuts reproduce it.

Checks
------
  (1) Single-step Q_t and cumulative Q̄_t = Q_1···Q_t are valid stochastic
      matrices, and the code's ``alpha_bar`` buffer equals the Q̄_t diagonal.
  (2) Closed-form posterior q(x_{t-1} | x_t, x_0) (derived from the code's
      schedule) matches a brute-force Bayes enumeration over ALL states, for
      every (t, x_0, x_t), at K=4, T=10. Also confirms the un-mask probability
      equals the 1/t that sample() hard-uses.
  (3) The cumulative matrix at t=T converges to the absorbing stationary
      distribution (all mass on [MASK]); equivalently ᾱ_T = 0.
  (4) The training loss is non-negative on random batches and an oracle denoiser
      that outputs the true-x_0 logits drives the variational term (the KL) to
      ~0; a wrong model is clearly positive.

Run:  .venv/bin/python -m verify.verify_math
Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

from d3pm.diffusion import AbsorbingDiffusion


# ── small config for the exhaustive checks ───────────────────────────────────
K = 4            # number of real opcode ids: 0..K-1
T = 10           # diffusion steps
MASK = K         # absorbing state id  (state space is 0..K for the matrices)
PAD = K + 1      # only matters for the loss; not part of the diffusion matrices
N_STATES = K + 1  # real tokens + [MASK]

ATOL = 1e-6


# ── independent reference: explicit absorbing transition matrices ────────────
def beta(t: int) -> float:
    """Paper schedule β_t = 1/(T − t + 1) for t = 1..T (App. A.7)."""
    return 1.0 / (T - t + 1)


def Q_single(t: int) -> torch.Tensor:
    """One-step absorbing transition matrix Q_t, shape (N_STATES, N_STATES).

    Row i = current state, col j = next state.
      * a real token i stays with prob 1−β_t, jumps to [MASK] with prob β_t;
      * [MASK] is absorbing: stays with prob 1.
    """
    b = beta(t)
    Q = torch.zeros(N_STATES, N_STATES, dtype=torch.float64)
    for i in range(K):
        Q[i, i] = 1.0 - b
        Q[i, MASK] = b
    Q[MASK, MASK] = 1.0
    return Q


def Q_bar(t: int) -> torch.Tensor:
    """Cumulative Q̄_t = Q_1 Q_2 ... Q_t (Q̄_0 = identity)."""
    Qb = torch.eye(N_STATES, dtype=torch.float64)
    for s in range(1, t + 1):
        Qb = Qb @ Q_single(s)
    return Qb


def bruteforce_posterior(t: int, x0: int, xt: int) -> torch.Tensor:
    """q(x_{t-1} | x_t, x_0) by Bayes over the explicit matrices.

        q(x_{t-1}=k | x_t=j, x_0=i)
          ∝ q(x_t=j | x_{t-1}=k) · q(x_{t-1}=k | x_0=i)
          =  Q_t[k, j]          · Q̄_{t-1}[i, k]

    Returns a length-N_STATES distribution over x_{t-1}.
    """
    Qt = Q_single(t)
    Qb_prev = Q_bar(t - 1)
    unnorm = torch.tensor(
        [Qt[k, xt] * Qb_prev[x0, k] for k in range(N_STATES)],
        dtype=torch.float64,
    )
    return unnorm / unnorm.sum()


def closedform_posterior(diff: AbsorbingDiffusion, t: int, x0: int, xt: int) -> torch.Tensor:
    """Posterior implied by the CODE's schedule buffer + its reduction.

    Built only from ``diff.alpha_bar`` and β_t — i.e. the exact quantities the
    implementation relies on — so a bug in the buffer (e.g. an off-by-one) shows
    up as a mismatch with the brute-force matrices.
    """
    ab = diff.alpha_bar.double()
    p = torch.zeros(N_STATES, dtype=torch.float64)
    if xt == MASK:
        denom = 1.0 - ab[t]                     # = q(x_t=MASK | x_0)
        p_unmask = beta(t) * ab[t - 1] / denom  # -> real value x_0
        p_stay = (1.0 - ab[t - 1]) / denom      # -> stay [MASK]
        p[x0] = p_unmask
        p[MASK] = p_stay
    else:
        # x_t is already a real token: forward can't un-mask, so x_{t-1}=x_t.
        p[xt] = 1.0
    return p


# ── oracle denoiser for the loss check ───────────────────────────────────────
class OracleDenoiser(nn.Module):
    """Outputs near-one-hot logits at the true x_0 token everywhere."""

    def __init__(self, x0: torch.Tensor, vocab: int, big: float = 30.0) -> None:
        super().__init__()
        self.x0 = x0
        self.vocab = vocab
        self.big = big
        self.dummy = nn.Parameter(torch.zeros(1))  # so loss() has a param graph

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x_t.shape
        logits = torch.zeros(B, L, self.vocab) + self.dummy
        logits.scatter_(-1, self.x0.unsqueeze(-1), self.big)
        return logits


class WrongDenoiser(nn.Module):
    """Always predicts token 0 with high confidence — a deliberately bad model."""

    def __init__(self, vocab: int, big: float = 30.0) -> None:
        super().__init__()
        self.vocab = vocab
        self.big = big
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x_t.shape
        logits = torch.zeros(B, L, self.vocab) + self.dummy
        logits[..., 0] = self.big
        return logits


# ── check runners ─────────────────────────────────────────────────────────────
def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    return ok


def main() -> int:
    torch.manual_seed(0)
    diff = AbsorbingDiffusion(T=T, mask_idx=MASK, pad_idx=PAD)
    all_ok = True

    print(f"\nD3PM math verification  (K={K} real tokens, T={T}, "
          f"state space size {N_STATES} incl. [MASK])")

    # ── (1) transition matrices + schedule buffer ───────────────────────────
    print("\n(1) Transition matrices Q_t, Q̄_t and the alpha_bar buffer")
    ok = True
    for t in range(1, T + 1):
        Qt = Q_single(t)
        ok &= torch.allclose(Qt.sum(1), torch.ones(N_STATES, dtype=torch.float64), atol=ATOL)
    all_ok &= check("every Q_t is row-stochastic (rows sum to 1)", ok)

    ok = True
    for t in range(1, T + 1):
        Qb = Q_bar(t)
        ok &= torch.allclose(Qb.sum(1), torch.ones(N_STATES, dtype=torch.float64), atol=ATOL)
    all_ok &= check("every Q̄_t is row-stochastic", ok)

    # code's alpha_bar[t] must equal the real-token diagonal of Q̄_t.
    ok = True
    worst = 0.0
    for t in range(0, T + 1):
        Qb = Q_bar(t)
        for i in range(K):
            diag = Qb[i, i].item()
            code_ab = diff.alpha_bar[t].item()
            worst = max(worst, abs(diag - code_ab))
            ok &= abs(diag - code_ab) < ATOL
            # and the off-diagonal mass is all on [MASK]:
            ok &= abs(Qb[i, MASK].item() - (1.0 - code_ab)) < ATOL
    all_ok &= check(
        "alpha_bar[t] == Q̄_t diagonal for all t (schedule matches matrices)",
        ok, f"max |diff| = {worst:.2e}",
    )

    # ── (2) closed-form posterior vs brute-force enumeration ────────────────
    print("\n(2) Posterior q(x_{t-1} | x_t, x_0): closed-form vs brute-force Bayes")
    ok = True
    worst = 0.0
    n_cases = 0
    n_skipped = 0
    for t in range(1, T + 1):
        Qb_t = Q_bar(t)
        for x0 in range(K):                 # x_0 is always a real token
            for xt in (x0, MASK):           # candidate x_t: stayed real, or masked
                # The posterior is only defined where the conditioning event is
                # possible: q(x_t | x_0) = Q̄_t[x0, xt] > 0. At t=T no real token
                # survives (ᾱ_T = 0), so (xt = real) is unreachable and skipped.
                if Qb_t[x0, xt].item() < ATOL:
                    n_skipped += 1
                    continue
                bf = bruteforce_posterior(t, x0, xt)
                cf = closedform_posterior(diff, t, x0, xt)
                d = (bf - cf).abs().max().item()
                worst = max(worst, d)
                ok &= d < ATOL
                n_cases += 1
    all_ok &= check(
        f"closed-form == brute-force over all {n_cases} reachable (t, x_0, x_t) cases",
        ok, f"max |diff| = {worst:.2e}; skipped {n_skipped} zero-probability states",
    )

    # the un-mask probability the code's sample() hard-codes is exactly 1/t.
    ok = True
    worst = 0.0
    for t in range(1, T + 1):
        bf = bruteforce_posterior(t, x0=0, xt=MASK)   # mass on the true token
        d = abs(bf[0].item() - 1.0 / t)
        worst = max(worst, d)
        ok &= d < ATOL
    all_ok &= check(
        "brute-force P(un-mask | masked) == 1/t  (the rule sample() uses)",
        ok, f"max |diff| = {worst:.2e}",
    )

    # ── (3) stationary distribution at t=T ──────────────────────────────────
    print("\n(3) Cumulative matrix at t=T converges to the absorbing stationary dist.")
    Qb_T = Q_bar(T)
    expected = torch.zeros(N_STATES, dtype=torch.float64)
    expected[MASK] = 1.0
    ok = True
    for i in range(N_STATES):
        ok &= torch.allclose(Qb_T[i], expected, atol=ATOL)  # every row -> δ_MASK
    all_ok &= check(
        "every row of Q̄_T is the point mass on [MASK]", ok,
        f"Q̄_T[real-token row] = {Qb_T[0].tolist()}",
    )
    all_ok &= check(
        "code's alpha_bar[T] == 0 (no real token survives)", True,
        f"alpha_bar[T] = {diff.alpha_bar[T].item():.1f}",
    ) if abs(diff.alpha_bar[T].item()) < ATOL else check(
        "code's alpha_bar[T] == 0 (no real token survives)", False,
        f"alpha_bar[T] = {diff.alpha_bar[T].item()}",
    )

    # ── (4) loss non-negativity + near-zero KL for the oracle ───────────────
    print("\n(4) Training loss: non-negativity and oracle -> ~0 variational term")
    vocab = K + 2          # real + MASK + PAD
    B, L = 16, 48
    diff_loss = AbsorbingDiffusion(T=T, mask_idx=MASK, pad_idx=PAD)

    # 4a. non-negative over many random batches / timesteps.
    ok = True
    min_loss = float("inf")
    for _ in range(200):
        x0 = torch.randint(0, K, (B, L))
        model = WrongDenoiser(vocab)
        out = diff_loss.loss(model, x0, pad_mask=None)
        for key in ("loss", "vb", "ce"):
            v = out[key].item()
            min_loss = min(min_loss, v)
            ok &= v >= -ATOL
    all_ok &= check("loss/vb/ce >= 0 over 200 random batches", ok,
                    f"min observed = {min_loss:.3e}")

    # 4b. oracle (true-x_0 logits) drives the variational/KL term to ~0.
    torch.manual_seed(1)
    vb_vals = []
    for _ in range(50):
        x0 = torch.randint(0, K, (B, L))
        oracle = OracleDenoiser(x0, vocab)
        out = diff_loss.loss(oracle, x0, pad_mask=None)
        vb_vals.append(out["vb"].item())
    max_vb = max(vb_vals)
    all_ok &= check("oracle x_0-logits -> variational term ~0 (near-zero KL)",
                    max_vb < 1e-10, f"max L_vb over 50 batches = {max_vb:.2e}")

    # 4c. a wrong model is clearly positive (sanity that the metric can fire).
    torch.manual_seed(2)
    x0 = torch.randint(1, K, (B, L))   # avoid token 0 so WrongDenoiser is wrong
    wrong_vb = diff_loss.loss(WrongDenoiser(vocab), x0, pad_mask=None)["vb"].item()
    all_ok &= check("wrong model -> variational term clearly > 0", wrong_vb > 0.5,
                    f"L_vb = {wrong_vb:.3f}")

    print()
    if all_ok:
        print("ALL CHECKS PASSED ✅")
        return 0
    print("SOME CHECKS FAILED ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())
