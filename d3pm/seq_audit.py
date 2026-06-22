"""
Deeper sequence diagnostics: memorization/novelty audit + transition-matrix KL.

These extend Stage-7 (`seq_eval.py`) with two questions a generative-model paper
must answer but corpus-level KL / n-gram overlap cannot:

  1. **Memorization / novelty audit.** Is the model *copying* training sequences
     rather than generating new ones? For every synthetic window we measure its
     similarity to its NEAREST training window (max k-gram Jaccard) plus exact-copy
     detection. The honest test isn't "is synthetic far from training" in absolute
     terms — a genuine new real sample also shares lots of k-grams with the corpus.
     So we compute the SAME nearest-neighbor similarity for a held-out set of REAL
     windows and compare: if synthetic is no closer to the training set than a real
     held-out sample is, the model generalizes rather than memorizes.

  2. **Transition-matrix KL (Tier 3 structural fidelity).** Unigram KL ignores
     order; n-gram F1 only checks set membership. The bigram *transition matrix*
     P(next | current) is the Markov structure of the opcode stream. We compare the
     real and synthetic transition distributions via KL averaged over current-opcode
     rows, weighted by how often each opcode actually occurs. Framed (like seq_eval)
     against a real-vs-real floor and a real-vs-random ceiling.

Both metrics are pure-numpy/stdlib (main `.venv`, no sklearn). Jaccard k-gram sets
are used for the audit instead of full edit-distance NN because edit distance over
hundreds of synthetic × thousands of real windows is intractable in Python, whereas
k-gram set overlap is the standard, fast memorization signal (used in LLM
memorization studies) and captures near-duplication just as well.

Run it:

    .venv/bin/python -m d3pm.seq_audit --family winwebsec \
        --malicia malicia --synthetic synthetic
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from d3pm.data import load_family_opcodes
from d3pm.seq_eval import load_synthetic, split_halves, random_like


# ── windowing (make real files and synthetic windows length-comparable) ───────


def chunk_corpus(
    seqs: list[list[str]], window: int, min_chunk: int
) -> list[list[str]]:
    """Cut each sequence into non-overlapping `window`-length windows (drop a tail
    shorter than `min_chunk`). Real files run to thousands of opcodes; synthetic are
    fixed-length, so we window both to compare like with like."""
    out = []
    for seq in seqs:
        for start in range(0, len(seq), window):
            chunk = seq[start : start + window]
            if len(chunk) >= min_chunk:
                out.append(chunk)
    return out


# ── 1. Memorization / novelty audit ──────────────────────────────────────────


def _kgram_set(seq: list[str], k: int) -> frozenset:
    """Set of length-k contiguous opcode tuples in a window (its 'shingles')."""
    if len(seq) < k:
        return frozenset([tuple(seq)])
    return frozenset(tuple(seq[i : i + k]) for i in range(len(seq) - k + 1))


def _max_jaccard(query: frozenset, reference_sets: list[frozenset]) -> float:
    """Largest Jaccard overlap between `query`'s shingles and any reference window."""
    best = 0.0
    for ref in reference_sets:
        inter = len(query & ref)
        if inter == 0:
            continue
        j = inter / len(query | ref)
        if j > best:
            best = j
            if best == 1.0:
                break
    return best


def novelty_audit(
    real_windows: list[list[str]],
    synth_windows: list[list[str]],
    *,
    k: int = 4,
    ref_cap: int = 2000,
    query_cap: int = 500,
    near_dup_thresh: float = 0.9,
    seed: int = 0,
) -> dict:
    """Nearest-training-neighbor similarity for synthetic vs a held-out-real baseline.

    For each synthetic window: max k-gram Jaccard to a (subsampled) pool of REAL
    training windows, and whether it exactly equals any training window. Then the
    same measurement using held-out REAL windows as queries against the *rest* of
    the real pool — this is the natural self-similarity of genuine samples and the
    reference the synthetic numbers must be read against.

    Verdict: synthetic max-Jaccard ≈ (not meaningfully above) the real-holdout
    baseline ⇒ the model generalizes; exact-copy rate near 0 ⇒ no rote copying.
    """
    rng = random.Random(seed)

    real = list(real_windows)
    rng.shuffle(real)
    # Split real into a reference "training" pool and a held-out query set.
    n_hold = min(query_cap, len(real) // 2)
    real_hold = real[:n_hold]
    real_pool = real[n_hold:]

    # Exact-copy detection compares against the TRAINING side only (real_pool), so
    # a held-out real query isn't counted as copying itself. Cheap: hashable tuples.
    train_tuples = {tuple(w) for w in real_pool}

    # Subsample the reference pool the Jaccard scan compares against (for speed).
    ref_pool = real_pool if len(real_pool) <= ref_cap else rng.sample(real_pool, ref_cap)
    ref_sets = [_kgram_set(w, k) for w in ref_pool]

    def _scan(queries: list[list[str]]) -> dict:
        qs = queries if len(queries) <= query_cap else rng.sample(queries, query_cap)
        sims, exact, near = [], 0, 0
        for w in qs:
            if tuple(w) in train_tuples:
                exact += 1
            j = _max_jaccard(_kgram_set(w, k), ref_sets)
            sims.append(j)
            if j >= near_dup_thresh:
                near += 1
        arr = np.array(sims, dtype=float)
        return {
            "n_queries": len(qs),
            "max_jaccard_mean": float(arr.mean()),
            "max_jaccard_median": float(np.median(arr)),
            "max_jaccard_p95": float(np.percentile(arr, 95)),
            "exact_copy_rate": exact / max(len(qs), 1),
            "near_dup_rate": near / max(len(qs), 1),
        }

    synth_stats = _scan(synth_windows)
    real_stats = _scan(real_hold)
    gap = synth_stats["max_jaccard_mean"] - real_stats["max_jaccard_mean"]
    # Memorization is judged RELATIVE to the corpus's own duplication: malware
    # families are highly templated, so genuine held-out windows already hit high
    # exact-copy / Jaccard rates. The model is only memorizing if it sits CLOSER to
    # the training set than a genuine new sample does — i.e. positive Jaccard gap or
    # an exact-copy rate above the real-holdout baseline.
    return {
        "k": k,
        "synthetic_vs_train": synth_stats,
        "real_holdout_vs_train": real_stats,
        "mean_jaccard_gap": gap,  # >0 means synth sits closer to train than real does
        "memorizing": bool(
            gap > 0.1 or synth_stats["exact_copy_rate"] > real_stats["exact_copy_rate"]
        ),
    }


# ── 2. Transition-matrix KL (bigram Markov structure) ─────────────────────────


def _transition_matrix(
    seqs: list[list[str]], vocab: list[str], eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Row-stochastic bigram transition matrix P[i,j] = P(next=j | curr=i) and the
    unigram marginal over `vocab`. Smoothed by `eps` so empty rows / unseen
    transitions don't produce log(0) in the KL."""
    idx = {op: i for i, op in enumerate(vocab)}
    K = len(vocab)
    counts = np.zeros((K, K), dtype=float)
    unigram = np.zeros(K, dtype=float)
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            ia, ib = idx.get(a), idx.get(b)
            if ia is not None and ib is not None:
                counts[ia, ib] += 1.0
        for op in seq:
            i = idx.get(op)
            if i is not None:
                unigram[i] += 1.0
    P = counts + eps
    P /= P.sum(axis=1, keepdims=True)
    unigram = unigram + eps
    unigram /= unigram.sum()
    return P, unigram


def transition_kl(
    real: list[list[str]],
    synthetic: list[list[str]],
    eps: float = 1e-6,
) -> dict[str, float]:
    """KL between the real and synthetic bigram transition distributions.

    Per current-opcode row i we take KL(P_real[i] ‖ P_synth[i]); we then average the
    rows weighted by how often opcode i actually occurs (the real unigram marginal),
    so transitions out of common opcodes count more than rare ones. Reported both
    directions plus the symmetric mean. Lower = the synthetic stream follows the same
    opcode-to-opcode dynamics as real malware (structure a unigram KL can't see)."""
    vocab = sorted({op for seq in real + synthetic for op in seq})
    Pr, pi_r = _transition_matrix(real, vocab, eps)
    Ps, pi_s = _transition_matrix(synthetic, vocab, eps)

    def weighted_kl(P: np.ndarray, Q: np.ndarray, w: np.ndarray) -> float:
        row_kl = np.sum(P * np.log(P / Q), axis=1)  # KL per row
        return float(np.sum(w * row_kl))

    kl_rs = weighted_kl(Pr, Ps, pi_r)
    kl_sr = weighted_kl(Ps, Pr, pi_s)
    return {
        "transition_kl_real_vs_synth": kl_rs,
        "transition_kl_synth_vs_real": kl_sr,
        "transition_symmetric_kl": (kl_rs + kl_sr) / 2,
        "vocab_size": len(vocab),
    }


# ── full audit report ─────────────────────────────────────────────────────────


def audit(
    family: str,
    real: list[list[str]],
    synthetic: list[list[str]],
    *,
    window: int = 256,
    min_chunk: int = 32,
    k: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Combined report: novelty audit + transition KL, each framed by baselines."""
    real_windows = chunk_corpus(real, window, min_chunk)
    real_opcodes = sorted({op for seq in real for op in seq})

    # Transition KL with the same real-vs-real / real-vs-random framing as seq_eval.
    real_a, real_b = split_halves(real, seed=seed)
    random_seqs = random_like(synthetic, real_opcodes, seed=seed)

    report = {
        "family": family,
        "n_real_windows": len(real_windows),
        "n_synthetic": len(synthetic),
        "novelty": novelty_audit(real_windows, synthetic, k=k, seed=seed),
        "transition_kl": {
            "synthetic": transition_kl(real, synthetic),
            "baseline_real_vs_real": transition_kl(real_a, real_b),
            "baseline_real_vs_random": transition_kl(real, random_seqs),
        },
    }
    if verbose:
        _print_audit(report)
    return report


def _print_audit(report: dict) -> None:
    fam = report["family"]
    nov = report["novelty"]
    s, r = nov["synthetic_vs_train"], nov["real_holdout_vs_train"]
    print(f"\n[{fam}] memorization / novelty audit  (k={nov['k']}-gram Jaccard NN)")
    print(f"  {'':<26}{'synthetic':>12}{'real-holdout':>14}")
    print("  " + "-" * 52)
    print(f"  {'max-Jaccard mean':<26}{s['max_jaccard_mean']:>12.3f}{r['max_jaccard_mean']:>14.3f}")
    print(f"  {'max-Jaccard p95':<26}{s['max_jaccard_p95']:>12.3f}{r['max_jaccard_p95']:>14.3f}")
    print(f"  {'exact-copy rate':<26}{s['exact_copy_rate']:>12.3f}{r['exact_copy_rate']:>14.3f}")
    print(f"  {'near-dup rate (J≥0.9)':<26}{s['near_dup_rate']:>12.3f}{r['near_dup_rate']:>14.3f}")
    if nov["memorizing"]:
        verdict = "MEMORIZING (synth sits as close to train as copies would)"
    elif nov["mean_jaccard_gap"] < -0.1:
        verdict = "no memorization (synth is MORE novel than genuine samples are to each other)"
    else:
        verdict = "generalizes (synth ≈ a genuine new sample)"
    print(f"  gap(synth-real) = {nov['mean_jaccard_gap']:+.3f}  → {verdict}")

    tk = report["transition_kl"]
    print(f"\n[{fam}] transition-matrix KL  (bigram Markov structure)")
    print(f"  {'':<26}{'synthetic':>12}{'real↔real':>12}{'real↔random':>14}")
    print("  " + "-" * 64)
    print(
        f"  {'symmetric transition KL':<26}"
        f"{tk['synthetic']['transition_symmetric_kl']:>12.4f}"
        f"{tk['baseline_real_vs_real']['transition_symmetric_kl']:>12.4f}"
        f"{tk['baseline_real_vs_random']['transition_symmetric_kl']:>14.4f}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Memorization audit + transition-matrix KL.")
    ap.add_argument("--family", required=True)
    ap.add_argument("--malicia", default="malicia")
    ap.add_argument("--synthetic", default="synthetic")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--min-chunk", type=int, default=32)
    ap.add_argument("--k", type=int, default=4, help="k-gram shingle size for Jaccard")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    real = load_family_opcodes(args.malicia, args.family, max_files=args.max_files)
    if not real:
        raise SystemExit(f"No real sequences for '{args.family}' under {args.malicia}")
    synthetic = load_synthetic(args.synthetic, args.family)

    report = audit(
        args.family, real, synthetic,
        window=args.window, min_chunk=args.min_chunk, k=args.k, seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.family}_seq_audit.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
