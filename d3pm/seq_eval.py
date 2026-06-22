"""
Stage 7 — sequence-level evaluation: is the synthetic opcode stream *realistic*?

Where this fits
---------------
Stages 5–6 trained a denoiser and sampled clean opcode sequences into
``synthetic/<family>/seq_*.txt``. But "no [MASK]/[PAD] survived" only proves the
*plumbing* works — it says nothing about whether the generated opcodes look like
real malware. Stage 7 answers that with three corpus-level metrics that compare
the real family against the synthetic set:

  1. **Opcode-frequency KL** — do the two corpora use opcodes in the same
     proportions? Treat each corpus as a unigram distribution over the shared
     opcode set and measure KL divergence. Lower = closer. (Eq.-free: this is a
     plain distributional distance, not part of the D3PM bound.)
  2. **N-gram precision / recall / F1** for n = 1, 2, 3 — do short opcode
     *patterns* (`push;mov;call`) recur in both? Captures local sequential
     structure that a unigram histogram is blind to.
  3. **Edit-distance distribution** — token-level Levenshtein between sampled
     pairs. Tells us how far a synthetic sequence is from real ones, and (via the
     synth↔synth pairing) whether the generator merely memorised / collapsed.

Why baselines matter
--------------------
A bare "symmetric KL = 0.07" means nothing in isolation. Every metric here is
reported alongside two reference points so the number is interpretable as a
position on a band, not an absolute:

  * **upper bound (best achievable)** — *real vs real*: split the real corpus in
    half and compare the halves. This is how close two genuine samples of the
    same family get; the model cannot reasonably beat it.
  * **lower bound (learned nothing)** — *real vs random*: uniform-random opcode
    sequences over the real opcode set, matched to the synthetic length
    distribution. This is what the metric looks like with zero structure learned.

A good model lands close to the real-vs-real bound and far from the random one.

Run it:

    .venv/bin/python -m d3pm.seq_eval --family cleaman

(by default it reads real opcodes from ``malicia/<family>/`` and synthetic ones
from ``synthetic/<family>/seq_*.txt`` — the files ``d3pm.generate`` writes.)
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


# ── 1. Opcode-frequency KL divergence ────────────────────────────────────────


def opcode_freq_kl(
    real: list[list[str]],
    synthetic: list[list[str]],
    eps: float = 1e-9,
) -> dict[str, float]:
    """Unigram KL divergence between the real and synthetic opcode distributions.

    Flatten each corpus to a bag of opcodes, normalise the counts into
    probability vectors ``p`` (real) and ``q`` (synthetic) over the *union* of
    opcodes seen in either corpus, then::

        KL(p‖q) = Σ_i p_i · log(p_i / q_i)

    KL is asymmetric (and infinite if some opcode has p>0 but q=0), so we add a
    tiny ``eps`` floor to every probability before renormalising — standard
    additive smoothing so an opcode the model never emitted doesn't blow the
    divergence up to infinity. We report both directions and their average; the
    symmetric value is the convenient single summary. Lower = the corpora use
    opcodes in more similar proportions.
    """
    real_counts = Counter(op for seq in real for op in seq)
    synth_counts = Counter(op for seq in synthetic for op in seq)

    opcodes = sorted(set(real_counts) | set(synth_counts))
    real_total = sum(real_counts.values()) or 1
    synth_total = sum(synth_counts.values()) or 1

    p = np.array([real_counts.get(op, 0) / real_total for op in opcodes], dtype=float)
    q = np.array([synth_counts.get(op, 0) / synth_total for op in opcodes], dtype=float)

    # Additive smoothing so zeros don't make log(p/q) explode, then renormalise.
    p = p + eps
    p /= p.sum()
    q = q + eps
    q /= q.sum()

    kl_rs = float(np.sum(p * np.log(p / q)))
    kl_sr = float(np.sum(q * np.log(q / p)))
    return {
        "kl_real_vs_synth": kl_rs,
        "kl_synth_vs_real": kl_sr,
        "symmetric_kl": (kl_rs + kl_sr) / 2,
        "vocab_real": len(real_counts),
        "vocab_synth": len(synth_counts),
        "vocab_overlap": len(set(real_counts) & set(synth_counts)),
    }


# ── 2. N-gram overlap ─────────────────────────────────────────────────────────


def _ngrams(seq: list[str], n: int) -> Counter:
    """Count the n-grams (length-``n`` contiguous opcode tuples) in one sequence."""
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def ngram_overlap(
    real: list[list[str]],
    synthetic: list[list[str]],
    ns: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    """Corpus-level n-gram precision / recall / F1 over n-gram *types* (sets).

    For each ``n`` we pool the n-grams across all sequences in each corpus and
    compare the *sets* of distinct n-grams that appear:

        precision = |real ∩ synth| / |synth|   (what fraction of the model's
                                                 n-grams are ones real malware uses)
        recall    = |real ∩ synth| / |real|    (what fraction of real n-grams the
                                                 model managed to reproduce)
        F1        = harmonic mean

    n=1 just re-checks vocabulary coverage; n=2,3 are where genuine sequential
    structure shows up (random opcodes share unigrams with real but almost no
    bigrams/trigrams). We use type sets rather than multisets so a few very
    common patterns don't dominate the score.
    """
    results: dict[str, float] = {}
    for n in ns:
        real_ngrams: Counter = sum((_ngrams(s, n) for s in real), Counter())
        synth_ngrams: Counter = sum((_ngrams(s, n) for s in synthetic), Counter())
        real_types = set(real_ngrams)
        synth_types = set(synth_ngrams)
        overlap = real_types & synth_types
        precision = len(overlap) / max(len(synth_types), 1)
        recall = len(overlap) / max(len(real_types), 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        results[f"{n}gram_precision"] = float(precision)
        results[f"{n}gram_recall"] = float(recall)
        results[f"{n}gram_f1"] = float(f1)
    return results


# ── 3. Edit-distance distribution ─────────────────────────────────────────────


def _levenshtein(a: list[str], b: list[str], max_dist: int = 200) -> int:
    """Token-level Levenshtein (edit) distance, truncating inputs for speed.

    The classic two-row dynamic program: ``prev[j]`` holds the edit distance
    between the first ``i-1`` tokens of ``a`` and the first ``j`` tokens of ``b``;
    each cell is the min cost of a delete / insert / (sub-or-match). Token-level,
    so the unit of edit is one whole opcode. Inputs are capped at ``max_dist``
    opcodes because the DP is O(len(a)·len(b)) and we sample many pairs — the cap
    keeps the comparison cheap while still reflecting how similar the prefixes are.
    """
    a = a[:max_dist]
    b = b[:max_dist]
    if len(a) < len(b):  # keep the inner loop over the shorter sequence
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,            # delete ca
                    curr[j - 1] + 1,        # insert cb
                    prev[j - 1] + (0 if ca == cb else 1),  # substitute / match
                )
            )
        prev = curr
    return prev[-1]


def edit_distance_stats(
    real: list[list[str]],
    synthetic: list[list[str]],
    n_pairs: int = 200,
    seed: int = 42,
    max_dist: int = 200,
) -> dict[str, dict[str, float]]:
    """Sampled pairwise edit distances: real↔synth, plus real↔real & synth↔synth.

    Computing every real×synthetic pair is quadratic and unnecessary; we
    Monte-Carlo it by sampling ``n_pairs`` random pairs (fixed seed → reproducible)
    and summarising the distances. Three pairings, each informative:

      * ``real_vs_synthetic`` — how far synthetic sequences sit from real ones.
      * ``real_vs_real``      — the natural spread *within* the real family (the
                                floor: synthetic can't be closer to real than real
                                is to itself).
      * ``synth_vs_synth``    — diversity of the generator. If this collapses near
                                zero the model is emitting near-duplicates (mode
                                collapse / memorisation), even if real↔synth looks fine.
    """
    rng = random.Random(seed)

    def _sample_pairs(A: list[list[str]], B: list[list[str]], k: int) -> list[int]:
        return [_levenshtein(rng.choice(A), rng.choice(B), max_dist) for _ in range(k)]

    def _stats(dists: list[int]) -> dict[str, float]:
        arr = np.array(dists, dtype=float)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
        }

    return {
        "real_vs_synthetic": _stats(_sample_pairs(real, synthetic, n_pairs)),
        "real_vs_real": _stats(_sample_pairs(real, real, n_pairs)),
        "synth_vs_synth": _stats(_sample_pairs(synthetic, synthetic, n_pairs)),
    }


# ── 4. Baselines (interpretability band) ──────────────────────────────────────


def split_halves(
    seqs: list[list[str]], seed: int = 0
) -> tuple[list[list[str]], list[list[str]]]:
    """Shuffle then split a corpus in two — the *real-vs-real* upper-bound source."""
    shuffled = list(seqs)
    random.Random(seed).shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


def random_like(
    reference: list[list[str]],
    opcode_vocab: Sequence[str],
    seed: int = 0,
) -> list[list[str]]:
    """Uniform-random opcode sequences — the *learned-nothing* lower-bound source.

    One random sequence per ``reference`` sequence, each the same length as its
    reference (so edit distances stay comparable) and each opcode drawn uniformly
    from ``opcode_vocab`` (the real family's opcode set). A real model should beat
    this comfortably on every metric; if it doesn't, it learned no structure.
    """
    rng = random.Random(seed)
    vocab = list(opcode_vocab)
    return [[rng.choice(vocab) for _ in range(len(seq))] for seq in reference]


# ── Full sequence evaluation ──────────────────────────────────────────────────


def _metric_block(
    real: list[list[str]],
    other: list[list[str]],
    n_edit_pairs: int,
    seed: int,
) -> dict:
    """Run all three metrics for one ``real`` vs ``other`` comparison."""
    return {
        "freq_kl": opcode_freq_kl(real, other),
        "ngram_overlap": ngram_overlap(real, other),
        "edit_distance": edit_distance_stats(real, other, n_pairs=n_edit_pairs, seed=seed),
    }


def evaluate(
    family: str,
    real: list[list[str]],
    synthetic: list[list[str]],
    *,
    n_edit_pairs: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Full Stage-7 report: synthetic metrics framed by upper/lower baselines.

    Returns a nested dict with three comparison blocks — ``synthetic`` (the model,
    real vs synthetic), ``baseline_real_vs_real`` (upper bound), and
    ``baseline_real_vs_random`` (lower bound) — each holding ``freq_kl``,
    ``ngram_overlap`` and ``edit_distance``. When ``verbose`` it also prints a
    compact side-by-side table so the three numbers can be read at a glance.
    """
    real_opcodes = sorted({op for seq in real for op in seq})
    real_a, real_b = split_halves(real, seed=seed)
    random_seqs = random_like(synthetic, real_opcodes, seed=seed)

    report = {
        "family": family,
        "n_real": len(real),
        "n_synthetic": len(synthetic),
        "synthetic": _metric_block(real, synthetic, n_edit_pairs, seed),
        # Upper bound: one real half vs the other.
        "baseline_real_vs_real": _metric_block(real_a, real_b, n_edit_pairs, seed),
        # Lower bound: real vs uniform-random opcodes.
        "baseline_real_vs_random": _metric_block(real, random_seqs, n_edit_pairs, seed),
    }

    if verbose:
        _print_report(report)
    return report


def _print_report(report: dict) -> None:
    """Pretty side-by-side print: model vs the two baselines, metric by metric."""
    fam = report["family"]
    model = report["synthetic"]
    upper = report["baseline_real_vs_real"]
    lower = report["baseline_real_vs_random"]

    print(f"\n[{fam}] Stage-7 sequence evaluation")
    print(f"  real={report['n_real']} seqs   synthetic={report['n_synthetic']} seqs")
    print(f"\n  {'metric':<22}{'synthetic':>12}{'real↔real':>12}{'real↔random':>14}")
    print(f"  {'(want →)':<22}{'≈ left':>12}{'(best)':>12}{'(worst)':>14}")
    print("  " + "-" * 58)

    def row(label: str, m: float, u: float, l: float, fmt: str = ".4f") -> None:
        print(f"  {label:<22}{m:>12{fmt}}{u:>12{fmt}}{l:>14{fmt}}")

    row(
        "symmetric KL",
        model["freq_kl"]["symmetric_kl"],
        upper["freq_kl"]["symmetric_kl"],
        lower["freq_kl"]["symmetric_kl"],
    )
    for n in (1, 2, 3):
        key = f"{n}gram_f1"
        row(
            f"{n}-gram F1",
            model["ngram_overlap"][key],
            upper["ngram_overlap"][key],
            lower["ngram_overlap"][key],
            fmt=".3f",
        )
    row(
        "edit dist (real↔X mean)",
        model["edit_distance"]["real_vs_synthetic"]["mean"],
        upper["edit_distance"]["real_vs_synthetic"]["mean"],
        lower["edit_distance"]["real_vs_synthetic"]["mean"],
        fmt=".1f",
    )

    fk = model["freq_kl"]
    print(
        f"\n  vocab: {fk['vocab_overlap']}/{fk['vocab_real']} real opcodes appear "
        f"in synthetic (synth uses {fk['vocab_synth']})"
    )
    ss = model["edit_distance"]["synth_vs_synth"]["mean"]
    print(f"  synth↔synth edit dist (diversity): {ss:.1f}  (near 0 ⇒ mode collapse)")


# ── Loading synthetic sequences from disk ─────────────────────────────────────


def load_synthetic(synthetic_dir: Path | str, family: str) -> list[list[str]]:
    """Read ``synthetic/<family>/seq_*.txt`` (the format ``d3pm.generate`` writes)."""
    seq_dir = Path(synthetic_dir) / family
    if not seq_dir.is_dir():
        raise FileNotFoundError(
            f"No synthetic sequences at {seq_dir} — run `d3pm.generate` first."
        )
    seqs = [
        [line.strip() for line in p.read_text().splitlines() if line.strip()]
        for p in sorted(seq_dir.glob("seq_*.txt"))
    ]
    seqs = [s for s in seqs if s]
    if not seqs:
        raise FileNotFoundError(f"No non-empty seq_*.txt files found in {seq_dir}.")
    return seqs


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Sequence-level evaluation of synthetic vs real opcode streams."
    )
    ap.add_argument("--family", required=True, help="family to evaluate")
    ap.add_argument("--malicia", default="malicia", help="root dir of real family folders")
    ap.add_argument("--synthetic", default="synthetic", help="root dir of synthetic/<family>/")
    ap.add_argument("--out-dir", default="eval_results", help="where to write the JSON report")
    ap.add_argument("--max-files", type=int, default=None, help="cap real files read")
    ap.add_argument("--n-pairs", type=int, default=200, help="sampled pairs for edit distance")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (pairs, splits, random baseline)")
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    real = load_family_opcodes(args.malicia, args.family, max_files=args.max_files)
    if not real:
        raise SystemExit(f"No real sequences for family '{args.family}' under {args.malicia}")
    synthetic = load_synthetic(args.synthetic, args.family)

    report = evaluate(
        args.family, real, synthetic, n_edit_pairs=args.n_pairs, seed=args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.family}_seq_eval.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
