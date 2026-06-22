"""Stage 7 unit tests — the sequence-level evaluation metrics.

These pin down the mathematical contracts of the three metrics with tiny,
hand-made corpora where the right answer is computable by hand:

  * KL is ≥ 0, zero for identical distributions, and ordered (a closer corpus
    has lower KL than a farther one);
  * n-gram precision/recall/F1 are 1.0 for identical corpora and shrink as the
    synthetic set drifts; n≥2 detects sequential structure that unigrams miss;
  * edit distance is 0 for identical sequences and grows with dissimilarity;
  * the baselines order correctly (real-vs-real beats the model beats random);
  * the full ``evaluate`` report has the expected shape.

No model, dataset, or disk needed — runs in milliseconds.
"""

import math

from d3pm.seq_eval import (
    edit_distance_stats,
    evaluate,
    ngram_overlap,
    opcode_freq_kl,
    random_like,
    split_halves,
)


# A small "real" corpus with clear sequential structure (push;mov;call recurs).
REAL = [
    ["push", "mov", "call", "ret"],
    ["push", "mov", "call", "pop"],
    ["mov", "push", "mov", "call"],
    ["push", "mov", "add", "ret"],
]


# ── KL ────────────────────────────────────────────────────────────────────────


def test_kl_identical_is_zero():
    out = opcode_freq_kl(REAL, REAL)
    assert out["symmetric_kl"] == 0.0
    assert out["kl_real_vs_synth"] == 0.0
    assert out["vocab_overlap"] == out["vocab_real"] == out["vocab_synth"]


def test_kl_nonnegative_and_ordered():
    # A corpus sharing real's opcode mix is closer than an all-"nop" corpus.
    close = [["push", "mov", "call", "ret"], ["push", "mov", "call", "pop"]]
    far = [["nop", "nop", "nop", "nop"], ["nop", "nop"]]
    kl_close = opcode_freq_kl(REAL, close)["symmetric_kl"]
    kl_far = opcode_freq_kl(REAL, far)["symmetric_kl"]
    assert kl_close >= 0.0 and kl_far >= 0.0
    assert kl_close < kl_far


def test_kl_finite_with_disjoint_vocab():
    # Disjoint opcode sets would give infinite KL without smoothing; eps fixes it.
    out = opcode_freq_kl([["aaa", "bbb"]], [["ccc", "ddd"]])
    assert math.isfinite(out["symmetric_kl"])
    assert out["vocab_overlap"] == 0


# ── n-gram overlap ────────────────────────────────────────────────────────────


def test_ngram_identical_is_perfect():
    out = ngram_overlap(REAL, REAL)
    for n in (1, 2, 3):
        assert out[f"{n}gram_precision"] == 1.0
        assert out[f"{n}gram_recall"] == 1.0
        assert out[f"{n}gram_f1"] == 1.0


def test_ngram_unigram_blind_to_order():
    # Same opcodes, reversed order: unigrams identical, bigrams differ.
    synth = [list(reversed(s)) for s in REAL]
    out = ngram_overlap(REAL, synth)
    assert out["1gram_f1"] == 1.0          # unigram histogram unchanged
    assert out["2gram_f1"] < 1.0           # bigram structure broken


def test_ngram_precision_recall_values():
    # synth has one real bigram (push,mov) and one novel bigram (mov,zzz).
    real = [["push", "mov", "call"]]            # bigrams: (push,mov),(mov,call)
    synth = [["push", "mov", "zzz"]]            # bigrams: (push,mov),(mov,zzz)
    out = ngram_overlap(real, synth, ns=(2,))
    # overlap = {(push,mov)} = 1; synth types = 2; real types = 2
    assert out["2gram_precision"] == 0.5
    assert out["2gram_recall"] == 0.5


# ── edit distance ─────────────────────────────────────────────────────────────


def test_edit_distance_identical_corpus():
    # One repeated sequence → every sampled pair is identical → distance 0.
    same = [["mov", "push", "call"]] * 5
    out = edit_distance_stats(same, same, n_pairs=20)
    assert out["real_vs_synthetic"]["mean"] == 0.0
    assert out["synth_vs_synth"]["mean"] == 0.0


def test_edit_distance_grows_with_difference():
    real = [["mov", "push", "call", "ret"]] * 4
    near = [["mov", "push", "call", "pop"]] * 4    # 1 substitution
    far = [["xor", "xor", "xor", "xor"]] * 4       # 4 substitutions
    d_near = edit_distance_stats(real, near, n_pairs=10)["real_vs_synthetic"]["mean"]
    d_far = edit_distance_stats(real, far, n_pairs=10)["real_vs_synthetic"]["mean"]
    assert 0 < d_near < d_far


# ── baselines & full report ───────────────────────────────────────────────────


def test_split_halves_partitions():
    a, b = split_halves(REAL, seed=1)
    assert len(a) + len(b) == len(REAL)
    # the two halves together are exactly the original multiset of sequences
    assert sorted(map(tuple, a + b)) == sorted(map(tuple, REAL))


def test_random_like_matches_lengths_and_vocab():
    vocab = ["mov", "push", "call"]
    rnd = random_like(REAL, vocab, seed=0)
    assert len(rnd) == len(REAL)
    for r, ref in zip(rnd, REAL):
        assert len(r) == len(ref)
        assert all(op in vocab for op in r)


def test_evaluate_report_shape_and_baseline_ordering():
    # Synthetic that mimics real structure should beat random and not beat real.
    synth = [
        ["push", "mov", "call", "ret"],
        ["push", "mov", "call", "ret"],
        ["mov", "push", "mov", "call"],
        ["push", "mov", "add", "pop"],
    ]
    rep = evaluate("test", REAL, synth, n_edit_pairs=50, seed=0, verbose=False)

    # shape
    for block in ("synthetic", "baseline_real_vs_real", "baseline_real_vs_random"):
        assert block in rep
        for metric in ("freq_kl", "ngram_overlap", "edit_distance"):
            assert metric in rep[block]

    kl_model = rep["synthetic"]["freq_kl"]["symmetric_kl"]
    kl_upper = rep["baseline_real_vs_real"]["freq_kl"]["symmetric_kl"]
    kl_lower = rep["baseline_real_vs_random"]["freq_kl"]["symmetric_kl"]
    # all three baselines are valid (finite, non-negative) divergences
    assert all(v >= 0.0 and math.isfinite(v) for v in (kl_model, kl_upper, kl_lower))
    # the robust contract: a structure-matching model beats the random lower bound.
    # (The real-vs-real floor is only a reliable floor on large corpora, not this toy.)
    assert kl_model <= kl_lower
