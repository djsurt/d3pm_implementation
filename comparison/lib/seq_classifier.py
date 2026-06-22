"""
Downstream sequence-classifier augmentation study  (D3PM's headline capability).
================================================================================

WHY THIS EXISTS
---------------
The embedding-level comparison (`evaluate.py`) shows D3PM and DDPM at *parity* on
DDPM's native, near-saturated metric — and the paper's own augmentation experiment
there was a no-op (full-data family classification sits at F1 ~= 1.0, so adding
synthetic data cannot help). That is the wrong axis to judge D3PM on.

This module measures **extrinsic utility on a task only D3PM's output can serve**:
classifying malware by its *opcode sequence*. DDPM emits a single mean-pooled
embedding per file — the sequence is unrecoverable — so it cannot supply training
data to a sequence classifier at all. D3PM emits real opcode sequences, so it can.

The experiment: multi-class family classification (winwebsec / zbot / zeroaccess)
from n-gram features of opcode windows, under an artificially restricted REAL-data
budget. For each budget N (real files/family) we compare:

    real-only            : train on N real files/family
    real + D3PM-synth    : train on N real files/family + D3PM-generated windows

Both are scored on the SAME held-out *real* test set (never augmented). The story
we expect, and that makes this paper-worthy:

    * at small N (scarce real data) synthetic augmentation LIFTS macro-F1
    * the lift shrinks toward 0 as N grows and the task saturates

That curve is the proof that D3PM's sequences carry real, usable signal — the
"capability" claim, not a metric-beating claim. n-gram features explicitly encode
opcode *order*, which is exactly the information DDPM's pooling destroys.

WHY n-gram + linear (not a neural net)
--------------------------------------
In the low-data regime a high-capacity net just overfits; a bag-of-(1,2,3)-grams +
linear SVM is the standard strong, stable, deterministic baseline for token
sequences, and its features directly expose the order information at issue here.

This lives in `comparison/` because it needs scikit-learn (py3.12 venv); the main
torch venv (py3.14) has no sklearn wheel.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from loader import load_family_opcodes


# ── data prep ───────────────────────────────────────────────────────────────


def chunk_sequence(ops: list[str], window: int, min_chunk: int) -> list[list[str]]:
    """Split one opcode list into non-overlapping `window`-length windows.

    Mirrors the D3PM training chunking: drop a trailing window shorter than
    `min_chunk` so every sample is reasonably full. A file shorter than `window`
    but >= `min_chunk` is kept whole (one window).
    """
    windows = []
    for start in range(0, len(ops), window):
        chunk = ops[start : start + window]
        if len(chunk) >= min_chunk:
            windows.append(chunk)
    return windows


def windows_to_docs(windows: list[list[str]]) -> list[str]:
    """TfidfVectorizer wants strings; opcodes are word-char tokens, so join with
    spaces and let the vectorizer split on whitespace."""
    return [" ".join(w) for w in windows]


def load_real_files(malicia_dir: Path, families: list[str]) -> dict[str, list[list[str]]]:
    """{family: [file_opcodes, ...]} — file-level, NOT yet chunked (we split files
    into train/test *before* chunking so windows from one file never straddle the
    split, which would leak)."""
    return load_family_opcodes(malicia_dir, families=families)


def load_synth_windows(
    run_dir: Path, families: list[str], window: int, min_chunk: int
) -> dict[str, list[list[str]]]:
    """Read D3PM-generated sequences from `<run_dir>/synthetic/<family>/seq_*.txt`
    and chunk them to the same window size as the real data."""
    synth_root = Path(run_dir) / "synthetic"
    out: dict[str, list[list[str]]] = {}
    for fam in families:
        fam_dir = synth_root / fam
        seqs = []
        for f in sorted(fam_dir.glob("seq_*.txt")):
            ops = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
            seqs.extend(chunk_sequence(ops, window, min_chunk))
        out[fam] = seqs
    return out


# ── one train/score round ───────────────────────────────────────────────────


def _fit_score(
    train_docs: list[str],
    train_labels: list[str],
    test_docs: list[str],
    test_labels: list[str],
    seed: int,
) -> float:
    """Fit TF-IDF(1-3gram) + LinearSVC on the training docs, return macro-F1 on
    the (always-real) test docs. Vectorizer is fit on the training side only."""
    clf = make_pipeline(
        TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 3), min_df=2),
        LinearSVC(C=1.0, random_state=seed),
    )
    clf.fit(train_docs, train_labels)
    pred = clf.predict(test_docs)
    return f1_score(test_labels, pred, average="macro")


def _labelled_docs(per_family_windows: dict[str, list[list[str]]]):
    """Flatten {family: [windows]} into (docs, labels)."""
    docs, labels = [], []
    for fam, windows in per_family_windows.items():
        docs.extend(windows_to_docs(windows))
        labels.extend([fam] * len(windows))
    return docs, labels


# ── the budget sweep ────────────────────────────────────────────────────────


def budget_sweep(
    malicia_dir: Path,
    run_dir: Path,
    families: list[str],
    budgets: list[Optional[int]],
    window: int = 256,
    min_chunk: int = 32,
    test_frac: float = 0.3,
    synth_per_family: int = 200,
    seeds: tuple[int, ...] = (0, 1, 2),
    test_seed: int = 1234,
) -> dict:
    """Run the real-only vs real+synth comparison across real-data budgets.

    A single FIXED real test split (held out by file, `test_seed`) is reused for
    every budget and seed so all numbers are mutually comparable. For each budget
    N and each `seed`, we subsample N train files/family, chunk them, and score two
    conditions on that fixed test set. `budgets` may contain None meaning "all
    available train files".
    """
    real = load_real_files(Path(malicia_dir), families)
    synth = load_synth_windows(Path(run_dir), families, window, min_chunk)

    # 1) Fixed test split, by FILE, per family. Remaining files form the train pool.
    rng = random.Random(test_seed)
    train_pool: dict[str, list[list[str]]] = {}
    test_windows: dict[str, list[list[str]]] = {}
    for fam in families:
        files = list(real[fam])
        rng.shuffle(files)
        n_test = max(1, int(round(len(files) * test_frac)))
        test_files, pool_files = files[:n_test], files[n_test:]
        tw = []
        for f in test_files:
            tw.extend(chunk_sequence(f, window, min_chunk))
        test_windows[fam] = tw
        train_pool[fam] = pool_files

    test_docs, test_labels = _labelled_docs(test_windows)

    pool_sizes = {fam: len(train_pool[fam]) for fam in families}
    results = []
    for budget in budgets:
        per_seed_real, per_seed_aug = [], []
        for seed in seeds:
            srng = random.Random(seed)
            # subsample N train files/family, then chunk
            real_train: dict[str, list[list[str]]] = {}
            for fam in families:
                pool = list(train_pool[fam])
                srng.shuffle(pool)
                take = pool if budget is None else pool[:budget]
                rw = []
                for f in take:
                    rw.extend(chunk_sequence(f, window, min_chunk))
                real_train[fam] = rw

            # condition A: real only
            rd, rl = _labelled_docs(real_train)
            f1_real = _fit_score(rd, rl, test_docs, test_labels, seed)

            # condition B: real + D3PM synthetic (capped, subsampled per seed)
            aug_train = {fam: list(real_train[fam]) for fam in families}
            for fam in families:
                sw = list(synth[fam])
                srng.shuffle(sw)
                aug_train[fam].extend(sw[:synth_per_family])
            ad, al = _labelled_docs(aug_train)
            f1_aug = _fit_score(ad, al, test_docs, test_labels, seed)

            per_seed_real.append(f1_real)
            per_seed_aug.append(f1_aug)

        rmean, rstd = float(np.mean(per_seed_real)), float(np.std(per_seed_real))
        amean, astd = float(np.mean(per_seed_aug)), float(np.std(per_seed_aug))
        results.append(
            {
                "budget": budget,
                "real_only_f1": rmean,
                "real_only_std": rstd,
                "real_plus_synth_f1": amean,
                "real_plus_synth_std": astd,
                "lift": amean - rmean,
            }
        )

    return {
        "families": families,
        "window": window,
        "min_chunk": min_chunk,
        "test_frac": test_frac,
        "synth_per_family": synth_per_family,
        "seeds": list(seeds),
        "pool_sizes": pool_sizes,
        "test_windows_per_family": {f: len(test_windows[f]) for f in families},
        "synth_windows_available": {f: len(synth[f]) for f in families},
        "sweep": results,
    }


# ── reporting ───────────────────────────────────────────────────────────────


def print_report(report: dict) -> None:
    print(f"\nDownstream sequence-classifier augmentation — {report['families']}")
    print(
        f"window={report['window']}  test_frac={report['test_frac']}  "
        f"synth/family={report['synth_per_family']}  seeds={report['seeds']}"
    )
    print(f"train-pool files/family: {report['pool_sizes']}")
    print(f"test windows/family:     {report['test_windows_per_family']}")
    print(f"synth windows available: {report['synth_windows_available']}\n")
    print(f"{'budget':>8} | {'real-only F1':>16} | {'real+synth F1':>16} | {'lift':>8}")
    print("-" * 60)
    for row in report["sweep"]:
        b = "all" if row["budget"] is None else str(row["budget"])
        print(
            f"{b:>8} | {row['real_only_f1']:.4f} ± {row['real_only_std']:.3f}   "
            f"| {row['real_plus_synth_f1']:.4f} ± {row['real_plus_synth_std']:.3f}   "
            f"| {row['lift']:+.4f}"
        )
    print()


def _parse_budgets(s: str) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        out.append(None if tok.lower() == "all" else int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--malicia", default="../malicia", help="malicia dataset dir")
    ap.add_argument(
        "--d3pm-run",
        default="../runs/run_01_baseline",
        help="run folder holding synthetic/<family>/seq_*.txt",
    )
    ap.add_argument(
        "--families",
        default="winwebsec,zbot,zeroaccess",
        help="comma-separated families to classify",
    )
    ap.add_argument(
        "--budgets",
        default="5,10,25,50,100,all",
        help="comma-separated real files/family budgets ('all' allowed)",
    )
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--min-chunk", type=int, default=32)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--synth-per-family", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/seq_classifier_report.json")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",")]
    seeds = tuple(int(s) for s in args.seeds.split(","))

    report = budget_sweep(
        malicia_dir=args.malicia,
        run_dir=args.d3pm_run,
        families=families,
        budgets=_parse_budgets(args.budgets),
        window=args.window,
        min_chunk=args.min_chunk,
        test_frac=args.test_frac,
        synth_per_family=args.synth_per_family,
        seeds=seeds,
    )
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
