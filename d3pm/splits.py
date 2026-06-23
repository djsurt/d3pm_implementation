"""
Shared, deterministic file-level train/test split — the single source of truth
that keeps the generator's training data DISJOINT from the downstream
classifier's held-out test set.

Why this module exists
----------------------
The downstream-utility experiment (`comparison/lib/seq_classifier.py`) holds out a
fraction of each family's *files* as a test set and scores augmentation on it. The
D3PM generator (`d3pm/train.py`) must never train on those test files — otherwise
the synthetic data it produces carries information about the test set, and the
measured augmentation lift is optimistically biased (train/test contamination).

The two live in different virtualenvs (the generator needs torch/py3.14; the
classifier needs scikit-learn/py3.12), so they cannot share much code. This module
is therefore deliberately **pure standard-library** (only `random`) so *both* sides
can import the exact same split logic and be guaranteed to agree on which sequences
are "test".

Alignment contract
------------------
Both sides load a family's sequences the same way: ``sorted(glob("*.asm.txt"))``,
read each file, drop empties (``if ops``). That yields an identically-ordered list
on both sides, so a split defined purely by *index into that list* is consistent
across the two venvs without passing any files around.

Determinism / order-independence
--------------------------------
We create a **fresh** ``random.Random(seed)`` for every call, so a family's split
depends only on its own sequence count and the seed — never on how many other
families were processed first. (The previous classifier code reused one RNG across
families, which made each family's split depend on family ordering and was
impossible to reproduce from the generator, which trains one family at a time.)
"""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")

DEFAULT_TEST_FRAC = 0.3
DEFAULT_SEED = 1234


def test_index_set(n: int, test_frac: float = DEFAULT_TEST_FRAC, seed: int = DEFAULT_SEED) -> set[int]:
    """Indices (into a canonical-order sequence list of length ``n``) that are TEST.

    Mirrors the classifier's original rule ``n_test = max(1, round(n * test_frac))``,
    selecting them by shuffling ``range(n)`` with a fresh, seeded RNG.
    """
    if n <= 0:
        return set()
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = max(1, round(n * test_frac))
    return set(idx[:n_test])


def split_sequences(
    sequences: list[T],
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_SEED,
) -> tuple[list[T], list[T]]:
    """Split a canonical-order sequence list into ``(train_pool, test)``.

    ``train_pool`` is everything the generator is allowed to train on; ``test`` is
    the held-out set the downstream classifier scores against. Order within each
    side is preserved (the original canonical order, minus the other side).
    """
    test_idx = test_index_set(len(sequences), test_frac, seed)
    train = [s for i, s in enumerate(sequences) if i not in test_idx]
    test = [s for i, s in enumerate(sequences) if i in test_idx]
    return train, test
