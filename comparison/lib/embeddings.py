"""
Per-family Word2Vec embeddings, as described in the paper.

For each family:
  1. Train a Word2Vec model on all opcode sequences in that family.
  2. For each file, average all opcode vectors -> one 104-dim embedding.
  3. Scale embeddings to [-1, 1].
"""

import numpy as np
from gensim.models import Word2Vec


EMBED_DIM = 104


def train_family_word2vec(sequences: list[list[str]], dim: int = EMBED_DIM) -> Word2Vec:
    model = Word2Vec(
        sentences=sequences,
        vector_size=dim,
        window=5,
        min_count=1,
        workers=4,
        epochs=10,
        sg=0,  # CBOW
    )
    return model


def file_embedding(model: Word2Vec, opcodes: list[str]) -> np.ndarray:
    """Mean of word vectors for all opcodes in a file."""
    vectors = []
    for op in opcodes:
        if op in model.wv:
            vectors.append(model.wv[op])
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def scale_to_range(
    embeddings: np.ndarray,
    low: float = -1.0,
    high: float = 1.0,
    ref: np.ndarray | None = None,
) -> np.ndarray:
    """Min-max scale embeddings to [low, high] per-dimension.

    If `ref` is provided, use its per-dim min/max as the scaling source
    (so synthetic and real end up in the same coordinate system).
    """
    src = ref if ref is not None else embeddings
    e_min = src.min(axis=0, keepdims=True)
    e_max = src.max(axis=0, keepdims=True)
    denom = np.where(e_max - e_min == 0, 1.0, e_max - e_min)
    scaled = (embeddings - e_min) / denom  # [0, 1]
    return scaled * (high - low) + low


def build_family_embeddings(
    family_sequences: dict[str, list[list[str]]],
    dim: int = EMBED_DIM,
) -> tuple[dict[str, np.ndarray], dict[str, Word2Vec]]:
    """
    Returns:
        embeddings: {family: np.ndarray shape (N, dim)} scaled to [-1, 1]
        models:     {family: Word2Vec}
    """
    embeddings: dict[str, np.ndarray] = {}
    models: dict[str, Word2Vec] = {}

    # Word2Vec model per family. Then file embedding per file, then scale.
    for family, sequences in family_sequences.items():
        model = train_family_word2vec(sequences, dim=dim)
        raw = np.stack([file_embedding(model, seq) for seq in sequences])
        embeddings[family] = scale_to_range(raw)
        models[family] = model

    return embeddings, models
