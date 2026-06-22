"""
Stage 1 — data pipeline: malicia opcode files -> padded integer tensors.

The raw data is one folder per malware family under `malicia/`, and inside each
folder one `*.asm.txt` file per sample, with **one opcode mnemonic per line**:

    push
    mov
    call
    ...

A neural network can't consume the *string* "mov"; it needs an integer. So this
module does three jobs:

  1. `load_family_opcodes` — read the .asm.txt files into Python lists of strings.
  2. `Vocabulary`           — a two-way map  opcode <-> integer id, plus two
                              special tokens [MASK] and [PAD].
  3. `OpcodeChunkedDataset` — slice every file into fixed-length integer tensors
                              the model trains on (a PyTorch `Dataset`).

Vocabulary index layout (kept consistent everywhere in the project):

    0 .. K-1   the K unique opcodes (sorted, so the mapping is deterministic)
    K          [MASK]   the *absorbing* token the diffusion process adds
    K+1        [PAD]     padding to make every chunk the same length
    vocab_size = K + 2
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ── 1. loading raw opcode sequences ──────────────────────────────────────────


def load_family_opcodes(
    malicia_dir: Path | str,
    family: str,
    max_files: int | None = None,
) -> list[list[str]]:
    """Read one family's `.asm.txt` files into a list of opcode-string lists.

    Returns `[[op, op, ...], [op, op, ...], ...]` — one inner list per file.
    Blank lines are dropped. `max_files` caps how many files we read (handy for
    fast local runs). Sorting the file list keeps runs reproducible.
    """
    family_dir = Path(malicia_dir) / family
    if not family_dir.is_dir():
        raise FileNotFoundError(f"Family directory not found: {family_dir}")

    files = sorted(family_dir.glob("*.asm.txt"))
    if max_files is not None:
        files = files[:max_files]

    sequences: list[list[str]] = []
    for f in files:
        ops = [line.strip() for line in f.read_text().splitlines() if line.strip()]
        if ops:
            sequences.append(ops)
    return sequences


# ── 2. vocabulary ────────────────────────────────────────────────────────────


class Vocabulary:
    """Bidirectional opcode <-> integer map, with [MASK] and [PAD] tokens.

    Build it once from the training sequences with `Vocabulary.from_sequences`,
    then `save` it next to the model checkpoint — at generation time you must
    decode with the *same* mapping you trained with.
    """

    MASK_TOKEN = "[MASK]"
    PAD_TOKEN = "[PAD]"

    def __init__(self) -> None:
        self._tok2idx: dict[str, int] = {}
        self._idx2tok: list[str] = []

    @classmethod
    def from_sequences(cls, sequences: list[list[str]]) -> "Vocabulary":
        """Collect the unique opcodes, sort them, then append [MASK], [PAD]."""
        vocab = cls()
        opcodes = sorted({op for seq in sequences for op in seq})
        # idx2tok uses indices as mapping to tokens.
        vocab._idx2tok = opcodes + [cls.MASK_TOKEN, cls.PAD_TOKEN]
        vocab._tok2idx = {tok: i for i, tok in enumerate(vocab._idx2tok)}
        return vocab

    @property
    def mask_idx(self) -> int:
        return self._tok2idx[self.MASK_TOKEN]

    @property
    def pad_idx(self) -> int:
        return self._tok2idx[self.PAD_TOKEN]

    @property
    def size(self) -> int:
        """Total number of ids = K opcodes + 2 special tokens."""
        return len(self._idx2tok)

    def encode(self, seq: list[str], max_len: int) -> torch.Tensor:
        """opcode strings -> LongTensor of ids, truncated then right-padded.

        An out-of-vocabulary opcode (only possible on data the vocab wasn't built
        from) maps to [MASK] as a safe fallback. Returns shape `(max_len,)`.
        """
        ids = [self._tok2idx.get(op, self.mask_idx) for op in seq[:max_len]]
        ids += [self.pad_idx] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    # decode is the inverse of encode, but it drops [MASK] and [PAD] tokens since
    # they aren't real opcodes and would just add noise to the output.
    def decode(self, ids: torch.Tensor) -> list[str]:
        """id tensor -> opcode strings, dropping [MASK] and [PAD]."""
        special = {self.mask_idx, self.pad_idx}
        return [self._idx2tok[i] for i in ids.tolist() if i not in special]

    def save(self, path: Path | str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ── 3. dataset ────────────────────────────────────────────────────────────────


class OpcodeChunkedDataset(Dataset):
    """Fixed-length training examples produced by chunking each file.

    Opcode files vary a lot in length (hundreds to tens of thousands of lines),
    but the transformer wants a fixed length `max_len`. Rather than truncate each
    file to its first `max_len` opcodes (throwing away most of the data), we slice
    every file into back-to-back, non-overlapping chunks of `max_len`. A file of
    1000 opcodes with `max_len=256` yields 3 full chunks + a 232-long remainder.

    The final partial chunk is right-padded with [PAD]; if it's shorter than
    `min_chunk` real opcodes it's dropped (too little signal once padded).

    A PyTorch `Dataset` just needs `__len__` and `__getitem__`. A `DataLoader`
    (used in training) calls those to assemble shuffled mini-batches for you.
    Each item is a dict of two tensors:
        tokens   LongTensor (max_len,)  the opcode ids (padded)
        pad_mask BoolTensor (max_len,)  True exactly where tokens == [PAD]
    `pad_mask` lets the model and the loss ignore padding positions.
    """

    def __init__(
        self,
        sequences: list[list[str]],
        vocab: Vocabulary,
        max_len: int,
        min_chunk: int = 32,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        chunks: list[torch.Tensor] = []
        for seq in sequences:
            for start in range(0, len(seq), max_len):
                piece = seq[start : start + max_len]
                if len(piece) < min_chunk:
                    continue  # drop a tiny tail chunk
                chunks.append(vocab.encode(piece, max_len))
        self.data = chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]
        pad_mask = tokens == self.vocab.pad_idx
        return {"tokens": tokens, "pad_mask": pad_mask}
