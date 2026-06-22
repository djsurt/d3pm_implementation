"""Stage 1 unit tests — Vocabulary and OpcodeChunkedDataset invariants.

These use tiny hand-made opcode sequences (no dataset on disk required), so they
run in milliseconds and pin down the exact behaviour we rely on later.
"""

import torch

from d3pm.data import OpcodeChunkedDataset, Vocabulary


SEQS = [
    ["push", "mov", "call", "ret"],
    ["mov", "xor", "push", "mov", "add"],
]


def test_vocab_layout():
    v = Vocabulary.from_sequences(SEQS)
    # 5 unique opcodes (add, call, mov, push, ret, xor -> actually 6) + MASK + PAD
    uniq = {"push", "mov", "call", "ret", "xor", "add"}
    assert v.size == len(uniq) + 2
    # special tokens are the last two ids, in MASK, PAD order
    assert v.mask_idx == len(uniq)
    assert v.pad_idx == len(uniq) + 1
    # opcodes are sorted -> deterministic ids
    assert v._idx2tok[: len(uniq)] == sorted(uniq)


def test_encode_decode_roundtrip():
    v = Vocabulary.from_sequences(SEQS)
    seq = SEQS[0]
    ids = v.encode(seq, max_len=8)
    assert ids.shape == (8,)
    assert ids.dtype == torch.long
    # decode drops the padding, recovering the original opcodes
    assert v.decode(ids) == seq


def test_encode_truncates_then_pads():
    v = Vocabulary.from_sequences(SEQS)
    ids = v.encode(SEQS[1], max_len=3)         # truncate a length-5 seq to 3
    assert ids.tolist() == [v._tok2idx[o] for o in SEQS[1][:3]]
    ids2 = v.encode(SEQS[0], max_len=6)        # pad a length-4 seq up to 6
    assert (ids2[4:] == v.pad_idx).all()       # padding is a suffix only


def test_oov_maps_to_mask():
    v = Vocabulary.from_sequences(SEQS)
    ids = v.encode(["mov", "not_a_real_opcode"], max_len=2)
    assert ids[1].item() == v.mask_idx


def test_chunking_lengths_and_padmask():
    # one long sequence -> multiple full chunks + a tail
    long_seq = ["mov"] * 10
    v = Vocabulary.from_sequences([long_seq])
    ds = OpcodeChunkedDataset([long_seq], v, max_len=4, min_chunk=2)
    # chunks: [0:4], [4:8], [8:10] (len 2 >= min_chunk) -> 3 chunks
    assert len(ds) == 3
    item = ds[0]
    assert item["tokens"].shape == (4,)
    assert item["pad_mask"].shape == (4,)
    assert item["pad_mask"].dtype == torch.bool
    # the last chunk is padded; pad_mask must mark exactly the PAD positions
    last = ds[2]
    assert last["pad_mask"].tolist() == [False, False, True, True]
    assert (last["tokens"][last["pad_mask"]] == v.pad_idx).all()


def test_tiny_tail_chunk_dropped():
    v = Vocabulary.from_sequences([["mov"]])
    # length 5, max_len 4, min_chunk 2 -> chunk [0:4] kept, tail [4:5] (len 1) dropped
    ds = OpcodeChunkedDataset([["mov"] * 5], v, max_len=4, min_chunk=2)
    assert len(ds) == 1
