"""Stage 1 verification — run the data pipeline on a real malicia family and
print human-readable diagnostics.

    .venv/bin/python -m verify.verify_data --family cleaman

What to look for:
  * Vocab size in the low hundreds (opcodes + 2 special tokens).
  * The decoded sample reads like real disassembly (mov / push / call / ...).
  * A DataLoader batch has shape (batch, max_len) and dtype int64 (torch.long).
  * The PAD-suffix invariant holds for every chunk (padding never appears mid-chunk).
"""

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from d3pm.config import Config
from d3pm.data import OpcodeChunkedDataset, Vocabulary, load_family_opcodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicia", default="malicia")
    ap.add_argument("--family", default="cleaman")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=Config.small().max_len)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    print(f"Loading family '{args.family}' from {args.malicia}/ ...")
    seqs = load_family_opcodes(args.malicia, args.family, max_files=args.max_files)
    lengths = [len(s) for s in seqs]
    print(f"  files loaded     : {len(seqs)}")
    print(f"  opcodes/file     : min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")

    vocab = Vocabulary.from_sequences(seqs)
    print(f"  vocab size       : {vocab.size}  (opcodes={vocab.size - 2}, MASK={vocab.mask_idx}, PAD={vocab.pad_idx})")
    top = Counter(op for s in seqs for op in s).most_common(8)
    print(f"  most common ops  : {top}")

    print(f"\nDecoded first 20 opcodes of file 0:")
    print("  " + " ".join(seqs[0][:20]))

    ds = OpcodeChunkedDataset(seqs, vocab, max_len=args.max_len)
    print(f"\nChunked dataset: {len(ds)} chunks of length {args.max_len}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    batch = next(iter(loader))
    tokens, pad_mask = batch["tokens"], batch["pad_mask"]
    print(f"  batch tokens shape : {tuple(tokens.shape)}  dtype={tokens.dtype}")
    print(f"  batch pad_mask shape: {tuple(pad_mask.shape)} dtype={pad_mask.dtype}")

    # PAD-suffix invariant: in every row, once PAD starts it never goes back to a
    # real token. Equivalent to: the boolean pad_mask is non-decreasing along the row.
    ok = True
    for row in pad_mask:
        r = row.int()
        if (r[1:] - r[:-1] < 0).any():   # a 1 -> 0 transition would be a violation
            ok = False
            break
    # pad_mask must also agree exactly with "token == pad_idx"
    agree = bool((pad_mask == (tokens == vocab.pad_idx)).all())
    print(f"\n  PAD-suffix invariant holds : {ok}")
    print(f"  pad_mask matches PAD tokens : {agree}")
    print("\nVERDICT:", "PASS ✅" if (ok and agree) else "FAIL ❌")


if __name__ == "__main__":
    main()
