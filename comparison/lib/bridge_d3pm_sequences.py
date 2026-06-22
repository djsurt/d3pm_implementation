"""
Bridge: turn fresh-implementation D3PM *sequences* into the embedding-level
layout that evaluate.py --variant d3pm reads.

The D3PM rebuild generates raw opcode sequences under
<run>/synthetic/<family>/seq_*.txt — one opcode per line. The paper's 6-metric
eval suite compares mean-pooled Word2Vec *embeddings*. This script is the missing
W2V bridge: per family it

  1. loads the *real* opcode sequences from malicia,
  2. trains a per-family Word2Vec on them (the D3PM W2V space),
  3. embeds the real files  -> checkpoints/<family>/d3pm_real_embeddings.npy,
  4. groups synthetic chunks into length-matched "files" (so the pooling depth
     matches real files — see _group_into_files), embeds them in the SAME space
     min-max scaled against the real reference -> synthetic/<family>/d3pm.npy.

Run from the comparison/ directory:

    cd comparison
    ../.venv/bin/python lib/bridge_d3pm_sequences.py --d3pm-run ../runs/run_01_baseline
    ../.venv/bin/python lib/evaluate.py --variant d3pm \
        --compare-against ddpm_baseline_report.json --out results
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np

from loader import load_family_opcodes
from embeddings import train_family_word2vec, file_embedding, scale_to_range
import paths


def _load_synth_sequences(seq_dir: Path) -> list[list[str]]:
    """Read <seq_dir>/seq_*.txt, one opcode per line, into a list of opcode lists."""
    sequences: list[list[str]] = []
    for f in sorted(seq_dir.glob("seq_*.txt")):
        ops = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        if ops:
            sequences.append(ops)
    return sequences


def _embed(w2v, sequences: list[list[str]]) -> np.ndarray:
    return np.stack([file_embedding(w2v, seq) for seq in sequences])


def _group_into_files(
    chunks: list[list[str]],
    chunks_per_file: int,
    n_files: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    """Bootstrap `n_files` synthetic 'files', each a concatenation of
    `chunks_per_file` chunks sampled with replacement.

    Mean-pooling a file of K*256 opcodes matches the pooling *depth* of the real
    files (avg 1940-5701 opcodes) instead of a single 256-token chunk, removing
    the length confound. Sampling with replacement keeps sample count healthy.
    """
    if chunks_per_file <= 1:
        return chunks
    files: list[list[str]] = []
    for _ in range(n_files):
        idx = rng.integers(0, len(chunks), chunks_per_file)
        flat: list[str] = []
        for j in idx:
            flat.extend(chunks[j])
        files.append(flat)
    return files


def bridge_family(
    family: str,
    *,
    malicia_dir: Path,
    synth_seq_dir: Path,
    checkpoints: Path,
    synthetic: Path,
    embed_dim: int,
    n_files: int,
    chunks_per_file: int | None,
    seed: int,
) -> None:
    print(f"\n[{family}]")

    corpus = load_family_opcodes(malicia_dir, families=[family])
    if family not in corpus:
        raise SystemExit(f"  real family '{family}' not found under {malicia_dir}")
    real_seqs = corpus[family]
    avg_real = sum(len(s) for s in real_seqs) / len(real_seqs)
    print(f"  real files: {len(real_seqs)}  (avg len {avg_real:.0f} opcodes)")

    if not synth_seq_dir.exists():
        raise SystemExit(f"  synthetic seq dir not found: {synth_seq_dir}")
    synth_chunks = _load_synth_sequences(synth_seq_dir)
    if not synth_chunks:
        raise SystemExit(f"  no seq_*.txt files in {synth_seq_dir}")
    chunk_len = sum(len(s) for s in synth_chunks) / len(synth_chunks)

    cpf = chunks_per_file or max(1, math.ceil(avg_real / chunk_len))
    rng = np.random.default_rng(seed)
    synth_seqs = _group_into_files(synth_chunks, cpf, n_files, rng)
    avg_synth = sum(len(s) for s in synth_seqs) / len(synth_seqs)
    print(f"  synth chunks: {len(synth_chunks)} (len {chunk_len:.0f}) "
          f"-> {len(synth_seqs)} files of {cpf} chunks (avg len {avg_synth:.0f} opcodes)")

    print(f"  training W2V on real sequences (dim={embed_dim})...")
    w2v = train_family_word2vec(real_seqs, dim=embed_dim)

    real_raw = _embed(w2v, real_seqs)
    real_scaled = scale_to_range(real_raw)
    synth_raw = _embed(w2v, synth_seqs)
    synth_scaled = scale_to_range(synth_raw, ref=real_raw)

    real_vocab = set(w2v.wv.key_to_index)
    synth_vocab = {op for seq in synth_chunks for op in seq}
    oov = synth_vocab - real_vocab
    print(f"  synth opcode coverage: {len(synth_vocab - oov)}/{len(synth_vocab)} in W2V"
          f"  ({len(oov)} OOV -> zero-contribution)")

    fd_ckpt = paths.family_ckpt_dir(checkpoints, family)
    fd_synth = paths.family_synth_dir(synthetic, family)
    np.save(fd_ckpt / paths.D3PM_REAL_EMBEDDINGS, real_scaled)
    np.save(fd_synth / paths.D3PM_SYNTHETIC, synth_scaled)
    with open(fd_ckpt / paths.D3PM_W2V, "wb") as fh:
        pickle.dump(w2v, fh)
    print(f"  saved real  {real_scaled.shape} -> {fd_ckpt / paths.D3PM_REAL_EMBEDDINGS}")
    print(f"  saved synth {synth_scaled.shape} -> {fd_synth / paths.D3PM_SYNTHETIC}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d3pm-run", type=Path, default=Path("../runs/run_01_baseline"),
                    help="run folder holding synthetic/<family>/seq_*.txt")
    ap.add_argument("--malicia", type=Path, default=Path("../malicia"),
                    help="real opcode dir")
    ap.add_argument("--families", nargs="+", default=["winwebsec", "zbot", "zeroaccess"])
    ap.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    ap.add_argument("--synthetic", type=Path, default=Path("synthetic"))
    ap.add_argument("--embed-dim", type=int, default=104)
    ap.add_argument("--n-files", type=int, default=500,
                    help="synthetic files to build per family (bootstrap from chunks)")
    ap.add_argument("--chunks-per-file", type=int, default=None,
                    help="chunks per synthetic file (default: ceil(avg_real_len/chunk_len))")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    synth_root = args.d3pm_run / "synthetic"
    print(f"Bridging D3PM sequences from {synth_root}\nReal opcodes from {args.malicia}")
    for fam in args.families:
        bridge_family(
            fam,
            malicia_dir=args.malicia,
            synth_seq_dir=synth_root / fam,
            checkpoints=args.checkpoints,
            synthetic=args.synthetic,
            embed_dim=args.embed_dim,
            n_files=args.n_files,
            chunks_per_file=args.chunks_per_file,
            seed=args.seed,
        )
    print("\nDone. Now run:\n  ../.venv/bin/python lib/evaluate.py --variant d3pm "
          "--compare-against ddpm_baseline_report.json --out results")


if __name__ == "__main__":
    main()
