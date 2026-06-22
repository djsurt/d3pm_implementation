"""
VirusShare PE → opcode `.asm.txt` extractor.

Why this exists
---------------
The Malicia pipeline trained on *already-disassembled* opcode text: one folder per
family, one `*.asm.txt` per sample, one opcode mnemonic per line (see `data.py`).
VirusShare instead ships *raw PE binaries* (hash-named, no extension, no labels).
This script turns those binaries into the exact same `.asm.txt` format so the rest
of the pipeline (`data.py` → `train.py` → `generate.py` → `seq_eval.py`) works
unchanged — for a first end-to-end run we point the trainer at the output dir as a
single unlabelled corpus (a "family" called e.g. `all`).

How it disassembles
-------------------
`pefile` parses the PE and locates the executable sections; `capstone` linearly
disassembles their bytes (x86 32/64 chosen from the PE header) and we keep just the
mnemonic of each instruction. This is the standard static opcode-extraction recipe
used in the malware-ML literature. Linear sweep on packed/obfuscated samples yields
junk — we drop any output with fewer than `--min-ops` instructions so the training
corpus isn't full of 3-opcode stubs.

NOTE: these are live malware binaries. We only ever *read* and statically
disassemble them — nothing is executed. Run this on a disposable pod/VM.

Usage
-----
    python -m d3pm.disasm --in /workspace/samples --out /workspace/vs_opcodes/all \
        --max-samples 4000 --workers 16

Then train on the result as a single-family corpus:
    python -m d3pm.train --family all --malicia /workspace/vs_opcodes --scaleup ...
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import capstone
import pefile


def _disassemble_one(path: str, min_ops: int) -> tuple[str, int]:
    """Disassemble one PE file → list of opcode mnemonics (newline-joined).

    Returns (mnemonics_text, n_ops). On any parse/format error returns ("", 0) so
    the caller can skip it; malware samples are frequently malformed and we want to
    keep going, not crash the whole run.
    """
    try:
        pe = pefile.PE(path, fast_load=True)
    except Exception:
        return "", 0

    # 32- vs 64-bit decides capstone's mode. 0x20b == PE32+ (64-bit).
    is_64 = pe.OPTIONAL_HEADER.Magic == 0x20B
    md = capstone.Cs(capstone.CS_ARCH_X86,
                     capstone.CS_MODE_64 if is_64 else capstone.CS_MODE_32)

    mnemonics: list[str] = []
    for section in pe.sections:
        # IMAGE_SCN_MEM_EXECUTE = 0x20000000 — only disassemble executable sections.
        if not (section.Characteristics & 0x20000000):
            continue
        code = section.get_data()
        base = pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress
        for insn in md.disasm(code, base):
            mnemonics.append(insn.mnemonic)

    pe.close()
    if len(mnemonics) < min_ops:
        return "", len(mnemonics)
    return "\n".join(mnemonics), len(mnemonics)


def _worker(args: tuple[str, str, int]) -> tuple[str, int]:
    """Pool worker: disassemble `src` and, if non-empty, write `<name>.asm.txt`."""
    src, out_dir, min_ops = args
    text, n = _disassemble_one(src, min_ops)
    if text:
        name = Path(src).name
        (Path(out_dir) / f"{name}.asm.txt").write_text(text)
    return src, n


def extract(in_dir: str, out_dir: str, *, max_samples: int | None = None,
            min_ops: int = 50, workers: int | None = None) -> dict:
    """Disassemble every PE under `in_dir` into `out_dir`/<name>.asm.txt.

    Walks `in_dir` recursively (VirusShare archives are often flat, but be safe),
    disassembles up to `max_samples` files in parallel, and reports how many
    produced a usable opcode sequence.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = [str(p) for p in Path(in_dir).rglob("*") if p.is_file()]
    files.sort()
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise ValueError(f"no files found under {in_dir}")

    workers = workers or os.cpu_count() or 4
    kept, total_ops = 0, 0
    tasks = [(f, out_dir, min_ops) for f in files]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            _src, n = fut.result()
            if n >= min_ops:
                kept += 1
                total_ops += n
            if i % 500 == 0:
                print(f"  processed {i}/{len(files)}  kept={kept}", flush=True)

    summary = {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "n_input": len(files),
        "n_kept": kept,
        "kept_frac": round(kept / max(1, len(files)), 3),
        "mean_ops_kept": round(total_ops / max(1, kept), 1),
    }
    print(f"\nDone: {kept}/{len(files)} samples → usable opcode files "
          f"({summary['kept_frac']:.0%}); mean {summary['mean_ops_kept']} ops each")
    print(f"Train with:  --family {out.name} --malicia {out.parent}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Disassemble VirusShare PE samples to opcode .asm.txt")
    ap.add_argument("--in", dest="in_dir", required=True, help="dir of raw PE samples")
    ap.add_argument("--out", dest="out_dir", required=True,
                    help="output dir (its basename is the 'family' for --family)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap how many samples to disassemble (subset run)")
    ap.add_argument("--min-ops", type=int, default=50,
                    help="drop outputs with fewer than this many opcodes (packed/junk)")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: all cores)")
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()
    extract(args.in_dir, args.out_dir, max_samples=args.max_samples,
            min_ops=args.min_ops, workers=args.workers)


if __name__ == "__main__":
    main()
