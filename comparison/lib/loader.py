"""
Load opcode sequences from malicia .asm.txt files.
Returns a dict mapping family_name -> list of opcode lists (one per file).
"""

from pathlib import Path
from typing import Optional


def load_family_opcodes(
    malicia_dir: Path,
    families: Optional[list[str]] = None,
    max_files_per_family: Optional[int] = None,
) -> dict[str, list[list[str]]]:
    """
    Returns {family: [[op, op, ...], [op, op, ...], ...]}
    """
    malicia_dir = Path(malicia_dir)
    family_dirs = sorted(d for d in malicia_dir.iterdir() if d.is_dir())

    if families is not None:
        family_dirs = [d for d in family_dirs if d.name in families]

    corpus: dict[str, list[list[str]]] = {}
    for family_dir in family_dirs:
        files = sorted(family_dir.glob("*.asm.txt"))
        if max_files_per_family is not None:
            files = files[:max_files_per_family]

        sequences = []
        for f in files:
            ops = [line.strip() for line in f.read_text().splitlines() if line.strip()]
            if ops:
                sequences.append(ops)

        if sequences:
            corpus[family_dir.name] = sequences

    return corpus
