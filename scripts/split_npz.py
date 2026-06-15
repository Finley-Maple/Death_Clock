"""
split_npz.py — Split a large .npz into smaller chunk files for transfer.

Background: featurize.cn scp silently truncates files larger than ~10 MB at
~57% completion without any error message.  Chunk large embedding .npz files
before scp transfer, then reconstruct on the Mac using join_npz.py.

Usage:
    python scripts/split_npz.py path/to/file.npz --chunk-mb 8
    python scripts/split_npz.py path/to/file.npz --chunk-mb 8 --out-dir chunks/

Output:
    {stem}_chunk_000.npz, {stem}_chunk_001.npz, ...
    Each chunk contains a subset of the eid keys.

Reconstruct with:
    python scripts/join_npz.py chunks/{stem}_chunk_*.npz -o reconstructed.npz
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def estimate_bytes_per_key(data, sample_n: int = 50) -> float:
    """Estimate mean compressed bytes per key using a small sample."""
    keys = list(data.keys())
    sample_keys = keys[:min(sample_n, len(keys))]
    total_bytes = sum(data[k].nbytes for k in sample_keys)
    return total_bytes / len(sample_keys)


def split_npz(input_path: Path, chunk_mb: float, out_dir: Path) -> None:
    print(f"Loading {input_path} …")
    data = np.load(input_path, allow_pickle=True)
    keys = list(data.keys())

    if not keys:
        print("ERROR: .npz file has no keys.", file=sys.stderr)
        sys.exit(1)

    n_keys = len(keys)
    bytes_per_key = estimate_bytes_per_key(data)
    target_bytes = chunk_mb * 1e6

    # Compressed npz is typically 50-70% of raw; use 0.6 as conservative factor
    # so we don't overshoot.  Users can tune --chunk-mb if needed.
    effective_bytes_per_key = bytes_per_key * 0.6
    chunk_size = max(1, int(target_bytes / effective_bytes_per_key))

    n_chunks = math.ceil(n_keys / chunk_size)

    print(f"  {n_keys} keys  |  ~{bytes_per_key / 1e3:.1f} KB/key (raw)  "
          f"|  chunk_size={chunk_size} keys  |  {n_chunks} chunks")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_keys)
        chunk_keys = keys[start:end]
        chunk_path = out_dir / f"{stem}_chunk_{chunk_idx:03d}.npz"

        arrays = {k: data[k] for k in chunk_keys}
        np.savez_compressed(chunk_path, **arrays)

        size_mb = chunk_path.stat().st_size / 1e6
        print(f"  Wrote {chunk_path.name}  ({len(chunk_keys)} keys, {size_mb:.2f} MB)")

    print(f"\nSplit {n_keys} keys into {n_chunks} chunks of ~{chunk_mb} MB each.")
    print(f"Output dir: {out_dir}")
    print(f"\nReconstruct with:")
    print(f"  python scripts/join_npz.py {out_dir}/{stem}_chunk_*.npz -o {stem}.npz")


def main():
    parser = argparse.ArgumentParser(
        description="Split a large .npz file into smaller chunk files for scp transfer."
    )
    parser.add_argument("input", type=Path, help="Path to the input .npz file.")
    parser.add_argument(
        "--chunk-mb", type=float, default=8.0,
        help="Target chunk size in MB (default: 8). Keep below 10 for featurize.cn."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for chunks (default: same directory as input file)."
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix != ".npz":
        print(f"WARNING: Expected .npz, got {input_path.suffix}", file=sys.stderr)

    out_dir = args.out_dir.resolve() if args.out_dir else input_path.parent / f"{input_path.stem}_chunks"
    split_npz(input_path, args.chunk_mb, out_dir)


if __name__ == "__main__":
    main()
