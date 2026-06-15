"""
join_npz.py — Reconstruct a .npz file from chunks produced by split_npz.py.

Usage:
    python scripts/join_npz.py chunks/stem_chunk_*.npz -o reconstructed.npz

All keys from every chunk are merged into a single .npz file.  If the same
key appears in more than one chunk the script aborts with an error.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def join_npz(chunk_paths: list[Path], output_path: Path) -> None:
    all_arrays: dict = {}

    for chunk_path in sorted(chunk_paths):
        if not chunk_path.exists():
            print(f"ERROR: Chunk file not found: {chunk_path}", file=sys.stderr)
            sys.exit(1)

        chunk = np.load(chunk_path, allow_pickle=True)
        keys = list(chunk.keys())
        print(f"  Loading {chunk_path.name}  ({len(keys)} keys)")

        for k in keys:
            if k in all_arrays:
                print(
                    f"ERROR: Duplicate key '{k}' found in {chunk_path.name}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            all_arrays[k] = chunk[k]

    if not all_arrays:
        print("ERROR: No keys found across all chunks.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSaving {len(all_arrays)} keys → {output_path} …")
    np.savez_compressed(output_path, **all_arrays)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Done. Output: {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge .npz chunk files produced by split_npz.py into one file."
    )
    parser.add_argument(
        "chunks", nargs="+", type=Path,
        help="One or more chunk .npz files (glob-expanded by the shell)."
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Path for the merged output .npz file."
    )
    args = parser.parse_args()

    chunk_paths = [p.resolve() for p in args.chunks]
    output_path = args.output.resolve()

    if output_path.exists():
        print(f"WARNING: Output file already exists and will be overwritten: {output_path}")

    print(f"Joining {len(chunk_paths)} chunk(s) → {output_path}")
    join_npz(chunk_paths, output_path)


if __name__ == "__main__":
    main()
