"""Command-line entry point: ``python -m tracking_validation ctc <folder> [...]``."""
from __future__ import annotations

import argparse
import sys


def _cmd_ctc(a: argparse.Namespace) -> int:
    from .ctc import read_ctc_tracks, summarize, write_tracks_csv

    data = read_ctc_tracks(
        a.folder,
        voxel_size=tuple(a.voxel_size) if a.voxel_size else None,
        dataset=a.dataset,
        pad_to_3d=a.pad_to_3d,
    )
    print(summarize(data))
    if a.out:
        print("wrote", write_tracks_csv(data, a.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tracking_validation")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("ctc", help="convert a Cell Tracking Challenge tracking folder to a track CSV")
    c.add_argument("folder", help="e.g. Fluo-N3DH-CE/01_GT/TRA or Fluo-N3DH-CE/01_RES")
    c.add_argument("--dataset", help="CTC dataset name for voxel-size lookup, e.g. Fluo-N3DH-CE")
    c.add_argument("--voxel-size", type=float, nargs="+", metavar="UM",
                   help="microns per voxel, x y [z] (overrides --dataset)")
    c.add_argument("--pad-to-3d", action="store_true", help="append z=0 to 2D data")
    c.add_argument("--out", help="output CSV path (default: print summary only)")
    c.set_defaults(func=_cmd_ctc)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
