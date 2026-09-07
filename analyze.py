from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from detector import detect_rallies
from editor import cut_video


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cut dead time out of tennis recordings."
    )
    parser.add_argument("input", type=Path, help="Input MP4 file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output MP4 (default: <input>-<mode>.mp4)",
    )
    parser.add_argument(
        "--mode", choices=["rally", "match"], default="rally",
        help="rally: only ball-in-play. match: adds serve setup + wider merges.",
    )
    parser.add_argument(
        "--sample-hz", type=float, default=2.0,
        help="Frames per second to analyze (higher = slower, more accurate).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only detect rallies; print segments and skip encoding.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_stem(f"{args.input.stem}-{args.mode}")
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")

    print(f"input:  {args.input}")
    print(f"output: {output}")
    print(f"mode:   {args.mode}\n")

    t0 = time.time()
    segments, duration = detect_rallies(
        args.input, mode=args.mode, sample_hz=args.sample_hz
    )
    t_detect = time.time() - t0

    kept = sum(e - s for s, e in segments)
    print(
        f"\nfound {len(segments)} segments — kept {format_hms(kept)} of "
        f"{format_hms(duration)} ({kept/duration*100:.1f}%) in {t_detect:.1f}s"
    )

    if args.dry_run:
        for i, (s, e) in enumerate(segments, 1):
            print(f"  {i:3d}. {format_hms(s)} → {format_hms(e)}  ({e-s:.1f}s)")
        return 0

    if not segments:
        print("no rallies detected — nothing to write.", file=sys.stderr)
        return 2

    print("\nencoding…")
    t1 = time.time()
    cut_video(args.input, output, segments)
    print(f"done in {time.time() - t1:.1f}s → {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
