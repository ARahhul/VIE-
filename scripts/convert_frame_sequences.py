"""Stitches frame-sequence datasets (e.g. CCD-style "CrashBest": clips as
numbered JPG sequences, no video files) into .mp4 clips the pipeline can
actually ingest and benchmark against.

Expects <source_dir>/<clip_id>_<frame_nn>.jpg, e.g. C_000001_01.jpg ..
C_000001_50.jpg. Frame rate is a guess (these datasets rarely state it) —
override with --fps if you know better.

Usage:
    python scripts/convert_frame_sequences.py Dataset/CrashBest --out Dataset/CrashBest_mp4 --limit 15
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2

FRAME_RE = re.compile(r"^(?P<clip_id>.+)_(?P<frame_num>\d+)\.jpg$", re.IGNORECASE)


def group_clips(source_dir: Path) -> dict[str, list[Path]]:
    clips: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for f in source_dir.iterdir():
        m = FRAME_RE.match(f.name)
        if not m:
            continue
        clips[m.group("clip_id")].append((int(m.group("frame_num")), f))
    return {clip_id: [p for _, p in sorted(frames)] for clip_id, frames in clips.items()}


def write_clip(frames: list[Path], out_path: Path, fps: float) -> None:
    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    try:
        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            writer.write(img)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N clips")
    args = parser.parse_args()

    out_dir = args.out or (args.source_dir.parent / f"{args.source_dir.name}_mp4")
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = group_clips(args.source_dir)
    clip_ids = sorted(clips.keys())
    if args.limit:
        clip_ids = clip_ids[: args.limit]

    for i, clip_id in enumerate(clip_ids, 1):
        frames = clips[clip_id]
        out_path = out_dir / f"{clip_id}.mp4"
        print(f"[{i}/{len(clip_ids)}] {clip_id}: {len(frames)} frames -> {out_path}")
        write_clip(frames, out_path, args.fps)

    print(f"wrote {len(clip_ids)} clips to {out_dir}")


if __name__ == "__main__":
    main()
