"""Benchmark harness: runs the quality-gate + detection pipeline against a
folder of test clips and reports latency + quality metrics per clip.

Usage:
    python scripts/benchmark.py path/to/clips_dir [--out results.csv]

Intended for the PRD's Phase 10 deliverable — run against public
crash-understanding datasets (e.g. CrashSight-style clips) plus your own
collected footage. No datasets ship with this repo; point it at whatever
clips you have (including low-light/two-wheeler clips per the PRD's
generalization risk — this harness doesn't distinguish scene type, so
tag/organize input folders yourself if you want a breakdown by condition).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.perception.detect_track import detect_and_track
from app.preprocessing.quality import assess_quality
from app.preprocessing.pipeline import run_quality_gate


def benchmark_clip(clip_path: Path, work_dir: Path) -> dict:
    row = {"clip": clip_path.name}

    t0 = time.perf_counter()
    quality_pre = assess_quality(str(clip_path))
    row["quality_score_pre"] = round(quality_pre.score, 3)
    row["resolution"] = f"{quality_pre.width}x{quality_pre.height}"

    processed_path = work_dir / f"processed_{clip_path.name}"
    result = run_quality_gate(str(clip_path), str(processed_path))
    row["quality_gate_s"] = round(time.perf_counter() - t0, 2)
    row["was_upscaled"] = result.was_upscaled
    row["quality_score_post"] = round(result.quality_post.score, 3)

    t1 = time.perf_counter()
    tracks_path = work_dir / f"tracks_{clip_path.stem}.json"
    try:
        summary = detect_and_track(str(processed_path), str(tracks_path))
        row["detect_track_s"] = round(time.perf_counter() - t1, 2)
        row["num_tracks"] = summary.num_tracks
    except Exception as exc:  # noqa: BLE001 - keep benchmarking the rest of the folder
        row["detect_track_s"] = None
        row["num_tracks"] = f"error: {exc}"

    row["total_s"] = round(time.perf_counter() - t0, 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.csv"))
    args = parser.parse_args()

    clips = sorted(
        p for p in args.clips_dir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
    )
    if not clips:
        print(f"no video clips found in {args.clips_dir}")
        return

    work_dir = args.clips_dir / "_benchmark_work"
    work_dir.mkdir(exist_ok=True)

    rows = []
    for clip in clips:
        print(f"benchmarking {clip.name}...")
        rows.append(benchmark_clip(clip, work_dir))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
