"""Benchmark harness: runs each clip through the *actual* compiled
StateGraph (the same one /ingest uses) and reports per-clip latency plus
whatever each node persisted, to CSV.

Earlier versions of this script called run_quality_gate()/detect_and_track()
directly, bypassing the graph — cheaper to iterate on, but it meant the
benchmark could silently drift from what production actually does (wrong
node order, a node that got added but not exercised here, etc.). Running
compiled_graph.invoke() end to end is a few seconds slower per clip but is
what the PRD's benchmarking deliverable is actually supposed to measure.

Usage:
    python scripts/benchmark.py path/to/clips_dir [--out results.csv]

Intended for the PRD's Phase 10 deliverable — run against public
crash-understanding datasets (e.g. CrashSight-style clips) plus your own
collected footage. No datasets ship with this repo; point it at whatever
clips you have (including low-light/two-wheeler clips per the PRD's
generalization risk — this harness doesn't distinguish scene type, so
tag/organize input folders yourself if you want a breakdown by condition).

Note: this uses whatever VIDEO_LLM_BACKEND/credentials are in your real
.env, same as the running app — if a backend is configured, every clip
makes a real video-LLM call. Set VIDEO_LLM_BACKEND=none in the environment
you run this script from if you want quality-gate/detection-only timing
without that cost.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _benchmark_clip(clip_path: Path, compiled_graph, SessionLocal, Incident, VideoAsset) -> dict:
    from app.graph.state import InvestigationState

    row = {"clip": clip_path.name}

    db = SessionLocal()
    try:
        incident = Incident(device_id=None)
        db.add(incident)
        db.flush()

        video_asset = VideoAsset(
            incident_id=incident.id,
            file_path=str(clip_path),
            original_filename=clip_path.name,
            size_bytes=clip_path.stat().st_size,
            is_valid=True,
        )
        db.add(video_asset)
        db.commit()
        db.refresh(video_asset)

        state: InvestigationState = {
            "incident_id": incident.id,
            "job_id": f"benchmark-{clip_path.stem}",
            "video_asset_id": video_asset.id,
            "video_path": str(clip_path),
            "sensor_log_path": None,
            "device_id": None,
        }

        t0 = time.perf_counter()
        result = compiled_graph.invoke(state)
        row["total_s"] = round(time.perf_counter() - t0, 2)
        row["error"] = result.get("error")

        db.refresh(video_asset)
        row["resolution"] = f"{video_asset.width}x{video_asset.height}"
        row["quality_score_pre"] = video_asset.quality_score_pre
        row["quality_score_post"] = video_asset.quality_score_post
        row["was_upscaled"] = video_asset.was_upscaled
        row["num_tracks"] = video_asset.num_tracks
        row["kinematics_method"] = video_asset.kinematics_method
        row["event_window_source"] = video_asset.event_window_source
        row["narrative_available"] = video_asset.narrative_available
        row["narrative_verified"] = video_asset.narrative_verified
        row["report_generated"] = bool(video_asset.report_path)
    finally:
        db.close()

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument(
        "--db", type=Path, default=None, help="dedicated sqlite file for this run (default: <clips_dir>/_benchmark.db)"
    )
    args = parser.parse_args()

    clips = sorted(
        p for p in args.clips_dir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
    )
    if not clips:
        print(f"no video clips found in {args.clips_dir}")
        return

    # A dedicated DB file — never the dev server's or the test suite's —
    # since this runs real graph.invoke() calls that persist to it.
    db_path = args.db or (args.clips_dir / "_benchmark.db")
    db_path.unlink(missing_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    from app.core.tracing import configure_tracing
    from app.db.base import SessionLocal, init_db
    from app.db.models import Incident, VideoAsset
    from app.graph.build import compiled_graph

    configure_tracing()  # must run before app.llm.backends' @observe decorators are used
    init_db()

    rows = []
    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] benchmarking {clip.name}...")
        rows.append(_benchmark_clip(clip, compiled_graph, SessionLocal, Incident, VideoAsset))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
