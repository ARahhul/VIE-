"""Invokes the compiled StateGraph directly, bypassing /ingest.

This is the gap that let a real bug through: every existing test drove the
pipeline via the /ingest endpoint, which creates uploads/<incident_id>/ as
a side effect of saving the uploaded file. Nodes that write into that same
directory (quality_gate, detect_and_track, kinematics, video_llm_reasoning,
report_generation) never created it themselves, so calling the graph any
other way — a benchmark script, a retry/reprocess path, anything that
doesn't go through the upload endpoint first — failed at quality_gate with
"cannot open ... for quality assessment", because the video writer never
had anywhere to write to.
"""

import cv2
import numpy as np

from app.db.base import SessionLocal, init_db
from app.db.models import Incident, VideoAsset
from app.graph.build import compiled_graph

init_db()


def _write_clip(path, frames=20, size=(320, 240)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 5) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_graph_invoke_creates_its_own_incident_directory(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    db = SessionLocal()
    try:
        incident = Incident(device_id=None)
        db.add(incident)
        db.flush()

        video_asset = VideoAsset(
            incident_id=incident.id,
            file_path=str(clip),
            original_filename="clip.mp4",
            size_bytes=clip.stat().st_size,
            is_valid=True,
        )
        db.add(video_asset)
        db.commit()
        db.refresh(video_asset)

        state = {
            "incident_id": incident.id,
            "job_id": "test-job",
            "video_asset_id": video_asset.id,
            "video_path": str(clip),
            "sensor_log_path": None,
            "device_id": None,
        }

        result = compiled_graph.invoke(state)

        assert result.get("error") is None, result.get("error")
        db.refresh(video_asset)
        assert video_asset.processed_path is not None
        assert bool(video_asset.report_path)
    finally:
        db.close()
