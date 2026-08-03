import json

import cv2
import numpy as np

from app.db.models import VideoAsset
from app.llm.schemas import EventClaim, InvestigationNarrative
from app.report.build import build_report_context
from app.report.render import render_html, render_pdf
from app.verification.claims import verify_narrative


def test_verify_narrative_downgrades_ungrounded_claims():
    narrative = InvestigationNarrative(
        summary="test",
        claims=[
            EventClaim(who="track_id=1", what="braked hard", when_start_s=1.0, when_end_s=2.0, where="lane", how="deceleration", confidence="high"),
            EventClaim(who="track_id=99", what="appeared from nowhere", when_start_s=1.0, when_end_s=2.0, where="lane", how="?", confidence="high"),
        ],
    )

    verified = verify_narrative(narrative, known_track_ids={1}, video_duration_s=10.0)

    assert verified.claims[0].grounded is True
    assert verified.claims[0].confidence == "high"
    assert verified.claims[1].grounded is False
    assert verified.claims[1].confidence == "low"
    assert "unverified" in verified.claims[1].what


def _write_clip(path, frames=10, size=(160, 120)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for _ in range(frames):
        out.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    out.release()


def test_report_renders_pdf_with_and_without_narrative(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(json.dumps([
        {"frame_index": 0, "timestamp_s": 0.0, "track_id": 1, "class_id": 2, "class_name": "car",
         "bbox": [10, 10, 50, 50], "confidence": 0.9},
    ]))

    kinematics_path = tmp_path / "kinematics.json"
    kinematics_path.write_text(json.dumps([
        {"t": 0.0, "track_id": 1, "speed_mps": 12.0, "error_margin": 0.15, "method": "fused", "confidence": "medium"},
    ]))

    video_asset = VideoAsset(
        id="v1",
        incident_id="i1",
        file_path=str(clip),
        processed_path=str(clip),
        original_filename="clip.mp4",
        size_bytes=1,
        tracks_path=str(tracks_path),
        kinematics_path=str(kinematics_path),
        narrative_available=False,
    )

    ctx = build_report_context(video_asset, tmp_path / "evidence")
    assert "No video-LLM narrative" in ctx.executive_summary
    assert len(ctx.kinematics) == 1
    assert ctx.kinematics[0].actor == "track_id=1"

    html = render_html(ctx)
    assert "VigilNetra Investigation Report" in html

    pdf_bytes = render_pdf(html)
    assert pdf_bytes[:4] == b"%PDF"
