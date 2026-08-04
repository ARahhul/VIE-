import cv2
import numpy as np

import app.graph.nodes as nodes_module
from app.llm.backends import _sample_frames_as_data_urls


def test_sample_frames_as_data_urls_returns_evenly_spaced_frames(tmp_path):
    clip = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(clip), fourcc, 10.0, (64, 48))
    for i in range(30):
        out.write(np.full((48, 64, 3), i, dtype=np.uint8))
    out.release()

    frames = _sample_frames_as_data_urls(str(clip), max_frames=5)

    assert len(frames) == 5
    timestamps = [t for t, _ in frames]
    assert timestamps == sorted(timestamps)
    for _t, data_url in frames:
        assert data_url.startswith("data:image/jpeg;base64,")


def test_video_llm_reasoning_degrades_gracefully_without_backend(monkeypatch):
    """No backend configured (forced here rather than relying on ambient env
    vars — a real .env with credentials must not change this test's outcome):
    the node must not fail the job, just record that no narrative is available."""
    monkeypatch.setattr(nodes_module, "get_backend", lambda: None)
    state = {"incident_id": "i1", "video_asset_id": "v1", "video_path": "unused.mp4"}

    result = nodes_module.video_llm_reasoning_node(state)

    assert result["narrative_available"] is False
    assert result["narrative_error"] is None
    assert "error" not in result or result["error"] is None


def test_video_llm_reasoning_skips_when_upstream_error_present():
    state = {"error": "quality_gate failed: boom"}
    result = nodes_module.video_llm_reasoning_node(state)
    assert result == state
