import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.chdir(tmp_path)

    from app.main import app

    with TestClient(app) as c:
        yield c


def _make_clip(path: Path, frames: int = 20, fps: float = 10.0, size=(320, 240)) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(frames):
        out.write(np.full((size[1], size[0], 3), (i * 5) % 255, dtype=np.uint8))
    out.release()
    return path


def test_ingest_valid_clip_queues_and_completes_job(client, tmp_path):
    clip = _make_clip(tmp_path / "clip.mp4")

    with clip.open("rb") as f:
        resp = client.post("/ingest", files={"video": ("clip.mp4", f, "video/mp4")}, data={"device_id": "vn-001"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["video_asset"]["is_valid"] is True
    assert body["job"]["status"] == "queued"

    job_id = body["job"]["id"]
    # generous timeout: the full pipeline now runs quality-gate + YOLO
    # detection/tracking + kinematics fusion, and the first run also
    # downloads YOLO weights.
    for _ in range(300):
        job = client.get(f"/jobs/{job_id}").json()
        status = job["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert status == "completed", job.get("error")


def test_ingest_rejects_corrupt_file(client, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")

    with bad.open("rb") as f:
        resp = client.post("/ingest", files={"video": ("bad.mp4", f, "video/mp4")})

    assert resp.status_code == 422


def test_ingest_rejects_unsupported_extension(client, tmp_path):
    bad = tmp_path / "clip.txt"
    bad.write_text("hello")

    with bad.open("rb") as f:
        resp = client.post("/ingest", files={"video": ("clip.txt", f, "text/plain")})

    assert resp.status_code == 422
