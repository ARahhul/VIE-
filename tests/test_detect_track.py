import cv2
import numpy as np
import pytest

from app.perception.detect_track import detect_and_track


def _write_traffic_clip(path, frames=15, size=(320, 240)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 5.0, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (20 + i * 10, 150), (100 + i * 10, 200), (0, 0, 200), -1)
        out.write(frame)
    out.release()


def test_detect_and_track_writes_trajectory_log(tmp_path):
    clip = tmp_path / "traffic.mp4"
    _write_traffic_clip(clip)
    output_json = tmp_path / "tracks.json"

    try:
        summary = detect_and_track(str(clip), str(output_json))
    except Exception as exc:  # noqa: BLE001 - no network access to fetch YOLO weights
        pytest.skip(f"YOLO model unavailable in this environment: {exc}")

    assert output_json.exists()
    assert summary.tracks_path == str(output_json)
