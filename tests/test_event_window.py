import cv2
import numpy as np

from app.events.event_window import select_event_window
from app.kinematics.sensor_log import SensorSample


def _write_clip(path, frames=40, fps=10.0, size=(160, 120)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 3) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_event_window_uses_sensor_log_impact_timestamp(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    samples = [
        SensorSample(t=0.0, imu_ax=0.0, imu_ay=0.0, imu_az=9.81),
        SensorSample(t=2.0, imu_ax=50.0, imu_ay=0.0, imu_az=9.81),
        SensorSample(t=4.0, imu_ax=0.0, imu_ay=0.0, imu_az=9.81),
    ]

    window = select_event_window(str(clip), samples, window_s=1.0)

    assert window.source == "sensor_log"
    assert window.confidence == "high"
    assert window.center_s == 2.0
    assert window.start_s == 1.0
    assert window.end_s == 3.0


def test_event_window_falls_back_without_sensor_log(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    window = select_event_window(str(clip), None, window_s=1.0)

    assert window.source in ("optical_flow_residual", "none")
    assert window.confidence == "low"
