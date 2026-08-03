"""Event-window selection — the hard, exact trigger when we have one.

The impact sensor / IMU G-force spike (or an airbag optocoupler tap, when
logged) marks the event timestamp directly from the sensor log: free,
exact, and more reliable than any vision heuristic. Motion-energy detection
doesn't work on a moving camera, so the fallback when there's no sensor log
is a spike in mean optical-flow-magnitude residual — a sudden deceleration
shows up as a global flow discontinuity — flagged lower confidence.

Note: the selected window isn't fed back into detect_and_track/kinematics
yet (they still run on the full clip) — this is the boundary-selection half
of the feature; trimming the rest of the pipeline to it is a follow-up.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.kinematics.sensor_log import SensorSample, find_impact_timestamp

DEFAULT_WINDOW_S = 10.0


@dataclass
class EventWindow:
    start_s: float
    end_s: float
    center_s: float | None
    source: str  # "sensor_log" | "optical_flow_residual" | "none"
    confidence: str  # "high" | "low"


def _clip_duration_s(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()


def _optical_flow_residual_spike(video_path: str) -> float | None:
    """Sparse-feature mean flow magnitude per frame; the event is the frame
    with the largest jump (residual) in that magnitude vs. the previous frame."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ok, prev = cap.read()
        if not ok:
            return None
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        magnitudes: list[float] = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=20)
            if pts is None:
                magnitudes.append(0.0)
                prev_gray = gray
                continue
            curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
            valid = status.flatten() == 1
            if valid.sum() == 0:
                magnitudes.append(0.0)
            else:
                disp = curr_pts[valid] - pts[valid]
                magnitudes.append(float(np.linalg.norm(disp, axis=-1).mean()))
            prev_gray = gray

        if len(magnitudes) < 2:
            return None
        residuals = np.abs(np.diff(magnitudes))
        spike_frame = int(np.argmax(residuals)) + 1  # +1: diff shifts index by one
        return spike_frame / fps
    finally:
        cap.release()


def select_event_window(
    video_path: str,
    sensor_samples: list[SensorSample] | None,
    window_s: float = DEFAULT_WINDOW_S,
) -> EventWindow:
    duration_s = _clip_duration_s(video_path)

    impact_t = find_impact_timestamp(sensor_samples) if sensor_samples else None
    if impact_t is not None:
        return EventWindow(
            start_s=max(0.0, impact_t - window_s),
            end_s=min(duration_s, impact_t + window_s) if duration_s else impact_t + window_s,
            center_s=impact_t,
            source="sensor_log",
            confidence="high",
        )

    spike_t = _optical_flow_residual_spike(video_path)
    if spike_t is not None:
        return EventWindow(
            start_s=max(0.0, spike_t - window_s),
            end_s=min(duration_s, spike_t + window_s) if duration_s else spike_t + window_s,
            center_s=spike_t,
            source="optical_flow_residual",
            confidence="low",
        )

    return EventWindow(start_s=0.0, end_s=duration_s, center_s=None, source="none", confidence="low")
