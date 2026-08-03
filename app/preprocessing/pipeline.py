"""Phase 2 quality gate: undistort -> stabilize -> conditional SR -> fps normalize.

This is the node that sits right after ingest in the compiled StateGraph.
Order matters: undistortion has to happen before stabilization (feature
tracking on a warped fisheye frame is unreliable) and before any homography
work downstream (Phase 4).
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import settings
from app.db.models import DeviceProfile
from app.preprocessing.framerate import normalize_frame_rate
from app.preprocessing.quality import QualityReport, assess_quality
from app.preprocessing.stabilize import stabilize_video
from app.preprocessing.super_resolution import get_backend
from app.preprocessing.undistort import undistort_video


@dataclass
class QualityGateResult:
    output_path: str
    quality_pre: QualityReport
    quality_post: QualityReport
    was_undistorted: bool
    was_upscaled: bool


def _upscale_video(input_path: str, output_path: str, scale: float) -> None:
    backend = get_backend()
    cap = cv2.VideoCapture(input_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open {input_path} for super-resolution")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(backend.upscale(frame, scale))
        finally:
            writer.release()
    finally:
        cap.release()


def run_quality_gate(
    input_path: str,
    output_path: str,
    device_profile: DeviceProfile | None = None,
    upscale_scale: float = 2.0,
) -> QualityGateResult:
    quality_pre = assess_quality(input_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        undistorted_path = tmp / "undistorted.mp4"
        was_undistorted = undistort_video(input_path, str(undistorted_path), device_profile)

        stabilized_path = tmp / "stabilized.mp4"
        stabilize_video(str(undistorted_path), str(stabilized_path))

        was_upscaled = quality_pre.needs_upscale
        if was_upscaled:
            upscaled_path = tmp / "upscaled.mp4"
            _upscale_video(str(stabilized_path), str(upscaled_path), upscale_scale)
            source_for_fps = upscaled_path
        else:
            source_for_fps = stabilized_path

        normalize_frame_rate(str(source_for_fps), output_path, settings.target_fps)

    quality_post = assess_quality(output_path)
    return QualityGateResult(
        output_path=output_path,
        quality_pre=quality_pre,
        quality_post=quality_post,
        was_undistorted=was_undistorted,
        was_upscaled=was_upscaled,
    )
