from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import settings


@dataclass
class VideoValidationResult:
    is_valid: bool
    error: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def validate_video_file(path: Path) -> VideoValidationResult:
    if path.suffix.lower() not in settings.allowed_video_extensions:
        return VideoValidationResult(False, f"unsupported extension: {path.suffix}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return VideoValidationResult(False, "file could not be opened as a video (possibly corrupt)")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, _frame = cap.read()
        if not ok:
            return VideoValidationResult(False, "could not decode any frames (possibly corrupt)")

        if not fps or fps <= 0:
            return VideoValidationResult(False, "invalid or missing frame rate")
        duration_s = frame_count / fps if frame_count > 0 else None

        if duration_s is not None:
            if duration_s < settings.min_video_duration_s:
                return VideoValidationResult(False, f"clip too short ({duration_s:.2f}s)", duration_s, width, height, fps)
            if duration_s > settings.max_video_duration_s:
                return VideoValidationResult(False, f"clip too long ({duration_s:.2f}s)", duration_s, width, height, fps)

        if width <= 0 or height <= 0:
            return VideoValidationResult(False, "invalid resolution", duration_s, width, height, fps)

        return VideoValidationResult(True, None, duration_s, width, height, fps)
    finally:
        cap.release()
