from dataclasses import dataclass

import cv2
import numpy as np

# Below this, a frame is considered meaningfully blurry (Laplacian-variance heuristic).
BLUR_VARIANCE_GOOD = 150.0
# Below this many pixels (720p-equivalent area), resolution alone drags the score down.
RESOLUTION_GOOD_AREA = 1280 * 720


@dataclass
class QualityReport:
    mean_blur_variance: float
    width: int
    height: int
    fps: float
    bitrate_bps: float
    score: float  # 0 (unusable) .. 1 (clean, high-res)

    @property
    def needs_upscale(self) -> bool:
        return self.score < 0.5


def _sample_frame_indices(frame_count: int, max_samples: int = 20) -> list[int]:
    if frame_count <= max_samples:
        return list(range(frame_count))
    step = frame_count / max_samples
    return [int(i * step) for i in range(max_samples)]


def assess_quality(video_path: str) -> QualityReport:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open {video_path} for quality assessment")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        import os

        size_bytes = os.path.getsize(video_path)
        duration_s = (frame_count / fps) if fps > 0 else 0.0
        bitrate_bps = (size_bytes * 8 / duration_s) if duration_s > 0 else 0.0

        variances: list[float] = []
        for idx in _sample_frame_indices(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variances.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        mean_blur_variance = float(np.mean(variances)) if variances else 0.0

        resolution_score = min(1.0, (width * height) / RESOLUTION_GOOD_AREA)
        blur_score = min(1.0, mean_blur_variance / BLUR_VARIANCE_GOOD)
        # weakest-link, not an average: a sharp 160x120 frame is still unusable,
        # and a blurry 4K frame is still blurry.
        score = min(resolution_score, blur_score)

        return QualityReport(
            mean_blur_variance=mean_blur_variance,
            width=width,
            height=height,
            fps=fps,
            bitrate_bps=bitrate_bps,
            score=score,
        )
    finally:
        cap.release()
