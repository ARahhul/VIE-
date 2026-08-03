"""SSIM/PSNR comparison against a ground-truth reference clip.

Used by benchmarks/tests (Phase 2 exit criteria, Phase 10 benchmarking) to
measure whether the quality-gate pipeline's output is actually better than a
naive upscale of the same degraded input — not just bigger.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class QualityGain:
    ssim_gain: float
    psnr_gain: float
    processed_ssim: float
    processed_psnr: float
    baseline_ssim: float
    baseline_psnr: float


def _read_frames(path: str, max_frames: int = 20) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    ssim = structural_similarity(a, b, channel_axis=2)
    psnr = peak_signal_noise_ratio(a, b, data_range=255)
    return float(ssim), float(psnr)


def measure_quality_gain(processed_path: str, baseline_path: str, reference_path: str) -> QualityGain:
    """Compares `processed` (pipeline output) and `baseline` (naive upscale)
    against `reference` (ground-truth high-quality clip), frame by frame."""
    ref_frames = _read_frames(reference_path)
    processed_frames = _read_frames(processed_path)
    baseline_frames = _read_frames(baseline_path)

    n = min(len(ref_frames), len(processed_frames), len(baseline_frames))
    if n == 0:
        raise ValueError("no comparable frames across processed/baseline/reference clips")

    proc_ssim, proc_psnr, base_ssim, base_psnr = [], [], [], []
    for i in range(n):
        s, p = _compare(ref_frames[i], processed_frames[i])
        proc_ssim.append(s)
        proc_psnr.append(p)
        s, p = _compare(ref_frames[i], baseline_frames[i])
        base_ssim.append(s)
        base_psnr.append(p)

    processed_ssim, processed_psnr = float(np.mean(proc_ssim)), float(np.mean(proc_psnr))
    baseline_ssim, baseline_psnr = float(np.mean(base_ssim)), float(np.mean(base_psnr))

    return QualityGain(
        ssim_gain=processed_ssim - baseline_ssim,
        psnr_gain=processed_psnr - baseline_psnr,
        processed_ssim=processed_ssim,
        processed_psnr=processed_psnr,
        baseline_ssim=baseline_ssim,
        baseline_psnr=baseline_psnr,
    )
