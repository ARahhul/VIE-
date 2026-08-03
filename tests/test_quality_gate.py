from pathlib import Path

import cv2
import numpy as np

from app.preprocessing.metrics import measure_quality_gain
from app.preprocessing.pipeline import run_quality_gate
from app.preprocessing.quality import assess_quality
from app.preprocessing.super_resolution import get_backend


def _write_clip(path: Path, size: tuple[int, int], frames: int = 24, fps: float = 10.0) -> Path:
    """A fixed-pixel-size checkerboard (sharp edges at any resolution) plus a
    moving circle, so blur-variance scoring reflects real sharpness rather
    than an artifact of a mostly-flat synthetic background."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    rng = np.random.default_rng(0)
    tile = 16
    for i in range(frames):
        xs, ys = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        checker = (((xs // tile) + (ys // tile)) % 2 * 200).astype(np.uint8)
        frame = np.stack([checker] * 3, axis=-1)
        cv2.circle(frame, (30 + i * 4, size[1] // 2), 20, (0, 200, 255), -1)
        frame = cv2.add(frame, rng.integers(0, 10, frame.shape, dtype=np.uint8))
        out.write(frame)
    out.release()
    return path


def test_quality_gate_upscales_low_res_clip(tmp_path):
    low_res = _write_clip(tmp_path / "low.mp4", size=(160, 120))
    output = tmp_path / "processed.mp4"

    result = run_quality_gate(str(low_res), str(output))

    assert result.was_upscaled is True
    assert result.quality_pre.needs_upscale is True
    post = assess_quality(str(output))
    assert post.width > result.quality_pre.width
    assert post.height > result.quality_pre.height


def test_quality_gate_skips_upscale_for_clean_clip(tmp_path):
    clean = _write_clip(tmp_path / "clean.mp4", size=(1280, 720))
    output = tmp_path / "processed.mp4"

    result = run_quality_gate(str(clean), str(output))

    assert result.was_upscaled is False
    assert result.quality_pre.needs_upscale is False


def test_super_resolution_backend_beats_naive_upscale_baseline(tmp_path):
    """Isolates the SR backend itself (Lanczos) from stabilization/fps-normalize,
    since the full pipeline can't disentangle those effects from upscale quality.

    Uses a photographic reference image (skimage's bundled sample), not a
    synthetic checkerboard: a grid pattern with edges perfectly aligned to the
    downsample factor is a pathological case where nearest-neighbor can exactly
    reconstruct the pattern, which flatters nearest-neighbor in a way no real
    footage would.
    """
    from skimage import data

    ref_frame = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    ref_frame = cv2.resize(ref_frame, (640, 480), interpolation=cv2.INTER_AREA)
    reference = tmp_path / "reference.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ref_writer = cv2.VideoWriter(str(reference), fourcc, 10.0, (640, 480))
    for _ in range(5):
        ref_writer.write(ref_frame)
    ref_writer.release()

    low_res_frame = cv2.resize(ref_frame, (160, 120), interpolation=cv2.INTER_AREA)

    backend = get_backend()
    processed = tmp_path / "processed.mp4"
    baseline = tmp_path / "baseline.mp4"
    proc_writer = cv2.VideoWriter(str(processed), fourcc, 10.0, (640, 480))
    base_writer = cv2.VideoWriter(str(baseline), fourcc, 10.0, (640, 480))
    for _ in range(5):
        proc_writer.write(backend.upscale(low_res_frame, 4.0))
        base_writer.write(cv2.resize(low_res_frame, (640, 480), interpolation=cv2.INTER_NEAREST))
    proc_writer.release()
    base_writer.release()

    gain = measure_quality_gain(str(processed), str(baseline), str(reference))
    assert gain.ssim_gain > 0
    assert gain.psnr_gain > 0
