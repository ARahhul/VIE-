"""Conditional super-resolution backend.

Production target per the PRD tech stack is FlashVSR (real-time diffusion
VSR, temporally consistent) with Real-ESRGAN as the constrained-hardware
fallback. Neither ships model weights or a GPU runtime in this environment,
so `LanczosUpscaler` is the working default: classical, dependency-light,
and enough to prove the conditional-gate + measurable-quality-gain pipeline
end to end. Swapping in FlashVSR/Real-ESRGAN later only means implementing
`SuperResolutionBackend` and pointing `get_backend()` at it — the gate logic
and pipeline plumbing don't change.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class SuperResolutionBackend(ABC):
    @abstractmethod
    def upscale(self, frame: np.ndarray, scale: float) -> np.ndarray: ...


class LanczosUpscaler(SuperResolutionBackend):
    """Classical fallback: sharper than bicubic, no model weights required."""

    def upscale(self, frame: np.ndarray, scale: float) -> np.ndarray:
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


class FlashVSRBackend(SuperResolutionBackend):
    """Not wired up yet — needs the FlashVSR checkpoint + a CUDA runtime."""

    def upscale(self, frame: np.ndarray, scale: float) -> np.ndarray:
        raise NotImplementedError("FlashVSR backend requires model weights + GPU runtime, not available yet")


class RealESRGANBackend(SuperResolutionBackend):
    """Not wired up yet — constrained-hardware fallback per the PRD tech stack."""

    def upscale(self, frame: np.ndarray, scale: float) -> np.ndarray:
        raise NotImplementedError("Real-ESRGAN backend requires model weights, not available yet")


def get_backend() -> SuperResolutionBackend:
    return LanczosUpscaler()
