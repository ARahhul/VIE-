"""Fisheye undistortion — runs before every other vision step.

Non-negotiable per the architecture notes: a homography (Phase 4) fitted to
a warped ground plane produces speeds that look plausible but are wrong.
Without a calibrated DeviceProfile for the source device, this is a no-op
passthrough rather than a guess.
"""

import shutil

import cv2
import numpy as np

from app.db.models import DeviceProfile


def _camera_matrices(device_profile: DeviceProfile) -> tuple[np.ndarray, np.ndarray] | None:
    if not device_profile.intrinsics or not device_profile.fisheye_coeffs:
        return None
    k = device_profile.intrinsics.get("K")
    d = device_profile.fisheye_coeffs.get("D")
    if not k or not d:
        return None
    return np.array(k, dtype=np.float64), np.array(d, dtype=np.float64)


def undistort_video(input_path: str, output_path: str, device_profile: DeviceProfile | None) -> bool:
    """Writes an undistorted copy to output_path. Returns whether it actually undistorted."""
    matrices = _camera_matrices(device_profile) if device_profile is not None else None
    if matrices is None:
        shutil.copyfile(input_path, output_path)
        return False

    k, d = matrices
    cap = cv2.VideoCapture(input_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open {input_path} for undistortion")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k, d, (width, height), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            k, d, np.eye(3), new_k, (width, height), cv2.CV_16SC2
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR))
        finally:
            writer.release()
        return True
    finally:
        cap.release()
