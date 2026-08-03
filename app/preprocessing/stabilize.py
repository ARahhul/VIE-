"""Feature-tracking video stabilization (translation + rotation + scale).

Classic two-pass approach: estimate frame-to-frame affine motion, smooth the
cumulative trajectory with a moving average, then re-render each frame with
the correction needed to follow the smoothed path instead of the raw,
suspension/handheld-shake-corrupted one.
"""

import cv2
import numpy as np


def _moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    window = 2 * radius + 1
    kernel = np.ones(window) / window
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def stabilize_video(input_path: str, output_path: str, smoothing_radius: int = 5) -> bool:
    cap = cv2.VideoCapture(input_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open {input_path} for stabilization")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if n_frames < 3:
            cap.release()
            import shutil

            shutil.copyfile(input_path, output_path)
            return False

        ok, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        transforms = []  # dx, dy, da per frame pair
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            if prev_pts is None:
                transforms.append((0.0, 0.0, 0.0))
                prev_gray = gray
                continue

            curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            valid = status.flatten() == 1
            prev_valid, curr_valid = prev_pts[valid], curr_pts[valid]

            if len(prev_valid) < 6:
                transforms.append((0.0, 0.0, 0.0))
                prev_gray = gray
                continue

            m, _inliers = cv2.estimateAffinePartial2D(prev_valid, curr_valid)
            if m is None:
                transforms.append((0.0, 0.0, 0.0))
            else:
                dx, dy = m[0, 2], m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
                transforms.append((dx, dy, da))
            prev_gray = gray

        transforms = np.array(transforms)
        trajectory = np.cumsum(transforms, axis=0)

        radius = min(smoothing_radius, max(1, len(trajectory) // 2))
        smoothed = np.column_stack(
            [_moving_average(trajectory[:, i], radius) for i in range(3)]
        )
        corrections = smoothed - trajectory

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        try:
            ok, frame = cap.read()
            writer.write(frame)  # first frame is the anchor, written as-is
            for dx, dy, da in corrections:
                ok, frame = cap.read()
                if not ok:
                    break
                m = np.array(
                    [[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]],
                    dtype=np.float64,
                )
                stabilized = cv2.warpAffine(frame, m, (width, height), borderMode=cv2.BORDER_REPLICATE)
                writer.write(stabilized)
        finally:
            writer.release()
        return True
    finally:
        cap.release()
