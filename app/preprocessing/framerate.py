"""Frame-rate normalization — resamples a clip to a fixed target fps.

Frame-duplication/dropping (nearest-frame resampling), not optical-flow
interpolation: cheap, deterministic, and good enough to give every
downstream stage (tracking, kinematics) a consistent timebase.
"""

import cv2


def normalize_frame_rate(input_path: str, output_path: str, target_fps: float) -> float:
    """Returns the fps the output was actually written at."""
    cap = cv2.VideoCapture(input_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open {input_path} for frame-rate normalization")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0 or abs(src_fps - target_fps) < 1e-6:
            cap.release()
            import shutil

            shutil.copyfile(input_path, output_path)
            return src_fps

        duration_s = frame_count / src_fps
        out_frame_count = max(1, round(duration_s * target_fps))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        try:
            for out_idx in range(out_frame_count):
                src_idx = min(frame_count - 1, round(out_idx * src_fps / target_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            writer.release()
        return target_fps
    finally:
        cap.release()
