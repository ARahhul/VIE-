"""Detection & multi-object tracking — per-frame trajectories for every
vehicle, pedestrian, and two-wheeler in the clip.

Uses Ultralytics YOLOv11 for detection and its built-in ByteTrack (default)
or BoT-SORT tracker for cross-frame identity, matching the PRD tech stack.
Output is a per-object trajectory log (bbox, class, track ID, frame index,
timestamp) written to disk as JSON — the raw-perception-blob store the PRD
calls out as living outside the relational DB (object storage/Mongo in
production; a local JSON file for now).
"""

import json
from dataclasses import asdict, dataclass

import cv2

# COCO class ids relevant to road-traffic incidents.
VEHICLE_CLASSES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class TrackPoint:
    frame_index: int
    timestamp_s: float
    track_id: int
    class_id: int
    class_name: str
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float


@dataclass
class DetectionSummary:
    tracks_path: str
    num_frames: int
    num_tracks: int


def detect_and_track(
    video_path: str,
    output_json_path: str,
    model_name: str = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
) -> DetectionSummary:
    from ultralytics import YOLO

    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    points: list[TrackPoint] = []
    track_ids: set[int] = set()

    results = model.track(
        source=video_path,
        tracker=tracker,
        classes=list(VEHICLE_CLASSES.keys()),
        persist=True,
        stream=True,
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        for box, track_id, cls_id, conf in zip(
            boxes.xyxy.tolist(), boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        ):
            track_id = int(track_id)
            cls_id = int(cls_id)
            track_ids.add(track_id)
            points.append(
                TrackPoint(
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=VEHICLE_CLASSES.get(cls_id, str(cls_id)),
                    bbox=[float(v) for v in box],
                    confidence=float(conf),
                )
            )

    with open(output_json_path, "w") as f:
        json.dump([asdict(p) for p in points], f)

    return DetectionSummary(
        tracks_path=output_json_path,
        num_frames=frame_index + 1 if points else 0,
        num_tracks=len(track_ids),
    )


def load_tracks(tracks_path: str) -> list[TrackPoint]:
    with open(tracks_path) as f:
        raw = json.load(f)
    return [TrackPoint(**r) for r in raw]
