"""Compact per-object context for the video-LLM prompt.

Previously the raw per-frame tracks JSON (every bbox for every frame for
every tracked object) was dumped straight into the prompt. On a clip with
40+ tracked vehicles that blew past the model's context window (verified:
400 "maximum context length is 32768 tokens... requested 51933") and
failed narrative generation outright. This summarizes each track to one
row — class, how long it was visible, speed range/method/confidence —
which is also just a better-shaped input for the model: it's reasoning
over per-vehicle behavior, not reconstructing it from thousands of frames.
"""

from dataclasses import asdict, dataclass

from app.perception.detect_track import TrackPoint, load_tracks


@dataclass
class TrackSummary:
    track_id: int
    class_name: str
    first_seen_s: float
    last_seen_s: float
    num_detections: int
    mean_confidence: float
    min_speed_mps: float | None = None
    max_speed_mps: float | None = None
    mean_speed_mps: float | None = None
    kinematics_method: str | None = None
    kinematics_confidence: str | None = None


def _summarize_tracks(tracks: list[TrackPoint]) -> dict[int, TrackSummary]:
    by_id: dict[int, list[TrackPoint]] = {}
    for t in tracks:
        by_id.setdefault(t.track_id, []).append(t)

    summaries: dict[int, TrackSummary] = {}
    for track_id, points in by_id.items():
        points.sort(key=lambda p: p.frame_index)
        summaries[track_id] = TrackSummary(
            track_id=track_id,
            class_name=points[0].class_name,
            first_seen_s=points[0].timestamp_s,
            last_seen_s=points[-1].timestamp_s,
            num_detections=len(points),
            mean_confidence=round(sum(p.confidence for p in points) / len(points), 3),
        )
    return summaries


def _merge_kinematics(summaries: dict[int, TrackSummary], kinematics_points: list[dict]) -> None:
    by_id: dict[int, list[dict]] = {}
    for p in kinematics_points:
        by_id.setdefault(p["track_id"], []).append(p)

    for track_id, points in by_id.items():
        if track_id not in summaries:
            continue
        speeds = [p["speed_mps"] for p in points]
        summaries[track_id].min_speed_mps = round(min(speeds), 2)
        summaries[track_id].max_speed_mps = round(max(speeds), 2)
        summaries[track_id].mean_speed_mps = round(sum(speeds) / len(speeds), 2)
        summaries[track_id].kinematics_method = points[-1]["method"]
        summaries[track_id].kinematics_confidence = points[-1]["confidence"]


def build_context_summary(
    tracks_path: str | None,
    kinematics_points: list[dict],
    event_window: dict,
) -> dict:
    tracks = load_tracks(tracks_path) if tracks_path else []
    summaries = _summarize_tracks(tracks)
    _merge_kinematics(summaries, kinematics_points)

    return {
        "event_window": event_window,
        "num_tracked_objects": len(summaries),
        "tracked_objects": [asdict(s) for s in sorted(summaries.values(), key=lambda s: s.track_id)],
    }
