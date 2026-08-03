"""Relative motion of tracked vehicles vs. the ego frame — the vision term.

Scale recovery here is the lane-width prior (~3.0-3.5m for an Indian
highway/urban lane): the coarsest of the three methods in the architecture
notes, used because it needs no per-frame vanishing-point estimation.
Homography/vanishing-point calibration (method 2) and Depth Anything V2
(method 3, scale-anchored by 1 or 2) are the documented upgrade path once
Phase 4 has calibrated test clips to validate against — this MVP scale
should be treated as coarse, not production-grade.
"""

from dataclasses import dataclass

from app.perception.detect_track import TrackPoint

DEFAULT_LANE_WIDTH_M = 3.5
# Fraction of frame width assumed to correspond to one lane, at the vehicle's
# on-screen scale — a placeholder until vanishing-point calibration lands.
ASSUMED_LANE_WIDTH_FRACTION = 0.3


@dataclass
class RelativeMotionPoint:
    t: float
    track_id: int
    speed_mps: float  # relative to the ego vehicle, magnitude only


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def compute_relative_motion(
    tracks: list[TrackPoint],
    frame_width: int,
    lane_width_m: float = DEFAULT_LANE_WIDTH_M,
) -> list[RelativeMotionPoint]:
    meters_per_pixel = lane_width_m / (frame_width * ASSUMED_LANE_WIDTH_FRACTION)

    by_track: dict[int, list[TrackPoint]] = {}
    for tp in tracks:
        by_track.setdefault(tp.track_id, []).append(tp)

    points: list[RelativeMotionPoint] = []
    for track_id, pts in by_track.items():
        pts.sort(key=lambda p: p.frame_index)
        for prev, curr in zip(pts, pts[1:]):
            dt = curr.timestamp_s - prev.timestamp_s
            if dt <= 0:
                continue
            (px1, py1), (px2, py2) = _bbox_center(prev.bbox), _bbox_center(curr.bbox)
            pixel_dist = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
            speed_mps = (pixel_dist * meters_per_pixel) / dt
            points.append(RelativeMotionPoint(t=curr.timestamp_s, track_id=track_id, speed_mps=speed_mps))
    return points
