"""Ego-motion — the vehicle's own speed/heading, measured from OBD/GPS/IMU.

This is the primary kinematic input, not a cross-check: ego speed is
measured directly rather than estimated from pixels, so the vision layer
(relative.py) only has to solve the easier relative-motion problem.
"""

import math
from dataclasses import dataclass

from app.kinematics.sensor_log import SensorSample


@dataclass
class EgoMotionPoint:
    t: float
    speed_mps: float
    heading_deg: float | None
    source: str  # "obd" | "gps"


def _haversine_speed(a: SensorSample, b: SensorSample) -> float | None:
    if None in (a.gps_lat, a.gps_lon, b.gps_lat, b.gps_lon) or b.t <= a.t:
        return None
    r = 6371000.0
    lat1, lat2 = math.radians(a.gps_lat), math.radians(b.gps_lat)
    dlat = math.radians(b.gps_lat - a.gps_lat)
    dlon = math.radians(b.gps_lon - a.gps_lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance_m = 2 * r * math.asin(math.sqrt(h))
    return distance_m / (b.t - a.t)


def _bearing_deg(a: SensorSample, b: SensorSample) -> float | None:
    if None in (a.gps_lat, a.gps_lon, b.gps_lat, b.gps_lon):
        return None
    lat1, lat2 = math.radians(a.gps_lat), math.radians(b.gps_lat)
    dlon = math.radians(b.gps_lon - a.gps_lon)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def compute_ego_motion(samples: list[SensorSample]) -> list[EgoMotionPoint]:
    """OBD speed wins when present (most direct measurement); GPS speed is
    the fallback, then GPS-derived speed from consecutive fixes."""
    points: list[EgoMotionPoint] = []
    for i, s in enumerate(samples):
        heading = _bearing_deg(samples[i - 1], s) if i > 0 else None

        if s.obd_speed_mps is not None:
            points.append(EgoMotionPoint(s.t, s.obd_speed_mps, heading, "obd"))
        elif s.gps_speed_mps is not None:
            points.append(EgoMotionPoint(s.t, s.gps_speed_mps, heading, "gps"))
        elif i > 0:
            derived = _haversine_speed(samples[i - 1], s)
            if derived is not None:
                points.append(EgoMotionPoint(s.t, derived, heading, "gps"))
    return points


def ego_speed_at(points: list[EgoMotionPoint], t: float) -> EgoMotionPoint | None:
    """Nearest-neighbor lookup — sensor logs and video frames are rarely on
    identical timebases, so this is intentionally not interpolation."""
    if not points:
        return None
    return min(points, key=lambda p: abs(p.t - t))
