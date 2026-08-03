"""absolute_speed(other_vehicle) = ego_speed(OBD/GPS) (+) relative_speed(vision)

Ego speed is measured, so error on the fused estimate is dominated by the
vision term rather than two compounding estimates. Without a sensor log,
the pipeline still runs, but every speed is flagged vision-only/low-confidence.
"""

from dataclasses import dataclass

from app.kinematics.ego import EgoMotionPoint, ego_speed_at
from app.kinematics.relative import RelativeMotionPoint

FUSED_ERROR_MARGIN = 0.15  # dominated by the vision (relative-motion) term
VISION_ONLY_ERROR_MARGIN = 0.35  # both terms are vision-estimated


@dataclass
class AbsoluteSpeedPoint:
    t: float
    track_id: int
    speed_mps: float
    error_margin: float
    method: str  # "fused" | "vision-only"
    confidence: str  # "high" | "medium" | "low"


def fuse_absolute_speed(
    relative_points: list[RelativeMotionPoint],
    ego_points: list[EgoMotionPoint] | None,
) -> list[AbsoluteSpeedPoint]:
    results: list[AbsoluteSpeedPoint] = []
    for rp in relative_points:
        ego = ego_speed_at(ego_points, rp.t) if ego_points else None
        if ego is not None:
            results.append(
                AbsoluteSpeedPoint(
                    t=rp.t,
                    track_id=rp.track_id,
                    speed_mps=ego.speed_mps + rp.speed_mps,
                    error_margin=FUSED_ERROR_MARGIN,
                    method="fused",
                    confidence="medium",
                )
            )
        else:
            results.append(
                AbsoluteSpeedPoint(
                    t=rp.t,
                    track_id=rp.track_id,
                    speed_mps=rp.speed_mps,
                    error_margin=VISION_ONLY_ERROR_MARGIN,
                    method="vision-only",
                    confidence="low",
                )
            )
    return results
