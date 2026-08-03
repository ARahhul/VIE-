from typing import TypedDict


class InvestigationState(TypedDict, total=False):
    """Shared state threaded through the compiled StateGraph.

    Only the fields needed by nodes that exist today are populated; later
    phases add fields (tracks, ego_motion, kinematics, narrative, ...) and
    new nodes rather than changing how the graph is invoked.
    """

    incident_id: str
    job_id: str
    video_asset_id: str
    video_path: str
    sensor_log_path: str | None
    device_id: str | None

    ingest_ok: bool

    processed_video_path: str | None
    quality_score_pre: float | None
    quality_score_post: float | None
    was_undistorted: bool
    was_upscaled: bool

    error: str | None
