from pathlib import Path

from app.core.config import settings
from app.db.base import SessionLocal
from app.db.models import DeviceProfile, VideoAsset
from app.graph.state import InvestigationState
from app.preprocessing.pipeline import run_quality_gate


def ingest_node(state: InvestigationState) -> InvestigationState:
    """Confirms the persisted clip (and sensor log, if any) are on disk.

    Validation itself already happened synchronously in the /ingest endpoint
    before the job was enqueued; this node is the graph's record that the
    ingest stage ran, and the attachment point for the quality gate.
    """
    return {**state, "ingest_ok": True}


def quality_gate_node(state: InvestigationState) -> InvestigationState:
    """Undistort -> stabilize -> conditional super-resolution -> fps normalize.

    Fisheye undistortion runs here, first, before anything else touches the
    frames — a homography fitted downstream (Phase 4) to a warped ground
    plane produces speeds that look plausible but are wrong.
    """
    db = SessionLocal()
    try:
        device_profile = None
        if state.get("device_id"):
            device_profile = db.get(DeviceProfile, state["device_id"])

        incident_dir = settings.upload_dir / state["incident_id"]
        output_path = incident_dir / f"processed_{Path(state['video_path']).name}"

        result = run_quality_gate(state["video_path"], str(output_path), device_profile)

        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is not None:
            video_asset.processed_path = result.output_path
            video_asset.quality_score_pre = result.quality_pre.score
            video_asset.quality_score_post = result.quality_post.score
            video_asset.was_upscaled = result.was_upscaled
            video_asset.was_undistorted = result.was_undistorted
            db.commit()

        return {
            **state,
            "processed_video_path": result.output_path,
            "quality_score_pre": result.quality_pre.score,
            "quality_score_post": result.quality_post.score,
            "was_undistorted": result.was_undistorted,
            "was_upscaled": result.was_upscaled,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced via state, job worker marks the job failed
        return {**state, "error": f"quality_gate failed: {exc}"}
    finally:
        db.close()
