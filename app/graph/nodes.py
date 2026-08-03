import json
from pathlib import Path

import cv2

from app.core.config import settings
from app.db.base import SessionLocal
from app.db.models import DeviceProfile, VideoAsset
from app.graph.state import InvestigationState
from app.kinematics.absolute import fuse_absolute_speed
from app.kinematics.ego import compute_ego_motion
from app.kinematics.relative import compute_relative_motion
from app.kinematics.sensor_log import load_sensor_log
from app.perception.detect_track import detect_and_track, load_tracks
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
    if state.get("error"):
        return state

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


def detect_and_track_node(state: InvestigationState) -> InvestigationState:
    """YOLOv11 detection + ByteTrack multi-object tracking on the processed clip.

    Produces the per-object trajectory log (bbox, class, track ID, frame
    index, timestamp) that Phase 4's relative-motion estimation is built on.
    """
    if state.get("error"):
        return state

    db = SessionLocal()
    try:
        video_path = state.get("processed_video_path") or state["video_path"]
        incident_dir = settings.upload_dir / state["incident_id"]
        tracks_path = incident_dir / "tracks.json"

        summary = detect_and_track(video_path, str(tracks_path))

        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is not None:
            video_asset.tracks_path = summary.tracks_path
            video_asset.num_tracks = summary.num_tracks
            db.commit()

        return {**state, "tracks_path": summary.tracks_path, "num_tracks": summary.num_tracks}
    except Exception as exc:  # noqa: BLE001 - surfaced via state, job worker marks the job failed
        return {**state, "error": f"detect_and_track failed: {exc}"}
    finally:
        db.close()


def kinematics_node(state: InvestigationState) -> InvestigationState:
    """Fuses ego-motion (OBD/GPS/IMU) with vision-derived relative motion.

    ego_speed is measured directly; vision only has to solve the easier
    relative-motion problem. Without a sensor log, every speed is still
    computed but flagged vision-only/low-confidence rather than withheld.
    """
    if state.get("error"):
        return state

    db = SessionLocal()
    try:
        tracks = load_tracks(state["tracks_path"]) if state.get("tracks_path") else []

        video_path = state.get("processed_video_path") or state["video_path"]
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        relative_points = compute_relative_motion(tracks, frame_width) if tracks and frame_width else []

        ego_points = None
        if state.get("sensor_log_path"):
            samples = load_sensor_log(state["sensor_log_path"])
            ego_points = compute_ego_motion(samples)

        absolute_points = fuse_absolute_speed(relative_points, ego_points)

        methods = {p.method for p in absolute_points}
        if not methods:
            method_summary = "none"
        elif len(methods) == 1:
            method_summary = next(iter(methods))
        else:
            method_summary = "mixed"

        incident_dir = settings.upload_dir / state["incident_id"]
        kinematics_path = incident_dir / "kinematics.json"
        with open(kinematics_path, "w") as f:
            json.dump(
                [
                    {
                        "t": p.t,
                        "track_id": p.track_id,
                        "speed_mps": p.speed_mps,
                        "error_margin": p.error_margin,
                        "method": p.method,
                        "confidence": p.confidence,
                    }
                    for p in absolute_points
                ],
                f,
            )

        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is not None:
            video_asset.kinematics_path = str(kinematics_path)
            video_asset.kinematics_method = method_summary
            db.commit()

        return {
            **state,
            "kinematics_path": str(kinematics_path),
            "kinematics_method": method_summary,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced via state, job worker marks the job failed
        return {**state, "error": f"kinematics failed: {exc}"}
    finally:
        db.close()
