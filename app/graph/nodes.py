import json
from pathlib import Path

import cv2
from langfuse.decorators import langfuse_context

from app.core.config import settings
from app.db.base import SessionLocal
from app.db.models import DeviceProfile, VideoAsset
from app.events.event_window import select_event_window
from app.graph.state import InvestigationState
from app.kinematics.absolute import fuse_absolute_speed
from app.kinematics.ego import compute_ego_motion
from app.kinematics.relative import compute_relative_motion
from app.kinematics.sensor_log import load_sensor_log
from app.llm.backends import get_backend
from app.llm.context import build_context_summary
from app.llm.schemas import InvestigationNarrative
from app.perception.detect_track import detect_and_track, load_tracks
from app.preprocessing.pipeline import run_quality_gate
from app.report.build import build_report_context
from app.report.render import render_html, render_pdf
from app.verification.claims import verify_narrative


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
        incident_dir.mkdir(parents=True, exist_ok=True)
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


def event_detection_node(state: InvestigationState) -> InvestigationState:
    """Selects the event window: sensor-log impact timestamp when available
    (exact, hard trigger), optical-flow-residual spike as the fallback.
    """
    if state.get("error"):
        return state

    db = SessionLocal()
    try:
        samples = load_sensor_log(state["sensor_log_path"]) if state.get("sensor_log_path") else None
        video_path = state.get("processed_video_path") or state["video_path"]

        window = select_event_window(video_path, samples)

        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is not None:
            video_asset.event_window_start_s = window.start_s
            video_asset.event_window_end_s = window.end_s
            video_asset.event_window_source = window.source
            video_asset.event_window_confidence = window.confidence
            db.commit()

        return {
            **state,
            "event_window_start_s": window.start_s,
            "event_window_end_s": window.end_s,
            "event_window_source": window.source,
            "event_window_confidence": window.confidence,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced via state, job worker marks the job failed
        return {**state, "error": f"event_detection failed: {exc}"}
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
        incident_dir.mkdir(parents=True, exist_ok=True)
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
        incident_dir.mkdir(parents=True, exist_ok=True)
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


def video_llm_reasoning_node(state: InvestigationState) -> InvestigationState:
    """Two-pass grounded narrative from the video-LLM, injected with the
    perception-layer data (tracks, kinematics) so it reasons over numbers the
    pipeline already trusts rather than eyeballing the footage for speed.

    Degrades gracefully: without a configured backend (no API key yet), this
    is a no-op rather than a failure — narrative_available=False, no error,
    job still completes. A configured backend that errors at call time is
    also non-fatal, recorded in narrative_error instead of failing the job.
    """
    if state.get("error"):
        return state

    backend = get_backend()
    if backend is None:
        return {**state, "narrative_available": False, "narrative_error": None}

    db = SessionLocal()
    try:
        kinematics_path = state.get("kinematics_path")
        kinematics_points = json.load(open(kinematics_path)) if kinematics_path else []
        context = build_context_summary(
            state.get("tracks_path"),
            kinematics_points,
            {
                "start_s": state.get("event_window_start_s"),
                "end_s": state.get("event_window_end_s"),
                "source": state.get("event_window_source"),
            },
        )
        video_path = state.get("processed_video_path") or state["video_path"]
        narrative = backend.reason(video_path, json.dumps(context))
        langfuse_context.flush()  # short-lived callers (CLI/benchmark) may exit before the background batcher fires

        incident_dir = settings.upload_dir / state["incident_id"]
        incident_dir.mkdir(parents=True, exist_ok=True)
        narrative_path = incident_dir / "narrative.json"
        with open(narrative_path, "w") as f:
            f.write(narrative.model_dump_json())

        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is not None:
            video_asset.narrative_path = str(narrative_path)
            video_asset.narrative_available = True
            video_asset.narrative_error = None
            db.commit()

        return {**state, "narrative_path": str(narrative_path), "narrative_available": True, "narrative_error": None}
    except Exception as exc:  # noqa: BLE001 - non-fatal: report generation flags the gap instead
        langfuse_context.flush()
        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is not None:
            video_asset.narrative_available = False
            video_asset.narrative_error = str(exc)
            db.commit()
        return {**state, "narrative_available": False, "narrative_error": str(exc)}
    finally:
        db.close()


def claim_verification_node(state: InvestigationState) -> InvestigationState:
    """Cross-checks every narrative claim against tracks/timing data.

    Mechanical, not another LLM call — this is what catches the LLM
    inventing a track or a time that doesn't exist. Ungrounded claims are
    downgraded to low confidence and annotated, not dropped.
    """
    if state.get("error"):
        return state
    if not state.get("narrative_available"):
        return {**state, "narrative_verified": False}

    db = SessionLocal()
    try:
        with open(state["narrative_path"]) as f:
            narrative = InvestigationNarrative.model_validate_json(f.read())

        tracks = load_tracks(state["tracks_path"]) if state.get("tracks_path") else []
        known_track_ids = {t.track_id for t in tracks}

        video_path = state.get("processed_video_path") or state["video_path"]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration_s = frame_count / fps if fps > 0 else 0.0

        verified = verify_narrative(narrative, known_track_ids, duration_s)
        with open(state["narrative_path"], "w") as f:
            f.write(verified.model_dump_json())

        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is not None:
            video_asset.narrative_verified = True
            db.commit()

        return {**state, "narrative_verified": True}
    except Exception as exc:  # noqa: BLE001 - surfaced via state, job worker marks the job failed
        return {**state, "error": f"claim_verification failed: {exc}"}
    finally:
        db.close()


def report_generation_node(state: InvestigationState) -> InvestigationState:
    """Renders the structured investigation report (PRD Section 7 schema) to PDF."""
    if state.get("error"):
        return state

    db = SessionLocal()
    try:
        video_asset = db.get(VideoAsset, state["video_asset_id"])
        if video_asset is None:
            return {**state, "error": "report_generation failed: video_asset not found"}

        incident_dir = settings.upload_dir / state["incident_id"]
        incident_dir.mkdir(parents=True, exist_ok=True)
        ctx = build_report_context(video_asset, incident_dir / "evidence", state.get("sensor_log_path"))
        html = render_html(ctx)
        pdf_bytes = render_pdf(html)

        report_path = incident_dir / "report.pdf"
        report_path.write_bytes(pdf_bytes)

        video_asset.report_path = str(report_path)
        db.commit()

        return {**state, "report_path": str(report_path)}
    except Exception as exc:  # noqa: BLE001 - surfaced via state, job worker marks the job failed
        return {**state, "error": f"report_generation failed: {exc}"}
    finally:
        db.close()
