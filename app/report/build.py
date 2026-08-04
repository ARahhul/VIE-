"""Assembles the report context: PRD Section 7's schema, given whatever the
pipeline actually produced. Every section degrades explicitly rather than
silently — a missing narrative shows as a stated gap, not an empty page.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from app.db.models import VideoAsset
from app.kinematics.ego import compute_ego_motion
from app.kinematics.sensor_log import load_sensor_log
from app.perception.detect_track import TrackPoint, load_tracks

DISCLAIMER = (
    "This is an AI-assisted draft investigation report, not a certified forensic or legal "
    "document. It describes what the footage and sensor data support; it does not determine "
    "fault or liability."
)


@dataclass
class KinematicsSummaryRow:
    actor: str
    class_name: str | None
    min_speed_mps: float
    max_speed_mps: float
    mean_speed_mps: float
    method: str
    confidence: str


@dataclass
class EvidenceFrame:
    track_id: int
    timestamp_s: float
    image_path: str


@dataclass
class ConfidenceEntry:
    claim: str
    confidence: str
    source_module: str


@dataclass
class ReportContext:
    incident_id: str
    executive_summary: str
    event_timeline: list[dict] = field(default_factory=list)
    kinematics: list[KinematicsSummaryRow] = field(default_factory=list)
    contributing_factors: list[str] = field(default_factory=list)
    evidence_frames: list[EvidenceFrame] = field(default_factory=list)
    confidence_appendix: list[ConfidenceEntry] = field(default_factory=list)
    driver_state_observations: list[str] = field(default_factory=list)
    driver_state_note: str | None = None
    disclaimer: str = DISCLAIMER


def _summarize_kinematics(kinematics_path: str | None) -> list[KinematicsSummaryRow]:
    if not kinematics_path or not Path(kinematics_path).exists():
        return []
    with open(kinematics_path) as f:
        points = json.load(f)

    by_track: dict[int, list[dict]] = {}
    for p in points:
        by_track.setdefault(p["track_id"], []).append(p)

    rows = []
    for track_id, pts in by_track.items():
        speeds = [p["speed_mps"] for p in pts]
        rows.append(
            KinematicsSummaryRow(
                actor=f"track_id={track_id}",
                class_name=None,
                min_speed_mps=min(speeds),
                max_speed_mps=max(speeds),
                mean_speed_mps=sum(speeds) / len(speeds),
                method=pts[-1]["method"],
                confidence=pts[-1]["confidence"],
            )
        )
    return rows


def _ego_vehicle_row(sensor_log_path: str | None) -> KinematicsSummaryRow | None:
    if not sensor_log_path or not Path(sensor_log_path).exists():
        return None
    samples = load_sensor_log(sensor_log_path)
    ego_points = compute_ego_motion(samples)
    if not ego_points:
        return None
    speeds = [p.speed_mps for p in ego_points]
    return KinematicsSummaryRow(
        actor="ego_vehicle",
        class_name="ego",
        min_speed_mps=min(speeds),
        max_speed_mps=max(speeds),
        mean_speed_mps=sum(speeds) / len(speeds),
        method="obd_gps",
        confidence="high",
    )


def _class_names_by_track(tracks_path: str | None) -> dict[int, str]:
    if not tracks_path or not Path(tracks_path).exists():
        return {}
    tracks = load_tracks(tracks_path)
    names: dict[int, str] = {}
    for t in tracks:
        names.setdefault(t.track_id, t.class_name)
    return names


def extract_evidence_frames(
    video_path: str, tracks_path: str | None, output_dir: Path, max_frames: int = 5
) -> list[EvidenceFrame]:
    if not tracks_path or not Path(tracks_path).exists():
        return []
    tracks = load_tracks(tracks_path)
    if not tracks:
        return []

    first_by_track: dict[int, TrackPoint] = {}
    for t in tracks:
        if t.track_id not in first_by_track:
            first_by_track[t.track_id] = t

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames: list[EvidenceFrame] = []
    try:
        for track_id, tp in list(first_by_track.items())[:max_frames]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tp.frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = (int(v) for v in tp.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, f"track {track_id}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
            image_path = output_dir / f"evidence_track_{track_id}.jpg"
            cv2.imwrite(str(image_path), frame)
            frames.append(EvidenceFrame(track_id=track_id, timestamp_s=tp.timestamp_s, image_path=str(image_path)))
    finally:
        cap.release()
    return frames


def build_report_context(
    video_asset: VideoAsset, evidence_dir: Path, sensor_log_path: str | None = None
) -> ReportContext:
    class_names = _class_names_by_track(video_asset.tracks_path)
    kinematics = _summarize_kinematics(video_asset.kinematics_path)
    for row in kinematics:
        track_id = int(row.actor.split("=")[1])
        row.class_name = class_names.get(track_id)

    ego_row = _ego_vehicle_row(sensor_log_path)
    if ego_row:
        kinematics.insert(0, ego_row)

    narrative = None
    if video_asset.narrative_available and video_asset.narrative_path and Path(video_asset.narrative_path).exists():
        with open(video_asset.narrative_path) as f:
            narrative = json.load(f)

    if narrative:
        executive_summary = narrative["summary"]
        event_timeline = narrative["claims"]
        contributing_factors = [c["what"] for c in narrative["claims"]]
        confidence_appendix = [
            ConfidenceEntry(claim=c["what"], confidence=c["confidence"], source_module="video_llm_reasoning")
            for c in narrative["claims"]
        ]
    else:
        # A raw exception/stack trace is not a report sentence — the previous
        # version interpolated narrative_error verbatim into the executive
        # summary, so a Pydantic validation failure rendered as a wall of
        # "2 validation errors for InvestigationNarrative ... errors.pydantic.dev"
        # where the reader expects prose. Keep the reason human-readable here;
        # the full error is still recorded on the VideoAsset row for debugging.
        reason = "no video-LLM backend is configured"
        if video_asset.narrative_error:
            reason = "the video-LLM reasoning step did not complete successfully"
        executive_summary = (
            f"No narrative account of this event was generated, because {reason}. "
            "The sections below reflect only the measured and computed pipeline output — "
            "detected objects, their tracks, and speed estimates — not a generated account "
            "of what happened."
        )
        event_timeline = []
        contributing_factors = []
        confidence_appendix = []

    confidence_appendix += [
        ConfidenceEntry(claim=f"{row.actor} speed", confidence=row.confidence, source_module="kinematics_fusion")
        for row in kinematics
    ]

    evidence_frames = extract_evidence_frames(
        video_asset.processed_path or video_asset.file_path, video_asset.tracks_path, evidence_dir
    )

    return ReportContext(
        incident_id=video_asset.incident_id,
        executive_summary=executive_summary,
        event_timeline=event_timeline,
        kinematics=kinematics,
        contributing_factors=contributing_factors,
        evidence_frames=evidence_frames,
        confidence_appendix=confidence_appendix,
        driver_state_observations=[],
        driver_state_note="Driver-facing camera analysis requires a video-LLM backend; not available yet.",
    )
