import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DeviceProfile(Base):
    """Per-device calibration, keyed on the VigilNetra unit (not per clip)."""

    __tablename__ = "device_profiles"

    device_id: Mapped[str] = mapped_column(String, primary_key=True)
    mount_height_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    yaw_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    intrinsics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fisheye_coeffs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    device_id: Mapped[str | None] = mapped_column(String, ForeignKey("device_profiles.device_id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    video_assets: Mapped[list["VideoAsset"]] = relationship(back_populates="incident")
    sensor_logs: Mapped[list["SensorLog"]] = relationship(back_populates="incident")
    report_jobs: Mapped[list["ReportJob"]] = relationship(back_populates="incident")


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    incident_id: Mapped[str] = mapped_column(String, ForeignKey("incidents.id"))

    file_path: Mapped[str] = mapped_column(String)
    original_filename: Mapped[str] = mapped_column(String)
    content_type: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer)

    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_valid: Mapped[bool] = mapped_column(default=False)
    validation_error: Mapped[str | None] = mapped_column(String, nullable=True)

    # Phase 2 — quality gate output
    processed_path: Mapped[str | None] = mapped_column(String, nullable=True)
    quality_score_pre: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score_post: Mapped[float | None] = mapped_column(Float, nullable=True)
    was_upscaled: Mapped[bool] = mapped_column(default=False)
    was_undistorted: Mapped[bool] = mapped_column(default=False)
    ssim_gain: Mapped[float | None] = mapped_column(Float, nullable=True)
    psnr_gain: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Phase 3 — detection & tracking output
    tracks_path: Mapped[str | None] = mapped_column(String, nullable=True)
    num_tracks: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Phase 4 — kinematics fusion output
    kinematics_path: Mapped[str | None] = mapped_column(String, nullable=True)
    kinematics_method: Mapped[str | None] = mapped_column(String, nullable=True)  # fused | vision-only | mixed | none

    # Phase 5 — event window
    event_window_start_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_window_end_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_window_source: Mapped[str | None] = mapped_column(String, nullable=True)  # sensor_log | optical_flow_residual | none
    event_window_confidence: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    incident: Mapped["Incident"] = relationship(back_populates="video_assets")


class SensorLog(Base):
    """Raw IMU/GPS/OBD time-series accompanying a clip, when the uploader has one."""

    __tablename__ = "sensor_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    incident_id: Mapped[str] = mapped_column(String, ForeignKey("incidents.id"))

    file_path: Mapped[str] = mapped_column(String)
    original_filename: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    incident: Mapped["Incident"] = relationship(back_populates="sensor_logs")


class ReportJob(Base):
    __tablename__ = "report_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    incident_id: Mapped[str] = mapped_column(String, ForeignKey("incidents.id"))

    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|completed|failed
    error: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    incident: Mapped["Incident"] = relationship(back_populates="report_jobs")
