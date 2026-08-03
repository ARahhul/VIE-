from datetime import datetime

from pydantic import BaseModel


class VideoAssetOut(BaseModel):
    id: str
    original_filename: str
    size_bytes: int
    duration_s: float | None
    width: int | None
    height: int | None
    fps: float | None
    is_valid: bool
    validation_error: str | None

    model_config = {"from_attributes": True}


class ReportJobOut(BaseModel):
    id: str
    incident_id: str
    status: str
    error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    incident_id: str
    video_asset: VideoAssetOut
    job: ReportJobOut
