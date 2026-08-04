from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Incident, ReportJob, VideoAsset

router = APIRouter()


class IncidentSummary(BaseModel):
    incident_id: str
    created_at: datetime
    device_id: str | None
    original_filename: str | None
    job_status: str | None
    report_available: bool
    num_tracks: int | None
    narrative_available: bool


@router.get("/incidents", response_model=list[IncidentSummary])
def list_incidents(db: Session = Depends(get_db)) -> list[IncidentSummary]:
    """Archives: every investigation run so far, newest first."""
    incidents = db.execute(select(Incident).order_by(Incident.created_at.desc())).scalars().all()

    results = []
    for incident in incidents:
        video_asset = (
            db.execute(
                select(VideoAsset).where(VideoAsset.incident_id == incident.id).order_by(VideoAsset.created_at.desc())
            )
            .scalars()
            .first()
        )
        job = (
            db.execute(
                select(ReportJob).where(ReportJob.incident_id == incident.id).order_by(ReportJob.created_at.desc())
            )
            .scalars()
            .first()
        )
        results.append(
            IncidentSummary(
                incident_id=incident.id,
                created_at=incident.created_at,
                device_id=incident.device_id,
                original_filename=video_asset.original_filename if video_asset else None,
                job_status=job.status if job else None,
                report_available=bool(video_asset.report_path) if video_asset else False,
                num_tracks=video_asset.num_tracks if video_asset else None,
                narrative_available=video_asset.narrative_available if video_asset else False,
            )
        )
    return results


@router.get("/incidents/{incident_id}/report")
def get_report(incident_id: str, db: Session = Depends(get_db)) -> FileResponse:
    incident = db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(404, "incident not found")

    video_asset = db.execute(
        select(VideoAsset).where(VideoAsset.incident_id == incident_id).order_by(VideoAsset.created_at.desc())
    ).scalars().first()
    if video_asset is None or not video_asset.report_path:
        raise HTTPException(404, "report not generated yet")

    return FileResponse(video_asset.report_path, media_type="application/pdf", filename=f"{incident_id}.pdf")
