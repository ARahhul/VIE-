from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Incident, VideoAsset

router = APIRouter()


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
