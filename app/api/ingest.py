from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.validation import validate_video_file
from app.db.base import get_db
from app.db.models import Incident, ReportJob, SensorLog, VideoAsset
from app.jobs.queue import enqueue
from app.schemas import IngestResponse

router = APIRouter()


async def _save_upload(upload: UploadFile, dest_dir: Path) -> tuple[Path, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / upload.filename
    size = 0
    with dest_path.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_size_bytes:
                out.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(413, "file exceeds max upload size")
            out.write(chunk)
    return dest_path, size


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    video: UploadFile = File(...),
    sensor_log: UploadFile | None = File(None),
    device_id: str | None = Form(None),
    db: Session = Depends(get_db),
) -> IngestResponse:
    incident = Incident(device_id=device_id)
    db.add(incident)
    db.flush()  # assign incident.id

    incident_dir = settings.upload_dir / incident.id
    video_path, video_size = await _save_upload(video, incident_dir)

    result = validate_video_file(video_path)
    video_asset = VideoAsset(
        incident_id=incident.id,
        file_path=str(video_path),
        original_filename=video.filename,
        content_type=video.content_type,
        size_bytes=video_size,
        duration_s=result.duration_s,
        width=result.width,
        height=result.height,
        fps=result.fps,
        is_valid=result.is_valid,
        validation_error=result.error,
    )
    db.add(video_asset)

    if not result.is_valid:
        db.commit()
        raise HTTPException(422, f"clip failed validation: {result.error}")

    sensor_log_path: str | None = None
    if sensor_log is not None:
        saved_path, _size = await _save_upload(sensor_log, incident_dir)
        db.add(
            SensorLog(
                incident_id=incident.id,
                file_path=str(saved_path),
                original_filename=sensor_log.filename,
            )
        )
        sensor_log_path = str(saved_path)

    job = ReportJob(incident_id=incident.id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(video_asset)
    db.refresh(job)

    await enqueue(
        {
            "incident_id": incident.id,
            "job_id": job.id,
            "video_path": str(video_path),
            "sensor_log_path": sensor_log_path,
            "device_id": device_id,
        }
    )

    return IngestResponse(incident_id=incident.id, video_asset=video_asset, job=job)
