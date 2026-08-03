from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import ReportJob
from app.schemas import ReportJobOut

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=ReportJobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> ReportJobOut:
    job = db.get(ReportJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job
