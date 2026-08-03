from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.kinematics.calibrate import upsert_device_profile

router = APIRouter()


class DeviceProfileIn(BaseModel):
    mount_height_m: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    intrinsics: dict | None = None
    fisheye_coeffs: dict | None = None


class DeviceProfileOut(DeviceProfileIn):
    device_id: str

    model_config = {"from_attributes": True}


@router.put("/device-profiles/{device_id}", response_model=DeviceProfileOut)
def put_device_profile(device_id: str, body: DeviceProfileIn, db: Session = Depends(get_db)) -> DeviceProfileOut:
    profile = upsert_device_profile(db, device_id, **body.model_dump())
    return profile
