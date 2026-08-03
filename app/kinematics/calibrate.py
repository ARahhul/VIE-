"""Per-device calibration registry — keyed on device_id, not per clip.

Intrinsics + fisheye coefficients are measured once per camera module with a
checkerboard; mount height/pitch/yaw are measured once per vehicle
installation. Both are fixed until the unit is reinstalled or swapped.
"""

from sqlalchemy.orm import Session

from app.db.models import DeviceProfile


def upsert_device_profile(
    db: Session,
    device_id: str,
    mount_height_m: float | None = None,
    pitch_deg: float | None = None,
    yaw_deg: float | None = None,
    intrinsics: dict | None = None,
    fisheye_coeffs: dict | None = None,
) -> DeviceProfile:
    profile = db.get(DeviceProfile, device_id)
    if profile is None:
        profile = DeviceProfile(device_id=device_id)
        db.add(profile)

    if mount_height_m is not None:
        profile.mount_height_m = mount_height_m
    if pitch_deg is not None:
        profile.pitch_deg = pitch_deg
    if yaw_deg is not None:
        profile.yaw_deg = yaw_deg
    if intrinsics is not None:
        profile.intrinsics = intrinsics
    if fisheye_coeffs is not None:
        profile.fisheye_coeffs = fisheye_coeffs

    db.commit()
    db.refresh(profile)
    return profile
