"""Parses the VigilNetra sensor log: raw IMU/GPS/OBD time-series accompanying
a clip. JSON schema (one object per sample, uploaded alongside the video):

    [{"t": 0.0, "obd_speed_mps": 12.3, "gps_speed_mps": 12.1,
      "gps_lat": 12.97, "gps_lon": 77.59,
      "imu_ax": 0.1, "imu_ay": -0.2, "imu_az": 9.8}, ...]

All fields except "t" are optional — a real VigilNetra unit logs all of
them, but partial logs (GPS-only, IMU-only) still parse.
"""

import json
import math
from dataclasses import dataclass


@dataclass
class SensorSample:
    t: float
    obd_speed_mps: float | None = None
    gps_speed_mps: float | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    imu_ax: float | None = None
    imu_ay: float | None = None
    imu_az: float | None = None

    @property
    def imu_magnitude(self) -> float | None:
        if self.imu_ax is None or self.imu_ay is None or self.imu_az is None:
            return None
        return math.sqrt(self.imu_ax**2 + self.imu_ay**2 + self.imu_az**2)


def load_sensor_log(path: str) -> list[SensorSample]:
    with open(path) as f:
        raw = json.load(f)
    return [SensorSample(**r) for r in raw]


def find_impact_timestamp(samples: list[SensorSample], gravity_mps2: float = 9.81) -> float | None:
    """The hard, exact event trigger: the IMU G-force spike / airbag tap.

    Approximated here as the sample whose acceleration magnitude deviates
    most from gravity — a real airbag-optocoupler tap would just be another
    field on SensorSample, checked first when present.
    """
    candidates = [(s.t, abs(s.imu_magnitude - gravity_mps2)) for s in samples if s.imu_magnitude is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[1])[0]
