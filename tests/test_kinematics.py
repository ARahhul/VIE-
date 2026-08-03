import json

from app.kinematics.absolute import fuse_absolute_speed
from app.kinematics.ego import compute_ego_motion
from app.kinematics.relative import compute_relative_motion
from app.kinematics.sensor_log import find_impact_timestamp, load_sensor_log
from app.perception.detect_track import TrackPoint


def test_ego_motion_prefers_obd_over_gps(tmp_path):
    log = [
        {"t": 0.0, "obd_speed_mps": 10.0, "gps_speed_mps": 9.5},
        {"t": 1.0, "obd_speed_mps": 11.0, "gps_speed_mps": 10.4},
    ]
    path = tmp_path / "log.json"
    path.write_text(json.dumps(log))

    samples = load_sensor_log(str(path))
    points = compute_ego_motion(samples)

    assert [p.speed_mps for p in points] == [10.0, 11.0]
    assert all(p.source == "obd" for p in points)


def test_ego_motion_falls_back_to_gps_derived_speed(tmp_path):
    # ~111m north over 10s at the equator ~= 11.1 m/s
    log = [
        {"t": 0.0, "gps_lat": 0.0, "gps_lon": 0.0},
        {"t": 10.0, "gps_lat": 0.001, "gps_lon": 0.0},
    ]
    path = tmp_path / "log.json"
    path.write_text(json.dumps(log))

    points = compute_ego_motion(load_sensor_log(str(path)))
    assert len(points) == 1
    assert 10.0 < points[0].speed_mps < 12.5
    assert points[0].source == "gps"


def test_find_impact_timestamp_picks_the_biggest_g_spike(tmp_path):
    log = [
        {"t": 0.0, "imu_ax": 0.0, "imu_ay": 0.0, "imu_az": 9.81},
        {"t": 1.0, "imu_ax": 0.0, "imu_ay": 0.0, "imu_az": 9.81},
        {"t": 2.0, "imu_ax": 40.0, "imu_ay": 0.0, "imu_az": 9.81},  # the crash
        {"t": 3.0, "imu_ax": 0.0, "imu_ay": 0.0, "imu_az": 9.81},
    ]
    path = tmp_path / "log.json"
    path.write_text(json.dumps(log))

    assert find_impact_timestamp(load_sensor_log(str(path))) == 2.0


def test_relative_motion_scales_pixel_displacement():
    tracks = [
        TrackPoint(0, 0.0, 1, 2, "car", [0, 100, 20, 140], 0.9),
        TrackPoint(1, 1.0, 1, 2, "car", [100, 100, 120, 140], 0.9),
    ]
    points = compute_relative_motion(tracks, frame_width=1000)
    assert len(points) == 1
    assert points[0].speed_mps > 0


def test_fuse_absolute_speed_uses_fused_method_when_ego_available():
    from app.kinematics.ego import EgoMotionPoint
    from app.kinematics.relative import RelativeMotionPoint

    relative = [RelativeMotionPoint(t=1.0, track_id=1, speed_mps=5.0)]
    ego = [EgoMotionPoint(t=1.0, speed_mps=10.0, heading_deg=None, source="obd")]

    result = fuse_absolute_speed(relative, ego)
    assert result[0].method == "fused"
    assert result[0].speed_mps == 15.0
    assert result[0].confidence == "medium"


def test_fuse_absolute_speed_flags_vision_only_without_sensor_log():
    from app.kinematics.relative import RelativeMotionPoint

    relative = [RelativeMotionPoint(t=1.0, track_id=1, speed_mps=5.0)]
    result = fuse_absolute_speed(relative, None)
    assert result[0].method == "vision-only"
    assert result[0].confidence == "low"
