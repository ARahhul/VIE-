import json

from app.llm.context import build_context_summary


def test_context_summary_is_compact_per_track_not_raw_frames(tmp_path):
    """Regression test: the old approach dumped every frame's raw bbox into
    the prompt and blew the model's context window on busy clips (verified
    live — 400 'maximum context length is 32768 tokens... requested 51933'
    on a 45-track clip). The summary must stay O(num_tracks), not
    O(num_tracks * num_frames)."""
    tracks = [
        {
            "frame_index": i,
            "timestamp_s": i / 10.0,
            "track_id": track_id,
            "class_id": 2,
            "class_name": "car",
            "bbox": [10.0, 10.0, 50.0, 50.0],
            "confidence": 0.9,
        }
        for track_id in range(50)
        for i in range(50)
    ]
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(json.dumps(tracks))

    kinematics_points = [
        {"t": i / 10.0, "track_id": track_id, "speed_mps": 10.0 + i, "error_margin": 0.15, "method": "fused", "confidence": "medium"}
        for track_id in range(50)
        for i in range(5)
    ]

    context = build_context_summary(str(tracks_path), kinematics_points, {"start_s": 0.0, "end_s": 5.0, "source": "sensor_log"})

    assert context["num_tracked_objects"] == 50
    assert len(context["tracked_objects"]) == 50
    # one compact row per track, not 2500 raw frame entries
    assert len(json.dumps(context)) < len(json.dumps(tracks))

    row = context["tracked_objects"][0]
    assert row["class_name"] == "car"
    assert row["min_speed_mps"] is not None
    assert row["kinematics_method"] == "fused"


def test_context_summary_handles_missing_tracks_and_kinematics():
    context = build_context_summary(None, [], {"start_s": None, "end_s": None, "source": "none"})
    assert context["num_tracked_objects"] == 0
    assert context["tracked_objects"] == []
