from app.graph.nodes import video_llm_reasoning_node


def test_video_llm_reasoning_degrades_gracefully_without_backend():
    """No API key configured (the realistic default state): the node must
    not fail the job, just record that no narrative is available."""
    state = {"incident_id": "i1", "video_asset_id": "v1", "video_path": "unused.mp4"}

    result = video_llm_reasoning_node(state)

    assert result["narrative_available"] is False
    assert result["narrative_error"] is None
    assert "error" not in result or result["error"] is None


def test_video_llm_reasoning_skips_when_upstream_error_present():
    state = {"error": "quality_gate failed: boom"}
    result = video_llm_reasoning_node(state)
    assert result == state
