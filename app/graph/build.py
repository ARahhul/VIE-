from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    claim_verification_node,
    detect_and_track_node,
    event_detection_node,
    ingest_node,
    kinematics_node,
    quality_gate_node,
    report_generation_node,
    video_llm_reasoning_node,
)
from app.graph.state import InvestigationState


def build_graph():
    """Compiles the investigation StateGraph: ingest -> quality gate -> event
    detection -> detect & track -> kinematics fusion -> video-LLM reasoning ->
    claim verification -> report generation. persist_and_serve is implicit —
    every node already persists its own output to the DB/filesystem as it
    runs, rather than a final node writing everything at once.
    """
    graph = StateGraph(InvestigationState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("event_detection", event_detection_node)
    graph.add_node("detect_and_track", detect_and_track_node)
    graph.add_node("kinematics", kinematics_node)
    graph.add_node("video_llm_reasoning", video_llm_reasoning_node)
    graph.add_node("claim_verification", claim_verification_node)
    graph.add_node("report_generation", report_generation_node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "quality_gate")
    graph.add_edge("quality_gate", "event_detection")
    graph.add_edge("event_detection", "detect_and_track")
    graph.add_edge("detect_and_track", "kinematics")
    graph.add_edge("kinematics", "video_llm_reasoning")
    graph.add_edge("video_llm_reasoning", "claim_verification")
    graph.add_edge("claim_verification", "report_generation")
    graph.add_edge("report_generation", END)
    return graph.compile()


compiled_graph = build_graph()
