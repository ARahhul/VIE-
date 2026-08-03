from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    detect_and_track_node,
    event_detection_node,
    ingest_node,
    kinematics_node,
    quality_gate_node,
)
from app.graph.state import InvestigationState


def build_graph():
    """Compiles the investigation StateGraph.

    Ingest -> quality gate -> event detection -> detect & track -> kinematics
    fusion are wired up so far. Later phases append video_llm_reasoning,
    claim_verification, report_generation, and persist_and_serve to this same
    graph rather than introducing a second pipeline mechanism.
    """
    graph = StateGraph(InvestigationState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("event_detection", event_detection_node)
    graph.add_node("detect_and_track", detect_and_track_node)
    graph.add_node("kinematics", kinematics_node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "quality_gate")
    graph.add_edge("quality_gate", "event_detection")
    graph.add_edge("event_detection", "detect_and_track")
    graph.add_edge("detect_and_track", "kinematics")
    graph.add_edge("kinematics", END)
    return graph.compile()


compiled_graph = build_graph()
