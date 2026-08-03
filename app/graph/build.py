from langgraph.graph import END, StateGraph

from app.graph.nodes import ingest_node
from app.graph.state import InvestigationState


def build_graph():
    """Compiles the investigation StateGraph.

    Phase 1 wires up only the ingest node. Later phases append quality_gate,
    detect_and_track, ego_motion, kinematics_fusion, video_llm_reasoning,
    claim_verification, report_generation, and persist_and_serve to this same
    graph rather than introducing a second pipeline mechanism.
    """
    graph = StateGraph(InvestigationState)
    graph.add_node("ingest", ingest_node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", END)
    return graph.compile()


compiled_graph = build_graph()
