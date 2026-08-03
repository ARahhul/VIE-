from langgraph.graph import END, StateGraph

from app.graph.nodes import ingest_node, quality_gate_node
from app.graph.state import InvestigationState


def build_graph():
    """Compiles the investigation StateGraph.

    Ingest -> quality gate are wired up so far. Later phases append
    detect_and_track, ego_motion, kinematics_fusion, video_llm_reasoning,
    claim_verification, report_generation, and persist_and_serve to this same
    graph rather than introducing a second pipeline mechanism.
    """
    graph = StateGraph(InvestigationState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "quality_gate")
    graph.add_edge("quality_gate", END)
    return graph.compile()


compiled_graph = build_graph()
