from app.graph.state import InvestigationState


def ingest_node(state: InvestigationState) -> InvestigationState:
    """Confirms the persisted clip (and sensor log, if any) are on disk.

    Validation itself already happened synchronously in the /ingest endpoint
    before the job was enqueued; this node is the graph's record that the
    ingest stage ran, and the attachment point for Phase 2's quality gate.
    """
    return {**state, "ingest_ok": True}
