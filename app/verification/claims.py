"""Claim verification — every LLM narrative statement gets cross-checked
against the detection/tracking/kinematics data the pipeline actually
computed, before it goes in a report. Mechanical, not another LLM call:
this pass exists specifically to catch the LLM inventing something the
data doesn't support.

Ungrounded claims aren't dropped — they're downgraded to low confidence and
flagged, so the report shows what wasn't verifiable instead of silently
smoothing it away.
"""

import re

from app.llm.schemas import EventClaim, InvestigationNarrative

_TRACK_ID_RE = re.compile(r"track_id=(\d+)")


def _referenced_track_id(who: str) -> int | None:
    m = _TRACK_ID_RE.search(who)
    return int(m.group(1)) if m else None


def verify_narrative(
    narrative: InvestigationNarrative,
    known_track_ids: set[int],
    video_duration_s: float,
) -> InvestigationNarrative:
    verified_claims: list[EventClaim] = []
    for claim in narrative.claims:
        grounded = True
        reasons: list[str] = []

        track_id = _referenced_track_id(claim.who)
        if track_id is not None and track_id not in known_track_ids:
            grounded = False
            reasons.append(f"track_id={track_id} was never detected")

        if claim.when_start_s < 0 or (video_duration_s and claim.when_end_s > video_duration_s + 1e-6):
            grounded = False
            reasons.append("time window falls outside the clip")

        if claim.when_end_s < claim.when_start_s:
            grounded = False
            reasons.append("end time precedes start time")

        confidence = claim.confidence
        what = claim.what
        if not grounded:
            confidence = "low"
            what = f"{claim.what} [unverified: {'; '.join(reasons)}]"

        verified_claims.append(claim.model_copy(update={"confidence": confidence, "what": what, "grounded": grounded}))

    return InvestigationNarrative(summary=narrative.summary, claims=verified_claims)
