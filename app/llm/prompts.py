"""Two-pass grounding prompts: a coarse pass finds where in the clip the
event happens, a fine pass reasons over that window in detail. Managed in
Langfuse under these names; the strings below are the fallback used when
Langfuse isn't configured (see app.core.prompts.get_prompt).
"""

from app.core.prompts import get_prompt

_COARSE_FALLBACK = """You are reviewing dashcam footage from a vehicle-mounted camera for an \
accident investigation. Watch the clip and identify the time window (in seconds from the start \
of the clip) where the notable event (collision, near-miss, hard braking, etc.) occurs. Respond \
with just the start and end second of that window."""

_FINE_FALLBACK = """You are an accident investigator writing a factual, evidence-grounded report \
from dashcam footage. You are given the video, the event time window, and structured perception \
data already computed by the pipeline (per-vehicle tracks and speed/acceleration estimates with \
their provenance — measured ego-motion from OBD/GPS/IMU, fused with vision, or vision-only).

Describe only what the footage and the provided data support. Every claim must cite the timestamp \
it applies to. Do not invent speeds, distances, or details not present in the video or the data. \
Do not assign fault or blame — describe what happened, not who was responsible. If something is \
uncertain, say so explicitly rather than stating it with unwarranted confidence.

Perception data:
{context_data}

Return a JSON object matching this schema: {{"summary": str, "claims": [{{"who": str, "what": \
str, "when_start_s": float, "when_end_s": float, "where": str, "how": str, "confidence": \
"high"|"medium"|"low"}}]}}"""


def coarse_localization_prompt() -> str:
    return get_prompt("vie.video_llm.coarse_localization", _COARSE_FALLBACK)


def fine_reasoning_prompt(context_data: str) -> str:
    template = get_prompt("vie.video_llm.fine_reasoning", _FINE_FALLBACK)
    return template.format(context_data=context_data)
