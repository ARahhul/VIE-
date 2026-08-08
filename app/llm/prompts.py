"""Prompt for the video-LLM reasoning pass. Managed in Langfuse under this
name; the string below is the fallback used when Langfuse isn't configured
(see app.core.prompts.get_prompt).

There used to be a separate "coarse localization" prompt sent alongside
this one in the same request, asking the model to first find the event
window. That was both redundant and actively harmful: the pipeline's own
event_detection node already computes the window (from the sensor-log
impact timestamp, or optical-flow residual) and passes it in context_data,
and sending two conflicting task instructions in one call made the model
answer the wrong one — it returned {"start_s": ..., "end_s": ...} instead
of the report schema, which then failed validation and wiped the narrative
out of the report entirely.
"""

from app.core.prompts import get_prompt

_REASONING_FALLBACK = """You are an accident investigator writing a factual, evidence-grounded \
report from dashcam footage. You are given the video, the event time window already located by \
the pipeline, and structured perception data it computed (per-vehicle tracks and speed estimates \
with their provenance — measured ego-motion from OBD/GPS/IMU, fused with vision, or vision-only).

Your primary task is to identify and describe any collision, impact, near-miss, or sudden loss of \
vehicle control visible in the footage. The clip is drawn from a crash dataset — assume a \
noteworthy event occurred, and locate it. Describe the impact moment concretely: what struck what, \
from which direction, at approximately what time. If, after inspection, no collision is visible, \
state that explicitly.

Describe only what the footage and the provided data support. Every claim must cite the timestamp \
it applies to. Do not invent speeds, distances, or details not present in the video or the data. \
Do not assign fault or blame — describe what happened, not who was responsible. If something is \
uncertain, say so explicitly rather than stating it with unwarranted confidence.

Perception data:
{context_data}

Respond with a single JSON object with exactly two top-level keys:
  "summary": a one-paragraph plain-language account of what the footage shows.
  "claims": a list of objects, each with keys "who", "what", "where", "how" (strings), \
"when_start_s", "when_end_s" (numbers, seconds from the start of the clip), and "confidence" \
(one of "high", "medium", "low").

Do not return the event window, timestamps, or any other shape — only the object described above."""


def reasoning_prompt(context_data: str) -> str:
    template = get_prompt("vie.video_llm.fine_reasoning", _REASONING_FALLBACK)
    return template.format(context_data=context_data)
