"""Structured JSON output contract for the video-LLM reasoning pass.

Every claim is timestamp-cited and who/what/when/where/how — not a
free-form paragraph — so Phase 8's claim-verification pass can check each
one against the tracks/kinematics data the pipeline already trusts.
"""

from pydantic import BaseModel, Field


class EventClaim(BaseModel):
    who: str = Field(description="which actor(s): e.g. 'ego vehicle', 'track_id=3 (car)'")
    what: str = Field(description="what happened, plain language")
    when_start_s: float
    when_end_s: float
    where: str = Field(description="spatial description, e.g. 'intersection, ego's lane'")
    how: str = Field(description="the mechanism/manner, plain language")
    confidence: str = Field(description="high | medium | low")


class InvestigationNarrative(BaseModel):
    summary: str
    claims: list[EventClaim]
