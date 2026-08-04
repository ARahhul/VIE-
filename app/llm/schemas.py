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
    grounded: bool = Field(default=True, description="set by claim verification, not the LLM")


class InvestigationNarrative(BaseModel):
    summary: str
    claims: list[EventClaim]


# Vertex AI's response_schema converter rejects the $ref/$defs indirection
# that Pydantic emits for a nested model ("Extra inputs are not permitted
# ... '#/$defs/EventClaim'"), so the constraint is spelled out inline and
# fully expanded. `grounded` is deliberately absent: claim verification
# sets it, not the model.
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "who": {"type": "STRING"},
                    "what": {"type": "STRING"},
                    "when_start_s": {"type": "NUMBER"},
                    "when_end_s": {"type": "NUMBER"},
                    "where": {"type": "STRING"},
                    "how": {"type": "STRING"},
                    "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
                },
                "required": ["who", "what", "when_start_s", "when_end_s", "where", "how", "confidence"],
            },
        },
    },
    "required": ["summary", "claims"],
}
