"""Driver-state observations from the second, driver-facing camera.

Reported as observation only, never as cause (the no-fault-attribution rule
matters more here than anywhere else in the system): eyes-on/off-road,
hands-on-wheel, head orientation. This genuinely needs a video-LLM pass
(Phase 6) — there's no meaningful classical-CV substitute for "was the
driver looking at the road" — so it's a documented stub until a model
backend (Qwen3-VL/Gemini) with API credentials is wired up.
"""

from dataclasses import dataclass


@dataclass
class DriverStateObservation:
    start_s: float
    end_s: float
    observation: str  # plain-language, descriptive only — never causal


def observe_driver_state(driver_facing_video_path: str, event_window_start_s: float, event_window_end_s: float) -> list[DriverStateObservation]:
    raise NotImplementedError(
        "driver-state observation requires a video-LLM backend (Qwen3-VL or Gemini); "
        "wire this up once API credentials are configured (see app.core.config.settings)"
    )
