"""Video-LLM backends. Two-pass grounding: coarse localization then fine
reasoning, both prompts from app.llm.prompts (Langfuse-managed).

Neither backend can be exercised in this environment (no API key / no
self-hosted endpoint configured yet) — they're written to be correct once
credentials exist, not to have been run here. get_backend() returns None
when unconfigured so callers degrade gracefully instead of crashing.
"""

import json
from abc import ABC, abstractmethod

from app.core.config import settings
from app.llm.prompts import coarse_localization_prompt, fine_reasoning_prompt
from app.llm.schemas import InvestigationNarrative


class VideoLLMBackend(ABC):
    @abstractmethod
    def reason(self, video_path: str, context_data: str) -> InvestigationNarrative: ...


class GeminiBackend(VideoLLMBackend):
    """Gemini 2.5/3 Pro via the google-genai SDK — the PRD's zero-shot-accuracy fallback."""

    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def reason(self, video_path: str, context_data: str) -> InvestigationNarrative:
        uploaded = self._client.files.upload(file=video_path)
        response = self._client.models.generate_content(
            model=self._model,
            contents=[uploaded, coarse_localization_prompt(), fine_reasoning_prompt(context_data)],
            config={"response_mime_type": "application/json"},
        )
        return InvestigationNarrative.model_validate(json.loads(response.text))


class QwenVLBackend(VideoLLMBackend):
    """Self-hosted Qwen3-VL behind an OpenAI-compatible endpoint (e.g. vLLM).

    Preferred per the PRD tech stack: native timestamp grounding, keeps
    footage off third-party servers. Sent as a video content part per the
    OpenAI-compatible vision API shape most self-hosted VLM servers expose.
    """

    def __init__(self, endpoint: str, api_key: str | None, model: str):
        from openai import OpenAI

        self._client = OpenAI(base_url=endpoint, api_key=api_key or "not-needed")
        self._model = model

    def reason(self, video_path: str, context_data: str) -> InvestigationNarrative:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": coarse_localization_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_path}},
                        {"type": "text", "text": fine_reasoning_prompt(context_data)},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        return InvestigationNarrative.model_validate(json.loads(response.choices[0].message.content))


def get_backend() -> VideoLLMBackend | None:
    if settings.video_llm_backend == "gemini" and settings.gemini_api_key:
        return GeminiBackend(settings.gemini_api_key, settings.gemini_model)
    if settings.video_llm_backend == "qwen_vl" and settings.qwen_vl_endpoint:
        return QwenVLBackend(settings.qwen_vl_endpoint, settings.qwen_vl_api_key, settings.qwen_vl_model)
    return None
