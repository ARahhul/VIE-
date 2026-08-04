"""Video-LLM backends. Two-pass grounding: coarse localization then fine
reasoning, both prompts from app.llm.prompts (Langfuse-managed).

Neither backend can be exercised in this environment (no API key / no
self-hosted endpoint configured yet) — they're written to be correct once
credentials exist, not to have been run here. get_backend() returns None
when unconfigured so callers degrade gracefully instead of crashing.
"""

import base64
import json
from abc import ABC, abstractmethod

import cv2

from app.core.config import settings
from app.llm.prompts import coarse_localization_prompt, fine_reasoning_prompt
from app.llm.schemas import InvestigationNarrative


def _sample_frames_as_data_urls(video_path: str, max_frames: int) -> list[tuple[float, str]]:
    """Evenly-spaced frames as (timestamp_s, data-URL) pairs, for backends
    that take images rather than a native video upload."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open {video_path} to sample frames")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return []

        if max_frames <= 1:
            indices = [frame_count // 2]
        elif frame_count <= max_frames:
            indices = list(range(frame_count))
        else:
            indices = [round(i * (frame_count - 1) / (max_frames - 1)) for i in range(max_frames)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
            frames.append((idx / fps, data_url))
        return frames
    finally:
        cap.release()


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


class NvidiaNIMBackend(VideoLLMBackend):
    """NVIDIA NIM (build.nvidia.com) hosted vision-language models — a
    free-tier alternative when no Qwen3-VL/Gemini access is set up yet.

    Not video-native like Qwen3-VL, and not even multi-frame by default: the
    hosted llama-3.2-90b-vision-instruct endpoint rejects more than 1 image
    per request (verified live — 400 "At most 1 image(s) may be provided in
    one request"), so a single representative frame (the clip's midpoint,
    via _sample_frames_as_data_urls with max_frames=1) stands in for the
    whole clip. That's a significant accuracy trade-off — this is the "get
    something running today" backend, not the target production one.
    """

    def __init__(self, endpoint: str, api_key: str, model: str, max_frames: int):
        from openai import OpenAI

        self._client = OpenAI(base_url=endpoint, api_key=api_key)
        self._model = model
        self._max_frames = max_frames

    def reason(self, video_path: str, context_data: str) -> InvestigationNarrative:
        frames = _sample_frames_as_data_urls(video_path, self._max_frames)
        if not frames:
            raise ValueError(f"no frames could be sampled from {video_path}")

        content = []
        for t, data_url in frames:
            content.append({"type": "text", "text": f"Frame at t={t:.2f}s:"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": fine_reasoning_prompt(context_data)})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": coarse_localization_prompt()},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        return InvestigationNarrative.model_validate(json.loads(response.choices[0].message.content))


def get_backend() -> VideoLLMBackend | None:
    if settings.video_llm_backend == "gemini" and settings.gemini_api_key:
        return GeminiBackend(settings.gemini_api_key, settings.gemini_model)
    if settings.video_llm_backend == "qwen_vl" and settings.qwen_vl_endpoint:
        return QwenVLBackend(settings.qwen_vl_endpoint, settings.qwen_vl_api_key, settings.qwen_vl_model)
    if settings.video_llm_backend == "nvidia_nim" and settings.nvidia_nim_api_key:
        return NvidiaNIMBackend(
            settings.nvidia_nim_endpoint,
            settings.nvidia_nim_api_key,
            settings.nvidia_nim_model,
            settings.nvidia_nim_max_frames,
        )
    return None
