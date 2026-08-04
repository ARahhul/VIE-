"""Video-LLM backends. Each takes the clip plus the pipeline's already-computed
perception data and returns a structured narrative (see app.llm.prompts).

get_backend() returns None when no backend is configured so callers
degrade gracefully instead of crashing. Each backend's reason() is traced
to Langfuse via @observe — model, the compact per-track context, and the
parsed narrative are all attached to the generation span, so a failure
(e.g. a context-length error) shows up with the exact input that caused it,
not just an exception message.
"""

import base64
import json
from abc import ABC, abstractmethod

import cv2
from langfuse.decorators import langfuse_context, observe

from app.core.config import settings
from app.llm.prompts import reasoning_prompt
from app.llm.schemas import GEMINI_RESPONSE_SCHEMA, InvestigationNarrative


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


MAX_INLINE_VIDEO_BYTES = 19 * 1024 * 1024  # Gemini inline-data request limit is ~20MB


class GeminiBackend(VideoLLMBackend):
    """Gemini 2.5/3 Pro via the google-genai SDK — the PRD's zero-shot-accuracy fallback.

    Two auth modes: a plain API key, or Vertex AI with a GCP service
    account (project + credentials file, no key). Vertex mode relies on
    GOOGLE_APPLICATION_CREDENTIALS already being exported to real
    os.environ by configure_tracing() — google-auth's ADC discovery reads
    that directly, it doesn't go through this Settings object.

    Sends the clip as inline bytes, not via client.files.upload(): verified
    live that Vertex AI raises "Vertex AI does not support creating files.
    You can upload files to GCS files instead" — the Files API is a
    Developer-API-only (API-key mode) feature. Inline bytes work
    identically in both auth modes, at the cost of a ~20MB size ceiling —
    fine given clips are already trimmed to short event windows.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        use_vertexai: bool = False,
        project: str | None = None,
        location: str = "us-central1",
    ):
        from google import genai

        if use_vertexai:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            self._client = genai.Client(api_key=api_key)
        self._model = model

    @observe(name="gemini_video_llm_reason", as_type="generation")
    def reason(self, video_path: str, context_data: str) -> InvestigationNarrative:
        from google.genai import types

        langfuse_context.update_current_observation(model=self._model, input={"video_path": video_path, "context_data": context_data})

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        if len(video_bytes) > MAX_INLINE_VIDEO_BYTES:
            raise ValueError(
                f"{video_path} is {len(video_bytes)} bytes, over the {MAX_INLINE_VIDEO_BYTES}-byte inline limit "
                "(GCS upload isn't wired up for the Vertex AI path yet)"
            )
        video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

        response = self._client.models.generate_content(
            model=self._model,
            contents=[video_part, reasoning_prompt(context_data)],
            # response_schema makes the shape a hard constraint rather than a
            # request the model can drift from — belt-and-braces alongside the
            # prompt now that a schema mismatch is a known real failure mode.
            # Inline dict, not the Pydantic model: Vertex AI's schema
            # converter rejects the $ref/$defs it emits for nested models
            # (verified live — "Extra inputs are not permitted
            # ... '#/$defs/EventClaim'").
            config={"response_mime_type": "application/json", "response_schema": GEMINI_RESPONSE_SCHEMA},
        )
        narrative = InvestigationNarrative.model_validate(json.loads(response.text))
        langfuse_context.update_current_observation(output=narrative.model_dump())
        return narrative


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

    @observe(name="qwen_vl_reason", as_type="generation")
    def reason(self, video_path: str, context_data: str) -> InvestigationNarrative:
        langfuse_context.update_current_observation(model=self._model, input={"video_path": video_path, "context_data": context_data})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_path}},
                        {"type": "text", "text": reasoning_prompt(context_data)},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        narrative = InvestigationNarrative.model_validate(json.loads(response.choices[0].message.content))
        langfuse_context.update_current_observation(output=narrative.model_dump())
        return narrative


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

    @observe(name="nvidia_nim_reason", as_type="generation")
    def reason(self, video_path: str, context_data: str) -> InvestigationNarrative:
        langfuse_context.update_current_observation(model=self._model, input={"video_path": video_path, "context_data": context_data})
        frames = _sample_frames_as_data_urls(video_path, self._max_frames)
        if not frames:
            raise ValueError(f"no frames could be sampled from {video_path}")

        content = []
        for t, data_url in frames:
            content.append({"type": "text", "text": f"Frame at t={t:.2f}s:"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": reasoning_prompt(context_data)})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        narrative = InvestigationNarrative.model_validate(json.loads(response.choices[0].message.content))
        langfuse_context.update_current_observation(output=narrative.model_dump())
        return narrative


def get_backend() -> VideoLLMBackend | None:
    if settings.video_llm_backend == "gemini":
        if settings.google_genai_use_vertexai and settings.google_cloud_project:
            return GeminiBackend(
                settings.gemini_model,
                use_vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
            )
        if settings.gemini_api_key:
            return GeminiBackend(settings.gemini_model, api_key=settings.gemini_api_key)
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
