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
import re
from abc import ABC, abstractmethod

import cv2
from langfuse.decorators import langfuse_context, observe

from app.core.config import settings
from app.llm.prompts import reasoning_prompt
from app.llm.schemas import InvestigationNarrative


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Strip markdown code fences from a model response before json.loads.
    Some hosted models (e.g. llama-3.2-*-vision-instruct on NVIDIA NIM) ignore
    response_format=json_object and wrap the body in ```json ... ``` anyway."""
    m = _JSON_FENCE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _sample_frames_as_data_urls(video_path: str, max_frames: int, max_width: int = 640) -> list[tuple[float, str]]:
    """Evenly-spaced frames as (timestamp_s, data-URL) pairs, for backends
    that take images rather than a native video upload.

    Frames are downscaled to max_width pixels wide (aspect preserved) and
    JPEG-encoded at moderate quality — base64-encoded raw 1280x720 frames
    are ~3k tokens each in a VLM's context, which blows past an 8k
    context window at 3+ frames. 640px keeps each frame well under 1k
    tokens with no meaningful loss for scene understanding.
    """
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
            h, w = frame.shape[:2]
            if w > max_width:
                new_h = int(h * (max_width / w))
                frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
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


class QwenVLBackend(VideoLLMBackend):
    """Self-hosted Qwen-VL behind an OpenAI-compatible endpoint.

    Two deploy targets in practice: vLLM (which does support {"type":
    "video_url"} native video ingest) and LM Studio / llama.cpp (which
    does NOT — it only accepts images via {"type": "image_url"}). Sending
    evenly-spaced sampled frames as images works on both, so that's what
    we do — trading Qwen's native timestamp grounding for portability
    across the deploy targets people actually run this on.
    """

    def __init__(self, endpoint: str, api_key: str | None, model: str, max_frames: int):
        from openai import OpenAI

        self._client = OpenAI(base_url=endpoint, api_key=api_key or "not-needed")
        self._model = model
        self._max_frames = max_frames

    @observe(name="qwen_vl_reason", as_type="generation")
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
        )
        raw = response.choices[0].message.content or ""
        narrative = InvestigationNarrative.model_validate(json.loads(_extract_json_object(raw)))
        langfuse_context.update_current_observation(output=narrative.model_dump())
        return narrative


class NvidiaNIMBackend(VideoLLMBackend):
    """NVIDIA NIM (build.nvidia.com) hosted vision-language models — a
    free-tier alternative when no self-hosted Qwen3-VL is set up yet.

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
        raw = response.choices[0].message.content or ""
        narrative = InvestigationNarrative.model_validate(json.loads(_extract_json_object(raw)))
        langfuse_context.update_current_observation(output=narrative.model_dump())
        return narrative


def get_backend() -> VideoLLMBackend | None:
    if settings.video_llm_backend == "qwen_vl" and settings.qwen_vl_endpoint:
        return QwenVLBackend(
            settings.qwen_vl_endpoint,
            settings.qwen_vl_api_key,
            settings.qwen_vl_model,
            settings.qwen_vl_max_frames,
        )
    if settings.video_llm_backend == "nvidia_nim" and settings.nvidia_nim_api_key:
        return NvidiaNIMBackend(
            settings.nvidia_nim_endpoint,
            settings.nvidia_nim_api_key,
            settings.nvidia_nim_model,
            settings.nvidia_nim_max_frames,
        )
    return None
