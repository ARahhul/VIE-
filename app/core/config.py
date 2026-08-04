from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./vie.db"
    upload_dir: Path = Path("uploads")

    max_upload_size_bytes: int = 500 * 1024 * 1024  # 500MB
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")
    min_video_duration_s: float = 0.5
    max_video_duration_s: float = 600.0

    # Quality gate (Phase 2)
    quality_upscale_threshold: float = 0.5  # QualityReport.score below this triggers SR
    target_fps: float = 15.0

    # LangSmith — every LangGraph node run emits to one unified trace via these
    # env vars; nodes must never open manual create_run calls on top of this.
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "vie-investigation-engine"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Langfuse — prompt management/versioning for the Phase 6 video-LLM prompts.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Phase 6 — video-LLM reasoning backend. "none" until credentials exist;
    # the graph node degrades gracefully (no narrative, no error) when unset
    # rather than failing the whole investigation job.
    video_llm_backend: str = "none"  # none | gemini | qwen_vl | nvidia_nim
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-pro"

    # Gemini via Vertex AI (GCP service-account auth) instead of a plain
    # API key — used when google_genai_use_vertexai is true. Requires
    # google_cloud_project and a service-account key file at
    # google_application_credentials (standard GCP ADC env var; must be
    # exported to real os.environ in configure_tracing(), same reason as
    # Langfuse — google-auth's credential discovery reads raw os.environ,
    # not this Settings object).
    google_genai_use_vertexai: bool = False
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    google_application_credentials: str | None = None
    qwen_vl_endpoint: str | None = None  # self-hosted OpenAI-compatible endpoint
    qwen_vl_api_key: str | None = None
    qwen_vl_model: str = "qwen3-vl-32b-instruct"

    # NVIDIA NIM — free-tier hosted vision-language models (build.nvidia.com),
    # OpenAI-compatible API. Not video-native like Qwen3-VL: a single frame
    # is sampled and sent as an image instead of the whole clip. Verified
    # live: the hosted llama-3.2-90b-vision-instruct endpoint rejects more
    # than 1 image per request (400: "At most 1 image(s) may be provided in
    # one request") unless you're running it yourself with
    # --limit-mm-per-prompt raised, so this is a real API constraint, not
    # an arbitrary default.
    nvidia_nim_api_key: str | None = None
    nvidia_nim_model: str = "meta/llama-3.2-90b-vision-instruct"
    nvidia_nim_endpoint: str = "https://integrate.api.nvidia.com/v1"
    nvidia_nim_max_frames: int = 1


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
