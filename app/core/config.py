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


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
