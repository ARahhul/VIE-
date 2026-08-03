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


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
