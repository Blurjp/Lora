"""
Configuration management using pydantic-settings.
Reads from .env file or environment variables with sensible defaults.
"""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with defaults."""

    # Backend configuration
    PREFERRED_BACKEND: Literal["cogvideox", "opensora", "mochi"] = "cogvideox"
    LOW_VRAM: bool = False

    # Default generation settings
    DEFAULT_FPS: int = 24
    DEFAULT_FRAMES: int = 64
    DEFAULT_WIDTH: int = 720
    DEFAULT_HEIGHT: int = 720

    # Device configuration
    DEVICE: Literal["cuda", "cpu"] = "cuda"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Paths
    WEIGHTS_DIR: Path = Path("./weights")
    OUTPUTS_DIR: Path = Path("./outputs")
    LOGS_DIR: Path = Path("./logs")

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
