"""
Pydantic schemas for API request/response validation.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Request schema for video generation."""

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Text description for video generation"
    )

    fps: int = Field(
        default=24,
        ge=1,
        le=60,
        description="Frames per second"
    )

    num_frames: int = Field(
        default=64,
        ge=8,
        le=240,
        description="Total number of frames to generate"
    )

    width: int = Field(
        default=720,
        ge=256,
        le=1920,
        description="Video width in pixels"
    )

    height: int = Field(
        default=720,
        ge=256,
        le=1080,
        description="Video height in pixels"
    )

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducibility"
    )

    prefer_backend: Optional[Literal["cogvideox", "opensora", "mochi"]] = Field(
        default=None,
        description="Preferred backend (will fallback if unavailable)"
    )

    low_vram: bool = Field(
        default=False,
        description="Enable low-VRAM optimizations"
    )

    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Ensure dimensions are multiples of 8 for most models."""
        if v % 8 != 0:
            # Round to nearest multiple of 8
            v = (v // 8) * 8
        return v


class GenerateResponse(BaseModel):
    """Response schema for video generation."""

    success: bool = Field(description="Whether generation succeeded")

    video_url: Optional[str] = Field(
        default=None,
        description="URL to download the generated video"
    )

    mp4_path: Optional[str] = Field(
        default=None,
        description="Local path to generated video file"
    )

    frames: Optional[int] = Field(
        default=None,
        description="Number of frames in generated video"
    )

    fps: Optional[int] = Field(
        default=None,
        description="Frames per second"
    )

    width: Optional[int] = Field(
        default=None,
        description="Video width"
    )

    height: Optional[int] = Field(
        default=None,
        description="Video height"
    )

    seed: Optional[int] = Field(
        default=None,
        description="Seed used for generation"
    )

    backend: Optional[str] = Field(
        default=None,
        description="Backend used for generation"
    )

    elapsed_time: Optional[float] = Field(
        default=None,
        description="Generation time in seconds"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )


class SystemInfoResponse(BaseModel):
    """Response schema for system information."""

    cuda_available: bool
    device_name: str
    total_vram_gb: float
    free_vram_gb: float
    used_vram_gb: float
    cpu_percent: float
    ram_available_gb: float
    ram_total_gb: float
    backend: str


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: Literal["healthy", "degraded", "unhealthy"]
    backend: str
    cuda_available: bool
    message: str
