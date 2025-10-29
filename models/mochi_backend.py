"""
Mochi (or alternative) backend for text+image to video generation.

Uses Hugging Face diffusers library with available video generation models.
Falls back to smaller models for low-VRAM scenarios.

Recommended models:
- Mochi 1: genmo/mochi-1-preview (requires ~18GB VRAM)
- Alternative: damo-vilab/text-to-video-ms-1.7b (~8GB VRAM)
- Low-VRAM: cerspense/zeroscope_v2_576w (~4GB VRAM)
"""
import logging
import time
from pathlib import Path
from typing import Optional, Dict
import torch
import numpy as np
from PIL import Image
import imageio

logger = logging.getLogger("video_service")


class BackendNotAvailable(Exception):
    """Raised when backend cannot be initialized."""
    pass


class MochiBackend:
    """
    Mochi/alternative backend using Hugging Face diffusers.

    Provides a working fallback for video generation with lower VRAM requirements.
    """

    def __init__(
        self,
        weights_dir: Path,
        device: str = "cuda",
        low_vram: bool = False
    ):
        """
        Initialize Mochi backend.

        Args:
            weights_dir: Path to weights directory
            device: torch device ('cuda' or 'cpu')
            low_vram: Enable low-VRAM optimizations
        """
        self.weights_dir = Path(weights_dir) / "mochi"
        self.device = device
        self.low_vram = low_vram
        self.pipeline = None

        logger.info(f"Initializing Mochi backend on {device}")

        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA not available, falling back to CPU (very slow)")
            self.device = "cpu"

        self._load_pipeline()

    def _load_pipeline(self):
        """
        Load diffusion pipeline for video generation.

        Tries multiple models in order of preference based on VRAM availability.
        """
        try:
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            from diffusers.utils import export_to_video

        except ImportError as e:
            raise BackendNotAvailable(
                f"Diffusers library not installed: {e}. "
                f"Install with: pip install diffusers transformers accelerate"
            )

        # Select model based on VRAM availability
        if self.low_vram:
            model_id = "cerspense/zeroscope_v2_576w"
            logger.info("Low-VRAM mode: using ZeroScope v2 576w")
        else:
            # Try to use better model if VRAM available
            model_id = "damo-vilab/text-to-video-ms-1.7b"
            logger.info("Using ModelScope Text-to-Video")

        try:
            # Check if model exists locally
            local_path = self.weights_dir / model_id.split("/")[-1]

            if local_path.exists():
                logger.info(f"Loading model from local path: {local_path}")
                model_path = str(local_path)
            else:
                logger.info(f"Loading model from Hugging Face: {model_id}")
                model_path = model_id

            # Load pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )

            # Optimize for low VRAM
            if self.low_vram and self.device == "cuda":
                logger.info("Enabling low-VRAM optimizations")
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()

                # Try to enable xformers if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception:
                    logger.debug("xformers not available, using standard attention")

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # Use faster scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            logger.info("Mochi backend initialized successfully")

        except Exception as e:
            raise BackendNotAvailable(
                f"Failed to load video generation pipeline: {e}\n"
                f"Try: huggingface-cli download {model_id} --local-dir {self.weights_dir / model_id.split('/')[-1]}"
            )

    def generate_video(
        self,
        prompt: str,
        ref_image_path: Optional[str] = None,
        fps: int = 24,
        num_frames: int = 64,
        width: int = 720,
        height: int = 720,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Generate video from text and optional reference image.

        Args:
            prompt: Text description
            ref_image_path: Optional path to reference image (limited support)
            fps: Frames per second
            num_frames: Total number of frames
            width: Video width
            height: Video height
            seed: Random seed for reproducibility

        Returns:
            Dictionary with generation results
        """
        start_time = time.time()

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            # Use smaller range to avoid int32 overflow
            seed = np.random.randint(0, 2**31 - 1)

        logger.info(
            f"Generating with Mochi: {num_frames} frames at {width}x{height}, "
            f"{fps} fps, seed={seed}"
        )

        # Adjust parameters for model constraints
        # Most models have specific size requirements
        if self.low_vram:
            # ZeroScope works best at 576x320
            width = min(width, 576)
            height = min(height, 320)
            num_frames = min(num_frames, 24)
        else:
            # ModelScope works at 256x256 by default
            # Adjust to nearest valid size
            width = 256 if width < 512 else 512
            height = 256 if height < 512 else 512
            num_frames = min(num_frames, 16)  # ModelScope limitation

        logger.info(f"Adjusted to model constraints: {width}x{height}, {num_frames} frames")

        try:
            # Handle reference image
            if ref_image_path:
                logger.warning(
                    "Image conditioning not fully supported in basic Mochi backend. "
                    "Using prompt only. For I2V, consider implementing image2pipe or AnimateDiff."
                )
                # Could implement basic I2V by prepending first frame
                # For now, just use the prompt

            # Generate video
            with torch.no_grad():
                video_frames = self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=25 if self.low_vram else 50,
                    guidance_scale=9.0,
                    generator=generator
                ).frames

            # video_frames shape: (batch, frames, channels, height, width)
            # Convert to (frames, height, width, channels) for imageio
            if isinstance(video_frames, torch.Tensor):
                frames = video_frames.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            else:
                # Already numpy array
                frames = video_frames[0]

            # Ensure uint8 format
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)

            # Generate output path
            from app.utils import generate_output_filename
            from app.settings import settings

            output_filename = generate_output_filename(prompt, "mp4")
            output_path = settings.OUTPUTS_DIR / output_filename

            # Save video with imageio
            imageio.mimwrite(
                str(output_path),
                frames,
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )

            elapsed = time.time() - start_time
            logger.info(f"Video saved to {output_path} in {elapsed:.2f}s")

            return {
                "mp4_path": str(output_path),
                "frames": len(frames),
                "fps": fps,
                "width": width,
                "height": height,
                "seed": seed,
                "backend": "mochi",
                "elapsed_time": round(elapsed, 2)
            }

        except Exception as e:
            logger.error(f"Mochi generation failed: {e}", exc_info=True)
            raise

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
