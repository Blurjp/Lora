"""
Backend selector with automatic fallback logic.

Tries to initialize preferred backend, falls back to alternatives if unavailable.
"""
import logging
from typing import Optional, Literal
from pathlib import Path

from models.opensora_backend import OpenSoraBackend, BackendNotAvailable as OpenSoraNA
from models.mochi_backend import MochiBackend, BackendNotAvailable as MochiNA
from models.cogvideox_backend import CogVideoXBackend, BackendNotAvailable as CogVideoXNA

logger = logging.getLogger("video_service")


class BackendSelector:
    """
    Manages backend selection with automatic fallback.
    """

    def __init__(
        self,
        weights_dir: Path,
        device: str = "cuda",
        prefer: Literal["cogvideox", "opensora", "mochi"] = "cogvideox",
        low_vram: bool = False
    ):
        """
        Initialize backend selector.

        Args:
            weights_dir: Path to weights directory
            device: torch device
            prefer: Preferred backend name
            low_vram: Enable low-VRAM mode
        """
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.prefer = prefer
        self.low_vram = low_vram
        self.backend = None
        self.backend_name = None

        self._initialize_backend()

    def _initialize_backend(self):
        """
        Try to initialize backends in order of preference with fallback.
        """
        backends_to_try = []

        # Determine backend order based on preference and low_vram mode
        if self.prefer == "cogvideox":
            backends_to_try = [
                ("cogvideox", CogVideoXBackend),
                ("mochi", MochiBackend),
                ("opensora", OpenSoraBackend)
            ]
        elif self.prefer == "opensora":
            backends_to_try = [
                ("opensora", OpenSoraBackend),
                ("cogvideox", CogVideoXBackend),
                ("mochi", MochiBackend)
            ]
        elif self.prefer == "mochi":
            backends_to_try = [
                ("mochi", MochiBackend),
                ("cogvideox", CogVideoXBackend),
                ("opensora", OpenSoraBackend)
            ]
        else:
            # Default: CogVideoX first (best quality)
            backends_to_try = [
                ("cogvideox", CogVideoXBackend),
                ("mochi", MochiBackend),
                ("opensora", OpenSoraBackend)
            ]

        logger.info(f"Backend preference: {self.prefer}, low_vram: {self.low_vram}")

        errors = []

        for backend_name, backend_class in backends_to_try:
            try:
                logger.info(f"Attempting to initialize {backend_name} backend...")

                self.backend = backend_class(
                    weights_dir=self.weights_dir,
                    device=self.device,
                    low_vram=self.low_vram
                )

                self.backend_name = backend_name
                logger.info(f"Successfully initialized {backend_name} backend")
                return

            except (OpenSoraNA, MochiNA, CogVideoXNA) as e:
                error_msg = f"{backend_name}: {str(e)}"
                logger.warning(f"Failed to initialize {backend_name}: {e}")
                errors.append(error_msg)
                continue

            except Exception as e:
                error_msg = f"{backend_name}: Unexpected error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                continue

        # If we get here, no backend could be initialized
        error_summary = "\n".join(f"  - {err}" for err in errors)
        raise RuntimeError(
            f"Failed to initialize any video generation backend:\n{error_summary}\n\n"
            f"Troubleshooting:\n"
            f"  1. Ensure CUDA is available: nvidia-smi\n"
            f"  2. Install required libraries: pip install -r requirements.txt\n"
            f"  3. Download model weights (see README.md)\n"
            f"  4. Try --low-vram mode or reduce resolution\n"
            f"  5. Check VRAM requirements:\n"
            f"     - CogVideoX-2B (low-VRAM): ~8GB\n"
            f"     - CogVideoX-5B (standard): ~16GB\n"
            f"     - Mochi: ~6-8GB\n"
            f"     - Open-Sora: ~10-15GB"
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
    ) -> dict:
        """
        Generate video using the active backend.

        Args:
            prompt: Text description
            ref_image_path: Optional reference image path
            fps: Frames per second
            num_frames: Number of frames
            width: Video width
            height: Video height
            seed: Random seed

        Returns:
            Generation result dictionary
        """
        if self.backend is None:
            raise RuntimeError("No backend available")

        return self.backend.generate_video(
            prompt=prompt,
            ref_image_path=ref_image_path,
            fps=fps,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed
        )

    def get_backend_name(self) -> str:
        """Get the name of the currently active backend."""
        return self.backend_name or "none"


# Singleton instance (lazy initialized)
_backend_instance: Optional[BackendSelector] = None


def get_backend(
    weights_dir: Path,
    device: str = "cuda",
    prefer: Literal["cogvideox", "opensora", "mochi"] = "cogvideox",
    low_vram: bool = False,
    force_reinit: bool = False
) -> BackendSelector:
    """
    Get or create backend selector instance.

    Args:
        weights_dir: Path to weights directory
        device: torch device
        prefer: Preferred backend
        low_vram: Low-VRAM mode
        force_reinit: Force reinitialization

    Returns:
        BackendSelector instance
    """
    global _backend_instance

    if _backend_instance is None or force_reinit:
        _backend_instance = BackendSelector(
            weights_dir=weights_dir,
            device=device,
            prefer=prefer,
            low_vram=low_vram
        )

    return _backend_instance


def reset_backend():
    """Reset the backend instance (useful for testing or config changes)."""
    global _backend_instance
    if _backend_instance is not None:
        del _backend_instance
    _backend_instance = None
