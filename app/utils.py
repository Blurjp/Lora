"""
Utility functions for VRAM detection, logging, and video handling.
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import torch
import psutil


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging to both console and file.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

    # Create logger - FORCE DEBUG for detailed I2V debugging
    logger = logging.getLogger("video_service")
    logger.setLevel(logging.DEBUG)  # Always DEBUG for detailed logs

    # Console handler - also DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Show debug in console too
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def get_vram_info() -> dict:
    """
    Get CUDA device and VRAM information.

    Returns:
        Dictionary with VRAM info: {
            'available': bool,
            'device_name': str,
            'total_vram_gb': float,
            'free_vram_gb': float,
            'used_vram_gb': float
        }
    """
    info = {
        'available': False,
        'device_name': 'CPU',
        'total_vram_gb': 0.0,
        'free_vram_gb': 0.0,
        'used_vram_gb': 0.0
    }

    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            info['available'] = True
            info['device_name'] = torch.cuda.get_device_name(device)

            # Get memory stats in bytes, convert to GB
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            allocated = torch.cuda.memory_allocated(device) / (1024**3)

            info['total_vram_gb'] = round(total, 2)
            # Free VRAM = total - allocated (not reserved, which can exceed total)
            info['free_vram_gb'] = round(max(0, total - allocated), 2)
            info['used_vram_gb'] = round(allocated, 2)
        except Exception as e:
            logging.warning(f"Error getting VRAM info: {e}")

    return info


def estimate_vram_requirement(
    backend: str,
    width: int,
    height: int,
    num_frames: int,
    low_vram: bool = False
) -> float:
    """
    Estimate VRAM requirement for video generation.

    Args:
        backend: Backend name ('opensora' or 'mochi')
        width: Video width in pixels
        height: Video height in pixels
        num_frames: Number of frames
        low_vram: Whether low-VRAM optimizations are enabled

    Returns:
        Estimated VRAM in GB
    """
    # Rough estimates based on typical usage patterns
    # These are conservative estimates and may vary

    base_model_size = {
        'opensora': 8.0,  # Model weights + overhead
        'mochi': 6.0      # Smaller model
    }

    # Calculate pixel count
    total_pixels = width * height * num_frames

    # Estimate intermediate activation memory (very rough)
    # Typically ~16 bytes per pixel for fp16 with attention overhead
    activation_gb = (total_pixels * 16) / (1024**3)

    # Low-VRAM mode uses gradient checkpointing and offloading
    if low_vram:
        activation_gb *= 0.5

    total_estimate = base_model_size.get(backend, 7.0) + activation_gb

    # Add 20% safety margin
    return round(total_estimate * 1.2, 2)


def can_run_backend(
    backend: str,
    width: int,
    height: int,
    num_frames: int,
    low_vram: bool = False
) -> Tuple[bool, str]:
    """
    Check if the backend can run with current VRAM.

    Args:
        backend: Backend name
        width: Video width
        height: Video height
        num_frames: Number of frames
        low_vram: Low-VRAM mode flag

    Returns:
        Tuple of (can_run: bool, message: str)
    """
    vram_info = get_vram_info()

    if not vram_info['available']:
        return False, "CUDA not available. GPU inference requires CUDA."

    required_vram = estimate_vram_requirement(backend, width, height, num_frames, low_vram)
    available_vram = vram_info['free_vram_gb']

    if available_vram < required_vram:
        msg = (
            f"Insufficient VRAM: {backend} needs ~{required_vram}GB, "
            f"but only {available_vram}GB available. "
            f"Try reducing resolution/frames or enable low-VRAM mode."
        )
        return False, msg

    return True, f"OK: {available_vram}GB available, ~{required_vram}GB needed"


def generate_output_filename(prompt: str, extension: str = "mp4") -> str:
    """
    Generate a timestamped filename from prompt.

    Args:
        prompt: Text prompt
        extension: File extension (default: mp4)

    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize prompt for filename (keep alphanumeric and spaces, limit length)
    safe_prompt = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)[:50]
    safe_prompt = re.sub(r'\s+', '_', safe_prompt.strip())

    if not safe_prompt:
        safe_prompt = "video"

    return f"{timestamp}_{safe_prompt}.{extension}"


def validate_input_prompt(prompt: str) -> Tuple[bool, Optional[str]]:
    """
    Basic validation to catch obviously prohibited content.
    This is NOT a comprehensive filter - user responsibility applies.

    Args:
        prompt: Input text prompt

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    if not prompt or len(prompt.strip()) < 3:
        return False, "Prompt must be at least 3 characters"

    if len(prompt) > 1000:
        return False, "Prompt too long (max 1000 characters)"

    # Content filtering disabled - all prompts allowed
    # User takes full responsibility for lawful use
    prohibited_keywords = []

    return True, None


def get_system_info() -> dict:
    """Get system resource information."""
    info = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    }
    info.update(get_vram_info())
    return info
