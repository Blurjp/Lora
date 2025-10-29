"""
FastAPI routes for video generation service.
"""
import logging
from pathlib import Path
from typing import Optional
import traceback

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.schemas import GenerateResponse, SystemInfoResponse, HealthResponse
from app.settings import settings
from app.utils import (
    validate_input_prompt,
    get_system_info,
    get_vram_info,
    can_run_backend
)
from models import get_backend

logger = logging.getLogger("video_service")
router = APIRouter()

# Global backend instance
backend_instance = None


def init_backend():
    """Initialize backend on first request."""
    global backend_instance

    if backend_instance is None:
        logger.info("="*70)
        logger.info("INITIALIZING BACKEND (first request)")
        logger.info("This will take 2-3 minutes to load model into GPU memory")
        logger.info("Subsequent requests will reuse the loaded model")
        logger.info("="*70)
        try:
            backend_instance = get_backend(
                weights_dir=settings.WEIGHTS_DIR,
                device=settings.DEVICE,
                prefer=settings.PREFERRED_BACKEND,
                low_vram=settings.LOW_VRAM
            )
            logger.info("="*70)
            logger.info(f"✓ Backend ready: {backend_instance.get_backend_name()}")
            logger.info("Model is now in GPU memory and ready for fast generation")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Video generation backend unavailable: {str(e)}"
            )
    else:
        logger.debug("Reusing already-loaded backend instance")

    return backend_instance


@router.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the HTML UI."""
    html_path = Path(__file__).parent / "html" / "index.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")

    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.post("/generate", response_model=GenerateResponse)
async def generate_video(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    fps: int = Form(default=24),
    num_frames: int = Form(default=64),
    width: int = Form(default=720),
    height: int = Form(default=720),
    seed: Optional[int] = Form(None),
    prefer_backend: Optional[str] = Form(None),
    low_vram: bool = Form(False),
    lawful_use_consent: bool = Form(False)
):
    """
    Generate video from text prompt and optional image.

    Args:
        prompt: Text description
        image: Optional reference image
        fps: Frames per second
        num_frames: Number of frames
        width: Video width
        height: Video height
        seed: Random seed
        prefer_backend: Preferred backend name
        low_vram: Enable low-VRAM mode
        lawful_use_consent: User agreement to lawful use

    Returns:
        GenerateResponse with video URL and metadata
    """
    logger.info(f"Generate request: prompt='{prompt[:50]}...', frames={num_frames}, size={width}x{height}")

    # Validate consent
    if not lawful_use_consent:
        return GenerateResponse(
            success=False,
            error="You must agree to lawful use terms before generating content."
        )

    # Validate prompt
    is_valid, error_msg = validate_input_prompt(prompt)
    if not is_valid:
        logger.warning(f"Invalid prompt: {error_msg}")
        return GenerateResponse(
            success=False,
            error=error_msg
        )

    # Ensure dimensions are multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Initialize backend
    try:
        backend = init_backend()
    except HTTPException as e:
        return GenerateResponse(
            success=False,
            error=str(e.detail)
        )

    # Check VRAM availability (informational only - if backend loaded, we're good)
    backend_name = backend.get_backend_name()
    can_run, vram_msg = can_run_backend(backend_name, width, height, num_frames, low_vram)

    # If backend loaded successfully, trust that VRAM is sufficient
    # (The check happens after model loading, so it may report false negatives)
    if not can_run:
        logger.warning(f"VRAM check shows potential issue: {vram_msg}")
        logger.info("Proceeding anyway since backend loaded successfully")
    else:
        logger.info(f"VRAM check: {vram_msg}")

    # Save uploaded image if provided
    ref_image_path = None
    if image:
        try:
            # Read image content first
            content = await image.read()

            # Check if actually uploaded (not just an empty form field)
            if len(content) < 100:  # Too small or empty - treat as no image
                logger.info("No image uploaded (empty field), using text-to-video mode")
                ref_image_path = None
            else:
                # Generate filename with timestamp if original filename is missing
                from datetime import datetime
                if image.filename:
                    image_filename = f"ref_{image.filename}"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"ref_image_{timestamp}.jpg"

                ref_image_path = settings.OUTPUTS_DIR / image_filename

                # Save image content
                with open(ref_image_path, "wb") as f:
                    f.write(content)

                # Verify the saved file can be opened as an image
                try:
                    from PIL import Image as PILImage
                    test_img = PILImage.open(ref_image_path)
                    test_img.verify()
                    logger.info(f"Saved reference image: {ref_image_path}")
                except Exception as verify_error:
                    logger.warning(f"Invalid image file: {verify_error}, using text-to-video mode")
                    ref_image_path = None  # Fall back to text-only
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return GenerateResponse(
                success=False,
                error=f"Failed to save reference image: {str(e)}"
            )

    # Generate video
    try:
        result = backend.generate_video(
            prompt=prompt,
            ref_image_path=str(ref_image_path) if ref_image_path else None,
            fps=fps,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed
        )

        # Construct video URL
        video_filename = Path(result["mp4_path"]).name
        video_url = f"/outputs/{video_filename}"

        logger.info(f"Generation successful: {video_url}")

        return GenerateResponse(
            success=True,
            video_url=video_url,
            **result
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.debug(traceback.format_exc())

        return GenerateResponse(
            success=False,
            error=f"Generation failed: {str(e)}"
        )


@router.get("/outputs/{filename}")
async def serve_output(filename: str):
    """
    Serve generated video files.

    Args:
        filename: Video filename

    Returns:
        Video file response
    """
    file_path = settings.OUTPUTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Security: ensure file is within outputs directory
    try:
        file_path.resolve().relative_to(settings.OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )


@router.get("/system-info", response_model=SystemInfoResponse)
async def get_system_info_endpoint():
    """
    Get system resource information.

    Returns:
        SystemInfoResponse with VRAM and backend info
    """
    try:
        backend = init_backend()
        backend_name = backend.get_backend_name()
    except Exception:
        backend_name = "unavailable"

    sys_info = get_system_info()

    return SystemInfoResponse(
        cuda_available=sys_info['available'],
        device_name=sys_info['device_name'],
        total_vram_gb=sys_info['total_vram_gb'],
        free_vram_gb=sys_info['free_vram_gb'],
        used_vram_gb=sys_info['used_vram_gb'],
        cpu_percent=sys_info['cpu_percent'],
        ram_available_gb=sys_info['ram_available_gb'],
        ram_total_gb=sys_info['ram_total_gb'],
        backend=backend_name
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    vram_info = get_vram_info()

    try:
        backend = init_backend()
        backend_name = backend.get_backend_name()
        status = "healthy"
        message = f"Service operational with {backend_name} backend"
    except Exception as e:
        backend_name = "none"
        status = "unhealthy"
        message = f"Backend initialization failed: {str(e)}"

    return HealthResponse(
        status=status,
        backend=backend_name,
        cuda_available=vram_info['available'],
        message=message
    )
