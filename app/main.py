"""
Main FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.utils import setup_logging
from app.routes import router


# Setup logging
logger = setup_logging(settings.LOGS_DIR, settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Local Video Generation Service Starting...")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"Preferred backend: {settings.PREFERRED_BACKEND}")
    logger.info(f"Low-VRAM mode: {settings.LOW_VRAM}")
    logger.info(f"Weights directory: {settings.WEIGHTS_DIR}")
    logger.info(f"Outputs directory: {settings.OUTPUTS_DIR}")
    logger.info("=" * 60)

    # Log VRAM info
    from app.utils import get_vram_info
    vram_info = get_vram_info()

    if vram_info['available']:
        logger.info(
            f"GPU: {vram_info['device_name']} | "
            f"VRAM: {vram_info['free_vram_gb']:.2f}GB free / "
            f"{vram_info['total_vram_gb']:.2f}GB total"
        )
    else:
        logger.warning("CUDA not available - running on CPU (very slow)")

    logger.info("Service ready. Backend will initialize on first request.")

    yield

    # Shutdown
    logger.info("Shutting down service...")
    logger.info("=" * 60)


# Create FastAPI app
app = FastAPI(
    title="Local Video Generation Service",
    description=(
        "Local-only image+text → video inference service. "
        "Supports Open-Sora and Mochi backends with automatic fallback. "
        "\n\n**Legal Notice:** This service has minimal content filtering. "
        "Users are fully responsible for lawful use. "
        "Prohibited: CSAM, sexual violence, non-consensual deepfakes, "
        "extremist content, instructions for illegal acts."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (for local use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local only, so allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


# Custom exception handlers
from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # Set to True for development
        log_level=settings.LOG_LEVEL.lower()
    )
