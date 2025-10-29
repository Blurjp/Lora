#!/usr/bin/env python3
"""
Quick start script for Local Video Generation Service.

Usage:
    python start.py                    # Start with default settings
    python start.py --low-vram        # Start with low-VRAM mode
    python start.py --port 8080       # Use custom port
    python start.py --reload          # Enable auto-reload (dev mode)
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Start Local Video Generation Service"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Force low-VRAM mode"
    )
    parser.add_argument(
        "--backend",
        choices=["opensora", "mochi"],
        help="Force specific backend"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set environment variables if flags provided
    if args.low_vram:
        os.environ["LOW_VRAM"] = "true"
        print("🔧 Low-VRAM mode enabled")

    if args.backend:
        os.environ["PREFERRED_BACKEND"] = args.backend
        print(f"🔧 Backend forced to: {args.backend}")

    os.environ["LOG_LEVEL"] = args.log_level

    # Import here after env vars are set
    import uvicorn
    from app.main import app

    print("=" * 60)
    print("🎬 Local Video Generation Service")
    print("=" * 60)
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"📊 Log Level: {args.log_level}")
    print(f"🔄 Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print("=" * 60)
    print("\n⏳ Starting server...\n")

    # Start uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()
