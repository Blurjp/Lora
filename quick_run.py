#!/usr/bin/env python3
"""
Quick Run Script - Kills existing process and starts service.

This is a minimal script for users who have already completed setup.
For first-time setup, use setup_and_run.bat (Windows) or setup_and_run.sh (Linux/Mac).

Usage:
    python quick_run.py
    python quick_run.py --port 9000
    python quick_run.py --host 0.0.0.0 --port 9000
"""
import argparse
import os
import sys
import signal
import subprocess
import time
import webbrowser
from pathlib import Path


def find_and_kill_process_on_port(port: int) -> bool:
    """
    Find and kill any process using the specified port.

    Args:
        port: Port number to check

    Returns:
        True if a process was killed, False otherwise
    """
    killed = False

    if sys.platform == "win32":
        # Windows
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=True
            )

            for line in result.stdout.split('\n'):
                if f":{port} " in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        try:
                            print(f"[INFO] Found process {pid} using port {port}")
                            print(f"[INFO] Killing process {pid}...")
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                check=True
                            )
                            print(f"[OK] Process {pid} killed successfully")
                            killed = True
                            time.sleep(1)  # Wait for port to be released
                        except subprocess.CalledProcessError:
                            print(f"[WARNING] Could not kill process {pid}")
        except subprocess.CalledProcessError:
            print("[WARNING] Could not check port status")
    else:
        # Linux/Mac
        try:
            # Try lsof first
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pid = result.stdout.strip()
                if pid:
                    print(f"[INFO] Found process {pid} using port {port}")
                    print(f"[INFO] Killing process {pid}...")
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"[OK] Process {pid} killed successfully")
                        killed = True
                        time.sleep(1)
                    except (ProcessLookupError, PermissionError) as e:
                        print(f"[WARNING] Could not kill process: {e}")
        except FileNotFoundError:
            # lsof not available, try fuser
            try:
                result = subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"[OK] Process on port {port} killed")
                    killed = True
                    time.sleep(1)
            except FileNotFoundError:
                print("[WARNING] Cannot check port status (lsof/fuser not found)")

    if not killed:
        print(f"[OK] Port {port} is available")

    return killed


def check_venv() -> bool:
    """Check if running in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )


def main():
    parser = argparse.ArgumentParser(
        description="Quick run script for Local Video Generation Service"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Local Video Generation Service - Quick Run")
    print("=" * 70)
    print()

    # Check if in virtual environment
    if not check_venv():
        print("[WARNING] Not running in virtual environment!")
        print("[INFO] Activate venv first:")
        if sys.platform == "win32":
            print("       venv\\Scripts\\activate")
        else:
            print("       source venv/bin/activate")
        print()
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("[INFO] Exiting. Please activate virtual environment first.")
            return 1
        print()

    # Check if dependencies are installed
    try:
        import fastapi
        import uvicorn
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("[INFO] Run setup_and_run script first or install manually:")
        print("       pip install -r requirements.txt")
        return 1

    # Kill existing process on port
    print(f"[INFO] Checking port {args.port}...")
    find_and_kill_process_on_port(args.port)
    print()

    # Start service
    print(f"[INFO] Starting service on http://{args.host}:{args.port}")
    print(f"[INFO] Press Ctrl+C to stop")
    print()
    print("=" * 70)
    print()

    # Open browser after delay
    if not args.no_browser:
        def open_browser():
            time.sleep(3)
            url = f"http://{args.host}:{args.port}"
            print(f"[INFO] Opening browser: {url}")
            webbrowser.open(url)

        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

    # Start uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("[INFO] Service stopped by user")
        print("=" * 70)
    except Exception as e:
        print(f"[ERROR] Failed to start service: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
