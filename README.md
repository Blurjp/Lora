# Local Video Generation Service

A local-only AI video generation service using CogVideoX, Open-Sora, and Mochi backends.

## Features

- **CogVideoX-5B**: State-of-the-art text-to-video and image-to-video generation
- **Multiple Backends**: Auto-fallback between CogVideoX, Open-Sora, and Mochi
- **Web UI**: Simple HTML interface for video generation
- **Local-Only**: No internet required after model download
- **High-VRAM Optimized**: Full GPU mode for 24GB VRAM cards

## Requirements

- Python 3.10+
- NVIDIA GPU with 16-24GB VRAM (CUDA 12.1+)
- Windows/Linux/Mac

## Quick Start

```bash
# Run the service
python quick_run.py

# Access the web UI
http://localhost:8765
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download CogVideoX-5B model
huggingface-cli download THUDM/CogVideoX-5b --local-dir ./weights/cogvideox/CogVideoX-5b
```

## Usage

1. Open http://localhost:8765 in your browser
2. Enter a text prompt describing your video
3. (Optional) Upload a reference image for image-to-video
4. Click "Generate Video"
5. Wait 3-15 minutes for generation to complete

## Configuration

Edit .env file:

```env
PORT=8765
DEVICE=cuda
PREFERRED_BACKEND=cogvideox
LOW_VRAM=false
```

## System Requirements

- **Minimum**: 16GB VRAM, Python 3.10
- **Recommended**: 24GB VRAM, Python 3.11
- **Storage**: ~50GB for models

## License

See individual model licenses:
- CogVideoX: Apache 2.0
- Open-Sora: Apache 2.0
- Mochi: MIT

## Credits

- CogVideoX by THUDM (Tsinghua University)
- Diffusers by Hugging Face
- FastAPI framework
