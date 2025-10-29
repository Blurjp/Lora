#!/usr/bin/env python3
"""
Verify and fix I2V model installation
"""
import os
from pathlib import Path
import sys

print("=" * 70)
print("CogVideoX I2V Model Verification")
print("=" * 70)
print()

weights_dir = Path("./weights/cogvideox/CogVideoX-5b-I2V")

if not weights_dir.exists():
    print("❌ I2V model directory not found!")
    print(f"   Expected: {weights_dir.absolute()}")
    print()
    print("Run this to download:")
    print("   .\\download_i2v_model.bat")
    sys.exit(1)

print("✓ Model directory exists")

# Check required files
required_files = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "scheduler/scheduler_config.json",
]

missing_files = []
empty_files = []

for file_path in required_files:
    full_path = weights_dir / file_path
    if not full_path.exists():
        missing_files.append(file_path)
    elif full_path.stat().st_size == 0:
        empty_files.append(file_path)
    else:
        print(f"✓ {file_path}")

if missing_files:
    print()
    print("❌ Missing files:")
    for f in missing_files:
        print(f"   - {f}")

if empty_files:
    print()
    print("❌ Empty/corrupted files:")
    for f in empty_files:
        print(f"   - {f}")

if missing_files or empty_files:
    print()
    print("=" * 70)
    print("FIX: Re-download the model")
    print("=" * 70)
    print()
    print("Run these commands:")
    print()
    print("# Delete corrupted download")
    print(f'rmdir /s /q "{weights_dir.absolute()}"')
    print()
    print("# Re-download")
    print(".\\download_i2v_model.bat")
    sys.exit(1)

# Try to load with diffusers
print()
print("=" * 70)
print("Testing model loading...")
print("=" * 70)
print()

try:
    from diffusers import CogVideoXImageToVideoPipeline
    import torch

    print("Loading I2V pipeline...")

    pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
        str(weights_dir),
        torch_dtype=torch.float16
    )

    print("✓ Pipeline loaded successfully!")
    print()
    print("=" * 70)
    print("✅ I2V model is working correctly!")
    print("=" * 70)
    print()
    print("Restart the service:")
    print("   python quick_run.py")
    print()

except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    print()
    print("=" * 70)
    print("FIX: Re-download the model")
    print("=" * 70)
    print()
    print("# Delete and re-download")
    print(f'rmdir /s /q "{weights_dir.absolute()}"')
    print(".\\download_i2v_model.bat")
    print()
    sys.exit(1)
