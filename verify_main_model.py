#!/usr/bin/env python3
"""
Verify main CogVideoX-5B model installation
"""
import os
from pathlib import Path
import sys

print("=" * 70)
print("CogVideoX-5B (T2V) Model Verification")
print("=" * 70)
print()

# Check if model is in weights dir or cache
weights_dir = Path("./weights/cogvideox/CogVideoX-5b")
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

model_path = None

if weights_dir.exists():
    print(f"✓ Found model in: {weights_dir}")
    model_path = weights_dir
else:
    print(f"⚠ Model not in weights dir: {weights_dir}")
    print(f"  Checking HuggingFace cache...")

    # Look for model in cache
    for item in cache_dir.glob("models--THUDM--CogVideoX-5b"):
        print(f"✓ Found in cache: {item}")
        model_path = item / "snapshots"
        # Get latest snapshot
        snapshots = list(model_path.glob("*"))
        if snapshots:
            model_path = snapshots[0]
            print(f"  Using snapshot: {model_path}")
        break

if not model_path:
    print()
    print("❌ CogVideoX-5B model not found!")
    print()
    print("Download it:")
    print("  huggingface-cli download THUDM/CogVideoX-5b --local-dir ./weights/cogvideox/CogVideoX-5b")
    print()
    sys.exit(1)

# Check required files
print()
print("Checking model files...")

required_files = [
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
]

for file_path in required_files:
    full_path = model_path / file_path
    if full_path.exists():
        print(f"✓ {file_path}")
    else:
        print(f"❌ Missing: {file_path}")

# Try to load
print()
print("=" * 70)
print("Testing model loading...")
print("=" * 70)
print()

try:
    from diffusers import CogVideoXPipeline
    import torch

    print("Loading CogVideoX-5B pipeline...")
    print("This may take 2-3 minutes...")
    print()

    pipeline = CogVideoXPipeline.from_pretrained(
        str(model_path) if weights_dir.exists() else "THUDM/CogVideoX-5b",
        torch_dtype=torch.float16
    )

    print("✓ Pipeline loaded successfully!")
    print()
    print("=" * 70)
    print("✅ Main model is working correctly!")
    print("=" * 70)
    print()

except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    print()
    print("=" * 70)
    print("FIX: Re-download the model")
    print("=" * 70)
    print()
    if weights_dir.exists():
        print("# Delete corrupted model")
        print(f'rmdir /s /q "{weights_dir.absolute()}"')
        print()
    print("# Download fresh copy")
    print("huggingface-cli download THUDM/CogVideoX-5b --local-dir ./weights/cogvideox/CogVideoX-5b")
    print()
    sys.exit(1)
