#!/usr/bin/env python3
"""
Quick setup test script to verify installation.

Checks:
- Python version
- Required packages
- CUDA availability
- Directory structure
- Configuration
"""
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("🐍 Python Version Check")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("   ❌ FAIL: Python 3.10+ required")
        return False

    print("   ✅ PASS")
    return True


def check_packages():
    """Check required packages."""
    print("\n📦 Package Check")

    required = [
        "fastapi",
        "uvicorn",
        "torch",
        "diffusers",
        "transformers",
        "imageio",
        "PIL",
        "pydantic",
    ]

    all_ok = True
    for pkg in required:
        try:
            if pkg == "PIL":
                __import__("PIL")
            else:
                __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} - NOT INSTALLED")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability."""
    print("\n🎮 CUDA Check")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"   ✅ CUDA Available")
            print(f"   GPU: {device_name}")
            print(f"   CUDA Version: {cuda_version}")

            # Get VRAM info
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   Total VRAM: {total_vram:.2f} GB")

            return True
        else:
            print("   ⚠️  CUDA Not Available (will use CPU - very slow)")
            return False

    except Exception as e:
        print(f"   ❌ Error checking CUDA: {e}")
        return False


def check_directories():
    """Check directory structure."""
    print("\n📁 Directory Check")

    required_dirs = [
        "app",
        "models",
        "weights",
        "outputs",
        "logs",
    ]

    all_ok = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - NOT FOUND")
            all_ok = False

    return all_ok


def check_config():
    """Check configuration."""
    print("\n⚙️  Configuration Check")

    env_example = Path(".env.example")
    env_file = Path(".env")

    if env_example.exists():
        print("   ✅ .env.example")
    else:
        print("   ❌ .env.example - NOT FOUND")

    if env_file.exists():
        print("   ✅ .env")
    else:
        print("   ⚠️  .env - Not found (will use defaults)")

    return True


def check_weights():
    """Check for model weights."""
    print("\n🎯 Model Weights Check")

    weights_dir = Path("weights")

    # Check for Mochi weights
    mochi_dirs = [
        weights_dir / "mochi" / "text-to-video-ms-1.7b",
        weights_dir / "mochi" / "zeroscope_v2_576w",
    ]

    has_mochi = False
    for mochi_dir in mochi_dirs:
        if mochi_dir.exists():
            print(f"   ✅ Found: {mochi_dir.name}")
            has_mochi = True

    if not has_mochi:
        print("   ⚠️  No Mochi weights found")
        print("   ℹ️  Weights will auto-download on first use")

    # Check for Open-Sora weights
    opensora_dir = weights_dir / "opensora"
    if opensora_dir.exists() and any(opensora_dir.iterdir()):
        print("   ✅ Open-Sora weights directory found")
    else:
        print("   ⚠️  No Open-Sora weights found")

    return True


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)

    if all_passed:
        print("✅ All checks passed! Ready to start.")
        print("\nTo start the service:")
        print("   python start.py")
        print("\nOr:")
        print("   uvicorn app.main:app --reload")
        print("\nThen open: http://localhost:8000")
    else:
        print("❌ Some checks failed. Please fix issues before starting.")
        print("\nInstall missing packages:")
        print("   pip install -r requirements.txt")
        print("\nFor CUDA support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    print("=" * 60)


def main():
    """Run all checks."""
    print("=" * 60)
    print("🔍 Local Video Generation Service - Setup Test")
    print("=" * 60)

    results = {
        "Python Version": check_python_version(),
        "Packages": check_packages(),
        "CUDA": check_cuda(),
        "Directories": check_directories(),
        "Configuration": check_config(),
        "Weights": check_weights(),
    }

    print_summary(results)


if __name__ == "__main__":
    main()
