@echo off
REM ============================================================================
REM Clean Installation Script - No Conflicts!
REM ============================================================================

echo.
echo ============================================================================
echo     Installing Dependencies (Conflict-Free)
echo ============================================================================
echo.

REM Step 1: Upgrade pip
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Step 2: Install PyTorch 2.3.1 (compatible with xformers 0.0.27)
echo [2/4] Installing PyTorch 2.3.1 with CUDA 12.1...
echo This may take 5-10 minutes...
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

IF ERRORLEVEL 1 (
    echo [WARNING] CUDA version failed, trying CPU version...
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
)
echo [OK] PyTorch installed
echo.

REM Step 3: Install compatible dependencies in order
echo [3/4] Installing ML libraries...
pip install "huggingface-hub>=0.23.2"
pip install "diffusers>=0.30.3"
pip install "transformers>=4.40.0"
pip install accelerate safetensors sentencepiece
pip install xformers==0.0.27 --no-deps
pip install einops
echo [OK] ML libraries installed
echo.

REM Step 4: Install FastAPI and utilities
echo [4/4] Installing FastAPI and utilities...
pip install fastapi==0.109.2 uvicorn[standard]==0.27.1
pip install python-multipart==0.0.9 pydantic==2.6.1 pydantic-settings==2.1.0
pip install imageio==2.34.0 imageio-ffmpeg==0.4.9
pip install opencv-python==4.9.0.80 Pillow==10.2.0
pip install python-dotenv==1.0.1 numpy==1.26.4 psutil==5.9.8
echo [OK] Utilities installed
echo.

echo ============================================================================
echo     Verifying Installation
echo ============================================================================
echo.

python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import diffusers; print('✓ Diffusers:', diffusers.__version__)"
python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
python -c "import fastapi; print('✓ FastAPI installed')"
python -c "import xformers; print('✓ xformers:', xformers.__version__)"
python -c "import torch; print('✓ CUDA Available:', torch.cuda.is_available())"
python -c "import torch; cuda=torch.cuda.is_available(); print('✓ GPU:', torch.cuda.get_device_name(0) if cuda else 'CPU only (slow!)')"

echo.
echo ============================================================================
echo     Installation Complete!
echo ============================================================================
echo.
echo Next steps:
echo   1. Run: python quick_run.py
echo   2. Open: http://localhost:8765
echo   3. Start generating videos!
echo.
pause
