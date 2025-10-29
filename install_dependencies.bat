@echo off
REM ============================================================================
REM Dependency Installation Script - Handles conflicts automatically
REM ============================================================================

echo.
echo ============================================================================
echo Installing dependencies in correct order...
echo ============================================================================
echo.

REM Step 1: Upgrade pip
echo [1/6] Upgrading pip...
python -m pip install --upgrade pip

REM Step 2: Install PyTorch first (largest dependency)
echo.
echo [2/6] Installing PyTorch 2.3.1 with CUDA 12.1...
echo This may take 5-10 minutes...
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

REM Step 3: Install core ML libraries
echo.
echo [3/6] Installing core ML libraries...
pip install transformers accelerate safetensors sentencepiece

REM Step 4: Install huggingface-hub (specific version needed for diffusers)
echo.
echo [4/6] Installing huggingface-hub...
pip install "huggingface-hub>=0.23.2"

REM Step 5: Install diffusers (requires newer huggingface-hub)
echo.
echo [5/6] Installing diffusers...
pip install "diffusers>=0.30.3"

REM Step 6: Install remaining dependencies
echo.
echo [6/6] Installing remaining dependencies...
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings
pip install imageio imageio-ffmpeg opencv-python Pillow
pip install python-dotenv numpy einops psutil xformers

echo.
echo ============================================================================
echo Installation complete!
echo ============================================================================
echo.

REM Verify installation
echo Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

echo.
echo [OK] All dependencies installed successfully!
echo.
pause
