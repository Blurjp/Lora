@echo off
REM ============================================================================
REM Conda Environment Setup Script
REM ============================================================================

echo.
echo ============================================================================
echo     CogVideoX Service - Conda Setup
echo ============================================================================
echo.

SET ENV_NAME=video_gen
SET PYTHON_VERSION=3.11

echo This script will create a conda environment named "%ENV_NAME%"
echo with Python %PYTHON_VERSION% and install all dependencies.
echo.
pause

echo.
echo [INFO] Creating conda environment: %ENV_NAME%
echo.
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y

IF ERRORLEVEL 1 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo [OK] Environment created successfully
echo.

echo [INFO] Activating environment...
call conda activate %ENV_NAME%

IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate environment
    echo Try running: conda activate %ENV_NAME%
    pause
    exit /b 1
)

echo [OK] Environment activated
echo.

REM Upgrade pip in conda environment
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo ============================================================================
echo Installing PyTorch 2.3.1 with CUDA 12.1
echo ============================================================================
echo.
echo This will take 5-10 minutes...
echo.

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

IF ERRORLEVEL 1 (
    echo [WARNING] PyTorch CUDA installation failed, trying CPU version...
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
)

echo [OK] PyTorch installed
echo.

echo ============================================================================
echo Installing ML Dependencies
echo ============================================================================
echo.

REM Install in correct order to avoid conflicts
pip install "huggingface-hub>=0.23.2"
pip install "diffusers>=0.30.3"
pip install transformers accelerate safetensors sentencepiece xformers

echo [OK] ML dependencies installed
echo.

echo ============================================================================
echo Installing FastAPI and Utilities
echo ============================================================================
echo.

pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings
pip install imageio imageio-ffmpeg opencv-python Pillow
pip install python-dotenv numpy einops psutil

echo [OK] All dependencies installed
echo.

echo ============================================================================
echo Verifying Installation
echo ============================================================================
echo.

python -c "import torch; print('[OK] PyTorch:', torch.__version__)"
python -c "import diffusers; print('[OK] Diffusers:', diffusers.__version__)"
python -c "import transformers; print('[OK] Transformers:', transformers.__version__)"
python -c "import fastapi; print('[OK] FastAPI installed')"
python -c "import torch; print('[OK] CUDA Available:', torch.cuda.is_available())"
python -c "import torch; cuda=torch.cuda.is_available(); print('[OK] GPU:', torch.cuda.get_device_name(0) if cuda else 'CPU only')"

echo.
echo ============================================================================
echo Setup Complete!
echo ============================================================================
echo.
echo Environment "%ENV_NAME%" is ready to use.
echo.
echo To activate it in the future:
echo     conda activate %ENV_NAME%
echo.
echo To start the service:
echo     python quick_run.py
echo.
echo Download models (optional but recommended):
echo     pip install huggingface-hub
echo     huggingface-cli download THUDM/CogVideoX-5b --local-dir ./weights/cogvideox/CogVideoX-5b
echo.
pause

REM Offer to start service
echo.
SET /P START_SERVICE="Start the service now? [Y/n]: "

IF /I "%START_SERVICE%"=="Y" (
    echo.
    echo [INFO] Starting service on port 8765...
    echo.
    python quick_run.py
) ELSE IF "%START_SERVICE%"=="" (
    echo.
    echo [INFO] Starting service on port 8765...
    echo.
    python quick_run.py
) ELSE (
    echo.
    echo [INFO] Setup complete. Start service manually with:
    echo        conda activate %ENV_NAME%
    echo        python quick_run.py
    echo.
    pause
)
