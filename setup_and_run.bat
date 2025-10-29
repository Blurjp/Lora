@echo off
REM ============================================================================
REM Local Video Generation Service - Automated Setup and Run Script (Windows)
REM ============================================================================
REM This script handles everything:
REM - Kills existing processes on port
REM - Creates virtual environment
REM - Installs all dependencies
REM - Downloads CogVideoX model (optional)
REM - Starts the service
REM ============================================================================

SETLOCAL EnableDelayedExpansion

REM Configuration
SET SERVICE_PORT=8765
SET SERVICE_HOST=127.0.0.1
SET VENV_DIR=venv
SET PYTHON_MIN_VERSION=3.10

echo.
echo ============================================================================
echo     Local Video Generation Service - Automated Setup
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
FOR /F "tokens=2" %%i IN ('python --version 2^>^&1') DO SET PYTHON_VERSION=%%i
echo [INFO] Found Python %PYTHON_VERSION%

REM Check Python version (basic check)
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
IF ERRORLEVEL 1 (
    echo [ERROR] Python 3.10+ required, found %PYTHON_VERSION%
    pause
    exit /b 1
)

echo [OK] Python version compatible
echo.

REM Step 1: Kill any process using the port
echo ============================================================================
echo Step 1: Checking port %SERVICE_PORT%
echo ============================================================================
echo.

FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr ":%SERVICE_PORT% " ^| findstr "LISTENING"') DO (
    echo [INFO] Found process %%P using port %SERVICE_PORT%
    echo [INFO] Killing process %%P...
    taskkill /F /PID %%P >nul 2>&1
    IF ERRORLEVEL 1 (
        echo [WARNING] Could not kill process %%P - you may need to run as Administrator
    ) ELSE (
        echo [OK] Process killed successfully
    )
)

echo [OK] Port %SERVICE_PORT% is now available
echo.
timeout /t 2 >nul

REM Step 2: Create virtual environment
echo ============================================================================
echo Step 2: Setting up Python virtual environment
echo ============================================================================
echo.

IF EXIST "%VENV_DIR%\" (
    echo [INFO] Virtual environment already exists at %VENV_DIR%
    echo [INFO] Using existing environment...
) ELSE (
    echo [INFO] Creating virtual environment...
    python -m venv %VENV_DIR%
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Step 3: Check if dependencies are installed
echo ============================================================================
echo Step 3: Installing dependencies
echo ============================================================================
echo.

REM Check if PyTorch is installed
python -c "import torch" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [INFO] PyTorch not found, installing PyTorch with CUDA 12.1...
    echo [INFO] This may take 5-10 minutes...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    IF ERRORLEVEL 1 (
        echo [WARNING] PyTorch installation failed, trying without CUDA...
        pip install torch torchvision torchaudio --quiet
    )
    echo [OK] PyTorch installed
) ELSE (
    echo [OK] PyTorch already installed
)

REM Check if diffusers is installed with correct version
python -c "import diffusers; from packaging import version; exit(0 if version.parse(diffusers.__version__) >= version.parse('0.30.0') else 1)" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [INFO] Installing/upgrading dependencies from requirements.txt...
    echo [INFO] This may take 5-10 minutes...
    pip install -r requirements.txt --quiet --upgrade
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) ELSE (
    echo [OK] Dependencies already installed
)

echo.

REM Check CUDA availability
echo [INFO] Checking CUDA availability...
python -c "import torch; print('[OK] CUDA Available: ' + str(torch.cuda.is_available())); print('[INFO] Device: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
echo.

REM Step 4: Setup configuration
echo ============================================================================
echo Step 4: Configuration
echo ============================================================================
echo.

IF NOT EXIST ".env" (
    echo [INFO] Creating .env configuration file...
    copy .env.example .env >nul
    REM Update port in .env
    powershell -Command "(Get-Content .env) -replace 'PORT=8000', 'PORT=%SERVICE_PORT%' | Set-Content .env"
    echo [OK] Configuration file created
) ELSE (
    echo [OK] Configuration file already exists
    REM Update port anyway
    powershell -Command "(Get-Content .env) -replace 'PORT=\d+', 'PORT=%SERVICE_PORT%' | Set-Content .env"
    echo [INFO] Updated port to %SERVICE_PORT% in .env
)

echo.

REM Step 5: Download models (optional)
echo ============================================================================
echo Step 5: Download CogVideoX Model (Optional)
echo ============================================================================
echo.
echo CogVideoX-5B is ~20GB and takes 10-20 minutes to download.
echo The model will auto-download on first use if you skip this step.
echo.
echo Download now? (Recommended for better experience)
echo [1] Yes - Download CogVideoX-5B (16GB VRAM, best quality)
echo [2] Yes - Download CogVideoX-2B (8GB VRAM, low-VRAM mode)
echo [3] No - Skip download (will auto-download on first use)
echo.

SET /P DOWNLOAD_CHOICE="Enter choice [1/2/3]: "

IF "%DOWNLOAD_CHOICE%"=="1" (
    echo.
    echo [INFO] Downloading CogVideoX-5B (~20GB)...
    echo [INFO] This will take 10-20 minutes depending on your internet speed...
    pip install huggingface-hub --quiet
    mkdir weights\cogvideox 2>nul
    python -c "from huggingface_hub import snapshot_download; snapshot_download('THUDM/CogVideoX-5b', local_dir='./weights/cogvideox/CogVideoX-5b')"
    IF ERRORLEVEL 1 (
        echo [WARNING] Download failed, model will auto-download on first use
    ) ELSE (
        echo [OK] CogVideoX-5B downloaded successfully
    )
) ELSE IF "%DOWNLOAD_CHOICE%"=="2" (
    echo.
    echo [INFO] Downloading CogVideoX-2B (~10GB)...
    echo [INFO] This will take 5-10 minutes depending on your internet speed...
    pip install huggingface-hub --quiet
    mkdir weights\cogvideox 2>nul
    python -c "from huggingface_hub import snapshot_download; snapshot_download('THUDM/CogVideoX-2b', local_dir='./weights/cogvideox/CogVideoX-2b')"
    IF ERRORLEVEL 1 (
        echo [WARNING] Download failed, model will auto-download on first use
    ) ELSE (
        echo [OK] CogVideoX-2B downloaded successfully
    )
    REM Set LOW_VRAM mode
    powershell -Command "(Get-Content .env) -replace 'LOW_VRAM=false', 'LOW_VRAM=true' | Set-Content .env"
    echo [INFO] Enabled LOW_VRAM mode in .env
) ELSE (
    echo [INFO] Skipping download - model will auto-download on first use
)

echo.

REM Step 6: Start the service
echo ============================================================================
echo Step 6: Starting Video Generation Service
echo ============================================================================
echo.

echo [INFO] Service will start on http://%SERVICE_HOST%:%SERVICE_PORT%
echo [INFO] Press Ctrl+C to stop the service
echo.
timeout /t 3 >nul

echo ============================================================================
echo     Service Starting...
echo ============================================================================
echo.

REM Start browser after a delay (in background)
start "" /B powershell -Command "Start-Sleep -Seconds 5; Start-Process 'http://%SERVICE_HOST%:%SERVICE_PORT%'"

REM Start the service
python -m uvicorn app.main:app --host %SERVICE_HOST% --port %SERVICE_PORT% --log-level info

REM If service exits
echo.
echo ============================================================================
echo Service stopped
echo ============================================================================
echo.
pause
