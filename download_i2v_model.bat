@echo off
REM ============================================================================
REM Download CogVideoX Image-to-Video Model
REM ============================================================================

echo.
echo ============================================================================
echo     Downloading CogVideoX-5B-I2V Model
echo ============================================================================
echo.
echo This will download ~20GB
echo Estimated time: 10-20 minutes
echo.
pause

echo.
echo [INFO] Installing huggingface-hub...
pip install huggingface-hub --quiet

echo.
echo [INFO] Creating weights directory...
mkdir weights\cogvideox 2>nul

echo.
echo ============================================================================
echo [INFO] Downloading CogVideoX-5B-I2V (~20GB)
echo ============================================================================
echo.
echo This may take 10-20 minutes depending on your internet speed...
echo.

python -c "from huggingface_hub import snapshot_download; snapshot_download('THUDM/CogVideoX-5b-I2V', local_dir='./weights/cogvideox/CogVideoX-5b-I2V')"

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Download failed!
    echo.
    echo Try manual download:
    echo   huggingface-cli download THUDM/CogVideoX-5b-I2V --local-dir ./weights/cogvideox/CogVideoX-5b-I2V
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo [OK] CogVideoX-5B-I2V downloaded successfully!
echo ============================================================================
echo.
echo Image-to-Video is now ready.
echo.
echo Next steps:
echo   1. Restart the service: python quick_run.py
echo   2. Upload an image in the web UI
echo   3. Enter a prompt describing the motion
echo   4. Generate!
echo.
pause
