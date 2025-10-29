@echo off
REM ============================================================================
REM Fix Corrupted I2V Model - Delete and Re-download
REM ============================================================================

echo.
echo ============================================================================
echo     Fixing Corrupted I2V Model
echo ============================================================================
echo.
echo This will:
echo   1. Delete the corrupted model
echo   2. Re-download clean copy (~20GB)
echo   3. Verify it works
echo.
pause

echo.
echo [1/3] Deleting corrupted model...
rmdir /s /q "weights\cogvideox\CogVideoX-5b-I2V" 2>nul
IF EXIST "weights\cogvideox\CogVideoX-5b-I2V" (
    echo [WARNING] Could not delete, trying with admin rights...
    echo Please manually delete: weights\cogvideox\CogVideoX-5b-I2V
    pause
    exit /b 1
)
echo [OK] Corrupted files deleted
echo.

echo [2/3] Re-downloading I2V model...
echo This will take 10-20 minutes (~20GB)
echo.

pip install huggingface-hub --quiet

python -c "from huggingface_hub import snapshot_download; snapshot_download('THUDM/CogVideoX-5b-I2V', local_dir='./weights/cogvideox/CogVideoX-5b-I2V', resume_download=True)"

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Download failed!
    echo.
    echo Try with huggingface-cli:
    echo   huggingface-cli download THUDM/CogVideoX-5b-I2V --local-dir ./weights/cogvideox/CogVideoX-5b-I2V --resume-download
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Download complete
echo.

echo [3/3] Verifying model...
python verify_and_fix_i2v.py

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Model still corrupted after download!
    echo.
    echo Try manual download with resume:
    echo   huggingface-cli download THUDM/CogVideoX-5b-I2V --local-dir ./weights/cogvideox/CogVideoX-5b-I2V --resume-download
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo [OK] I2V Model Fixed Successfully!
echo ============================================================================
echo.
echo Now restart the service:
echo   python quick_run.py
echo.
pause
