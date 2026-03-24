@echo off
echo ============================================
echo   BookMind - Setup
echo ============================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo [2/3] Installing Python dependencies...
venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo [3/3] Downloading llama.cpp binaries...
echo.
echo Choose version:
echo   1 - CUDA 12.4 (GPU, recommended if you have NVIDIA GPU)
echo   2 - CPU only (no GPU required)
echo.
set /p CHOICE="Enter 1 or 2: "

if not exist bin mkdir bin
if not exist models mkdir models

if "%CHOICE%"=="1" goto :download_cuda
goto :download_cpu

:download_cuda
echo Downloading CUDA 12.4 build...
curl -L https://github.com/ggml-org/llama.cpp/releases/download/b8429/llama-b8429-bin-win-cuda-12.4-x64.zip -o llama_bin.zip
goto :extract

:download_cpu
echo Downloading CPU-only build...
curl -L https://github.com/ggml-org/llama.cpp/releases/download/b8429/llama-b8429-bin-win-avx2-x64.zip -o llama_bin.zip
goto :extract

:extract
if not exist llama_bin.zip (
    echo WARNING: Download failed.
    echo You can still use cloud providers.
    goto :done
)
echo Extracting...
tar -xf llama_bin.zip -C bin
del /f /q llama_bin.zip
echo Binaries installed to bin\

:done
echo.
echo ============================================
echo   Setup complete!
echo.
echo   Download LFM2-2.6B-GGUF model and place
echo   it in the models\ folder.
echo.
echo   Run 'run.bat' to start the app.
echo ============================================
pause
pause
