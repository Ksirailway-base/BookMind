@echo off
echo Starting Local Book Tutor...
echo.

for %%f in (models\*.gguf) do (
    echo Found model: %%f
    echo Starting llama-server...
    start "" bin\llama-server.exe -m %%f --port 8080 -ngl 99
    timeout /t 5 /nobreak >nul
    goto :start_app
)

echo No .gguf model found in models\ folder.
echo Starting in cloud-only mode (OpenAI / Gemini).
echo To use local mode, download a .gguf model to models\
echo.

:start_app
venv\Scripts\python.exe app.py
pause
