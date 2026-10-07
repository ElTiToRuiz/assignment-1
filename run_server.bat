@echo off
rem ---------------------------------------------------------------------------
rem  Train everything on a Windows machine WITHOUT git and WITHOUT admin rights.
rem
rem  1. Download the code:  https://github.com/ElTiToRuiz/assignment-1/archive/refs/heads/main.zip
rem     and extract it.
rem  2. If "uv" is not installed, download
rem     https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
rem     and extract uv.exe into a folder called "uv" next to this file (uv\uv.exe).
rem  3. Double-click this file (or run "run_server.bat" in cmd). Keep the window open.
rem     Use "run_server.bat quick" for a 1-minute test first.
rem  4. Bring back results_upload.zip (USB, e-mail, Drive...).
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

where uv >nul 2>nul
if errorlevel 1 (
    if exist "%~dp0uv\uv.exe" (
        set "PATH=%~dp0uv;%PATH%"
    ) else (
        echo uv not found. Download uv-x86_64-pc-windows-msvc.zip and put uv.exe in "%~dp0uv\"
        exit /b 1
    )
)

echo === Installing Python and the dependencies (user folder, no admin) ===
uv sync || exit /b 1
uv run pytest -q || exit /b 1

if /i "%1"=="quick" (
    echo === QUICK test of the Optuna pipeline ===
    uv run python -m experiments.tune --quick --workers 2 --storage smoke\studies.db --envs cliff_walking --algos SARSA || exit /b 1
) else (
    echo === Optuna: 12 studies, resumable. If it stops, run this file again ===
    uv run python -m experiments.tune --workers %NUMBER_OF_PROCESSORS% || exit /b 1
)

echo === Held-out evaluation and figures ===
set RL_JOBS=%NUMBER_OF_PROCESSORS%
uv run python -m experiments.run_all || exit /b 1
uv run python docs\make_slides.py || exit /b 1

echo === Packing the results ===
powershell -NoProfile -Command "Compress-Archive -Force -Path results,docs -DestinationPath results_upload.zip" || exit /b 1
echo.
echo DONE. Bring back results_upload.zip
endlocal
