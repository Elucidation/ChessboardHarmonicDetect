@echo off
setlocal enabledelayedexpansion

set INPUT_DIR=.\input_images
set TRIALS=50

echo Running benchmark for all files in %INPUT_DIR% ...
echo.

for %%F in (%INPUT_DIR%\*) do (
    echo ============================================================
    echo Input: %%F
    echo ============================================================
    python benchmark.py --input "%%F" -n %TRIALS%
    echo.
)

echo All done.
endlocal
