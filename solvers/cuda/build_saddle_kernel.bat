@echo off
setlocal

:: Define the path to the MSVC environment script [Change as needed]
set VCVARS_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
:: Define the output path for the shared library
set OUTPUT_PATH=saddle_solver.dll

:: 1. Initialize the x64 Visual Studio build tools environment
call %VCVARS_PATH%

:: 2. Compile the CUDA kernel into a shared library
nvcc -arch=sm_86 --shared -o %OUTPUT_PATH% saddle_kernel.cu

:: Check if the compilation succeeded
if %ERRORLEVEL% EQU 0 (
    echo Build successful: %OUTPUT_PATH% created.
) else (
    echo Build failed with error code %ERRORLEVEL%.
)

endlocal
