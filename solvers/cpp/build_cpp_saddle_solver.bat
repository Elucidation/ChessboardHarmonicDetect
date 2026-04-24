@echo off
setlocal

:: Define the path to the MSVC environment script [Change as needed]
set VCVARS_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
:: Define the output path for the shared library
set OUTPUT_PATH=cpp_saddle_solver.dll

:: 1. Initialize the x64 Visual Studio build tools environment
call %VCVARS_PATH%

:: 2. Compile the C++ source into a shared library
cl.exe /O2 /openmp /LD /Fe%OUTPUT_PATH% saddle_solver.cpp

:: Check if the compilation succeeded
if %ERRORLEVEL% EQU 0 (
    echo Build successful: %OUTPUT_PATH% created.
    :: Cleanup object file
    del saddle_solver.obj
    del cpp_saddle_solver.exp
    del cpp_saddle_solver.lib
) else (
    echo Build failed with error code %ERRORLEVEL%.
)

endlocal
