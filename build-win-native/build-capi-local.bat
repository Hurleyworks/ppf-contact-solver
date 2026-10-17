@echo off
REM File: build-win-native/build-capi-local.bat
REM Code: Claude Code
REM License: Apache v2.0
REM
REM Local build of the LightWave embedding (crates/ppf-cts-capi) against the
REM installed toolchains: system CUDA (CUDA_PATH, 12.9 tested), VS 2022, the
REM system Python 3.11 and Eigen 3.4.0 from a sibling fork. It does NOT run
REM warmup.bat or download anything.
REM
REM Replaces build-cuda-dll-local.bat from the old fork. The orchestrator TUs
REM (src/cpp) are gone; the CUDA backend is now the kernels-only C ABI library
REM built by scripts\build-cuda.bat (named libsimbackend_cuda on Windows,
REM libppfbe_cuda on Linux). This script reuses that recipe unchanged.
REM
REM Usage:  build-capi-local.bat [nocuda|cudaonly]
REM   nocuda    skip the CUDA library step when it is already built.
REM   cudaonly  build only the CUDA library.
REM
REM Outputs:
REM   crates\ppf-cts-capi\include\ppf_cts.h
REM   target\capi\release\ppf_cts_capi.lib            (Rust staticlib, /MT)
REM   crates\ppf-cts-compute\cuda\build\lib\libsimbackend_cuda.{dll,lib}
REM   target\capi\native-static-libs.txt              (host link line)
setlocal enabledelayedexpansion

set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%
for %%I in ("%BUILD_WIN%\..") do set SRC_DIR=%%~fI

if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not defined PPF_EIGEN_PARENT set "PPF_EIGEN_PARENT=E:\code\open_source\forks\modified_ppf-contact-solver\build-win-native\deps"
set EIGEN_STEM=eigen-3.4.0
set DEPS=%PPF_EIGEN_PARENT%
set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat

REM Variables scripts\build-cuda.bat expects (see its header).
set CPP_DIR=%SRC_DIR%\crates\ppf-cts-compute\cuda
set KERNEL_DIR=%SRC_DIR%\crates\ppf-cts-solver\src\kernels
set KERNELGEN=%SRC_DIR%\crates\ppf-cts-compute\seam\kernelgen.py
set OUT_DIR=%CPP_DIR%\build
set KERNELGEN_DIR=%OUT_DIR%\kernelgen
set LIB_DIR=%OUT_DIR%\lib

for /f "delims=" %%P in ('where python') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE ( echo ERROR: python not on PATH & exit /b 1 )
if not exist "%DEPS%\%EIGEN_STEM%\Eigen\Dense" ( echo ERROR: Eigen not found at %DEPS%\%EIGEN_STEM% & exit /b 1 )
if not exist "%CUDA_PATH%\bin\nvcc.exe" ( echo ERROR: nvcc not found under %CUDA_PATH% & exit /b 1 )

echo Setting up MSVC environment...
call "%VCVARS%" >nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )

if /i "%~1"=="nocuda" goto :cargo
echo [1/2] CUDA backend library via scripts\build-cuda.bat
call "%BUILD_WIN%\scripts\build-cuda.bat"
if errorlevel 1 ( echo ERROR: CUDA backend library failed & exit /b 1 )
if /i "%~1"=="cudaonly" exit /b 0

:cargo
REM The solver's build scripts run `python3` (Command::new("python3")). The
REM WindowsApps alias is a Store stub, so give them a real python3.exe: a copy
REM of the system interpreter plus its DLLs, with PYTHONHOME pointing back.
for %%I in ("%PYTHON_EXE%") do set PY_HOME=%%~dpI
set PY_HOME=%PY_HOME:~0,-1%
set PY3_SHIM=%SRC_DIR%\target\capi\py3shim
if not exist "%PY3_SHIM%" mkdir "%PY3_SHIM%"
copy /y "%PYTHON_EXE%" "%PY3_SHIM%\python3.exe" >nul
copy /y "%PY_HOME%\python3*.dll" "%PY3_SHIM%\" >nul
copy /y "%PY_HOME%\vcruntime140*.dll" "%PY3_SHIM%\" >nul
set "PYTHONHOME=%PY_HOME%"
set "PATH=%PY3_SHIM%;%LIB_DIR%;%PATH%"

echo [2/2] cargo build -p ppf-cts-capi (Release, +crt-static)
set CARGO_TARGET_DIR=%SRC_DIR%\target\capi
set RUSTFLAGS=-C target-feature=+crt-static
pushd "%SRC_DIR%"
cargo rustc --release -p ppf-cts-capi --features ppf-cts-solver/cuda -- --print native-static-libs 2> "%CARGO_TARGET_DIR%\cargo-capi.log"
set CARGO_RC=!ERRORLEVEL!
popd
type "%CARGO_TARGET_DIR%\cargo-capi.log" | findstr /v /c:"Compiling" /c:"warning: unused"
if not "!CARGO_RC!"=="0" ( echo ERROR: cargo build failed, see %CARGO_TARGET_DIR%\cargo-capi.log & exit /b 1 )
findstr /c:"native-static-libs:" "%CARGO_TARGET_DIR%\cargo-capi.log" > "%CARGO_TARGET_DIR%\native-static-libs.txt"

echo.
echo [DONE]
echo   header : %SRC_DIR%\crates\ppf-cts-capi\include\ppf_cts.h
echo   capi   : %CARGO_TARGET_DIR%\release\ppf_cts_capi.lib
echo   cuda   : %LIB_DIR%\libsimbackend_cuda.lib (+ .dll)
echo   host link line:
type "%CARGO_TARGET_DIR%\native-static-libs.txt"
endlocal
exit /b 0
