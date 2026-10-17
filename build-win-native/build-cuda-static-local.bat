@echo off
REM File: build-win-native/build-cuda-static-local.bat
REM Code: Claude Code
REM License: Apache v2.0
REM
REM EXPERIMENTAL: the CUDA backend as a STATIC library built /MT, so a
REM LightWave plugin (.p) can carry the solver with no DLL beside it.
REM
REM Run after build-capi-local.bat has built the DLL once: it reuses that
REM build's rendered kernels (build\kernelgen) and its compile command list
REM (build\compile-jobs.txt), recompiling every TU with /MT into build-mt\.
REM Then it does the device link itself (nvcc -dlink -dlto, the same native
REM cubins as the DLL) and archives host objects + the device-link object
REM with lib.exe. stub.cpp is left out: its empty print_rust would collide
REM with the Rust one the capi staticlib defines.
REM
REM The archive keeps the DLL's import-lib name, libsimbackend_cuda.lib, so
REM the host link line cargo prints works unchanged: point /LIBPATH at
REM build-mt\lib instead of build\lib, and link cudart_static.lib in place of
REM cudart.lib.
setlocal enabledelayedexpansion

set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%
for %%I in ("%BUILD_WIN%\..") do set SRC_DIR=%%~fI
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
set CPP_DIR=%SRC_DIR%\crates\ppf-cts-compute\cuda
set DLL_OUT=%CPP_DIR%\build
set MT_OUT=%CPP_DIR%\build-mt
set NVCC="%CUDA_PATH%\bin\nvcc.exe"
for /f "delims=" %%P in ('where python') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"

if not exist "%DLL_OUT%\compile-jobs.txt" ( echo ERROR: run build-capi-local.bat first & exit /b 1 )

call "%VCVARS%" >nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )

if not exist "%MT_OUT%\obj" mkdir "%MT_OUT%\obj"
if not exist "%MT_OUT%\cuda-entry" mkdir "%MT_OUT%\cuda-entry"
if not exist "%MT_OUT%\lib" mkdir "%MT_OUT%\lib"

REM /MD -> /MT, build\ -> build-mt\, drop stub.cpp; list the objects.
"%PYTHON_EXE%" "%BUILD_WIN%\scripts\derive_mt_jobs.py" "%DLL_OUT%" "%MT_OUT%"
if errorlevel 1 ( echo ERROR: could not derive the /MT job list & exit /b 1 )

echo Compiling the CUDA backend /MT...
"%PYTHON_EXE%" "%BUILD_WIN%\scripts\run_jobs.py" "%MT_OUT%\compile-jobs.txt" --jobs %NUMBER_OF_PROCESSORS%
if errorlevel 1 ( echo ERROR: a /MT compile failed & exit /b 1 )

REM Same native cubins as the DLL, from the manifest.
set ARCH_FLOOR=
set GENCODE=
for /f "tokens=1,2" %%A in (%CPP_DIR%\cuda_arch.txt) do if "%%A"=="floor" set ARCH_FLOOR=%%B
for /f "tokens=1,2" %%A in (%CPP_DIR%\cuda_arch.txt) do if "%%A"=="cubin" set "GENCODE=!GENCODE! -gencode arch=compute_!ARCH_FLOOR!,code=sm_%%B"

echo Device-linking (nvcc -dlink -dlto)...
%NVCC% -dlink -dlto -Wno-deprecated-gpu-targets !GENCODE! -Xcompiler "/MT" -optf "%MT_OUT%\objects.rsp" -o "%MT_OUT%\obj\device_link.obj"
if errorlevel 1 ( echo ERROR: device link failed & exit /b 1 )

echo Archiving libsimbackend_cuda.lib (static)...
del /q "%MT_OUT%\lib\libsimbackend_cuda.lib" 2>nul
lib /nologo /OUT:"%MT_OUT%\lib\libsimbackend_cuda.lib" @"%MT_OUT%\objects.rsp" "%MT_OUT%\obj\device_link.obj"
if errorlevel 1 ( echo ERROR: lib.exe failed & exit /b 1 )

echo [DONE] %MT_OUT%\lib\libsimbackend_cuda.lib  (link with cudart_static.lib)
endlocal
exit /b 0
