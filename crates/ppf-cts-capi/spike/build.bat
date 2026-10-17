@echo off
REM File: crates/ppf-cts-capi/spike/build.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Builds the console spike (spike.cpp) /MT, the way a LightWave plugin is
REM built, and stages the runtime DLLs next to it. The host links
REM ppf_cts_capi.lib + the native-static-libs cargo printed for it (which
REM already name libsimbackend_cuda and cudart).
REM
REM Prereq (from the repo root):  build-win-native\build-capi-local.bat
REM   -> target\capi\release\ppf_cts_capi.lib
REM   -> target\capi\native-static-libs.txt
REM   -> crates\ppf-cts-compute\cuda\build\lib\libsimbackend_cuda.{dll,lib}
setlocal enabledelayedexpansion

set SPIKE_DIR=%~dp0
set SRC_DIR=%SPIKE_DIR%..\..\..
for %%I in ("%SRC_DIR%") do set SRC_DIR=%%~fI

set INCLUDE_DIR=%SRC_DIR%\crates\ppf-cts-capi\include
set CAPI_LIB_DIR=%SRC_DIR%\target\capi\release
set NATIVE_LIBS_FILE=%SRC_DIR%\target\capi\native-static-libs.txt
set BACKEND_LIB_DIR=%SRC_DIR%\crates\ppf-cts-compute\cuda\build\lib
set OUT_DIR=%SPIKE_DIR%build
REM "build.bat static" links the /MT static CUDA backend (build-mt, from
REM build-win-native\build-cuda-static-local.bat) and cudart_static.lib, so
REM spike.exe needs no DLLs. Output goes to build-static.
set STATIC=0
if /i "%~1"=="static" (
    set STATIC=1
    set BACKEND_LIB_DIR=%SRC_DIR%\crates\ppf-cts-compute\cuda\build-mt\lib
    set OUT_DIR=%SPIKE_DIR%build-static
)

if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set CUDA_LIB_DIR=%CUDA_PATH%\lib\x64
set CUDA_BIN_DIR=%CUDA_PATH%\bin
set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat

if not exist "%CAPI_LIB_DIR%\ppf_cts_capi.lib" ( echo ERROR: %CAPI_LIB_DIR%\ppf_cts_capi.lib not found; run build-win-native\build-capi-local.bat & exit /b 1 )
if not exist "%NATIVE_LIBS_FILE%" ( echo ERROR: %NATIVE_LIBS_FILE% not found; run build-win-native\build-capi-local.bat & exit /b 1 )
if not exist "%BACKEND_LIB_DIR%\libsimbackend_cuda.lib" ( echo ERROR: libsimbackend_cuda.lib not found in %BACKEND_LIB_DIR% & exit /b 1 )

REM The line reads "note: native-static-libs: a.lib b.lib ...": keep the libs.
set NATIVE_LIBS=
for /f "usebackq tokens=1,* delims=:" %%A in ("%NATIVE_LIBS_FILE%") do set "REST=%%B"
for /f "tokens=1,* delims=:" %%A in ("!REST!") do set "NATIVE_LIBS=%%B"
if not defined NATIVE_LIBS ( echo ERROR: could not parse %NATIVE_LIBS_FILE% & exit /b 1 )
if "%STATIC%"=="1" set "NATIVE_LIBS=!NATIVE_LIBS: cudart.lib= cudart_static.lib!"
echo native-static-libs:!NATIVE_LIBS!

REM windows.0.52.0.lib (named by the windows-targets crate) ships in the
REM windows_x86_64_msvc crate's lib folder in the cargo registry.
set WINLIB_DIR=
for /d %%R in ("%USERPROFILE%\.cargo\registry\src\*") do for /d %%W in ("%%R\windows_x86_64_msvc-0.52.*") do if exist "%%W\lib\windows.0.52.0.lib" set "WINLIB_DIR=%%W\lib"
if not defined WINLIB_DIR ( echo ERROR: windows.0.52.0.lib not found in the cargo registry & exit /b 1 )

echo Setting up MSVC environment...
call "%VCVARS%" >nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo Compiling and linking spike.exe (/MT)...
cl /nologo /EHsc /MT /O2 /std:c++17 /I "%INCLUDE_DIR%" ^
    /Fo"%OUT_DIR%\\" /Fe"%OUT_DIR%\spike.exe" ^
    "%SPIKE_DIR%spike.cpp" ^
    /link ^
    /LIBPATH:"%CAPI_LIB_DIR%" ^
    /LIBPATH:"%BACKEND_LIB_DIR%" ^
    /LIBPATH:"%CUDA_LIB_DIR%" ^
    /LIBPATH:"!WINLIB_DIR!" ^
    ppf_cts_capi.lib !NATIVE_LIBS!
if errorlevel 1 ( echo ERROR: build failed & exit /b 1 )

if "%STATIC%"=="1" goto :done
echo Staging runtime DLLs next to spike.exe...
if exist "%BACKEND_LIB_DIR%\libsimbackend_cuda.dll" copy /y "%BACKEND_LIB_DIR%\libsimbackend_cuda.dll" "%OUT_DIR%\" >nul
for %%F in ("%CUDA_BIN_DIR%\cudart64_*.dll") do copy /y "%%F" "%OUT_DIR%\" >nul

:done
echo.
echo [DONE] spike built at %OUT_DIR%\spike.exe
endlocal
