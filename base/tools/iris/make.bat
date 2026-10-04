@echo off
REM ============================================================================
REM Iris - C Image Generation Engine
REM Windows build script (port of the Unix Makefile)
REM
REM Toolchain: Visual Studio with the LLVM/clang component, plus the Vulkan SDK.
REM
REM This script initializes the MSVC x64 environment itself (via vswhere +
REM vcvars64.bat), so it can be run from an ordinary cmd/PowerShell prompt --
REM no need to open an "x64 Native Tools Command Prompt for VS" first. If you
REM already are in a developer prompt, that environment is reused as-is.
REM The Vulkan SDK must be installed and the VULKAN_SDK environment variable
REM set (the SDK installer does this automatically).
REM
REM Usage:
REM   make.bat            Build the Vulkan backend (default)
REM   make.bat vulkan     Build the Vulkan backend (GPU compute)
REM   make.bat cpu        Build the pure-C backend (slow, no deps)
REM   make.bat lib        Build the static library
REM   make.bat clean      Remove build artifacts
REM   make.bat info       Show build configuration
REM   make.bat help       Show this help
REM ============================================================================

setlocal enabledelayedexpansion

REM --- Configuration ----------------------------------------------------------
REM Use the GNU-style clang driver (accepts the gcc flags from the Makefile).
set "CC=clang"
set "AR=llvm-ar"
set "BUILD_DIR=build"

REM Generated SPIR-V headers live in BUILD_DIR, so it is always on the include
REM path. -march=native and -ffast-math mirror the Makefile.
set "CFLAGS_BASE=-Wall -Wextra -O3 -march=native -ffast-math -I%BUILD_DIR% -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS"

REM Source files (matches SRCS in the Makefile).
set "SRCS=iris.c iris_kernels.c iris_tokenizer.c iris_vae.c iris_transformer_flux.c iris_sample.c iris_image.c jpeg.c iris_safetensors.c iris_gguf.c iris_qwen3.c iris_qwen3_tokenizer.c iris_upscale.c iris_depth.c"

set "TARGET=%BUILD_DIR%\iris.exe"
set "LIBFILE=%BUILD_DIR%\libiris.a"

REM --- Dispatch ---------------------------------------------------------------
set "GOAL=%~1"
if "%GOAL%"=="" set "GOAL=vulkan"

if /I "%GOAL%"=="vulkan" goto :vulkan
if /I "%GOAL%"=="cpu"    goto :cpu
if /I "%GOAL%"=="lib"    goto :lib
if /I "%GOAL%"=="clean"  goto :clean
if /I "%GOAL%"=="info"   goto :info
if /I "%GOAL%"=="help"   goto :help

echo Unknown target: %GOAL%
echo.
goto :help

REM ============================================================================
REM Backend: vulkan (cross-platform GPU compute, GEMM offload)
REM ============================================================================
:vulkan
call :setup_msvc
if errorlevel 1 exit /b 1
if not defined VULKAN_SDK (
    echo ERROR: VULKAN_SDK is not set. Install the Vulkan SDK from
    echo        https://vulkan.lunarg.com/ and reopen your shell.
    exit /b 1
)
set "GLSLC=%VULKAN_SDK%\Bin\glslc.exe"
if not exist "%GLSLC%" (
    echo ERROR: glslc not found at "%GLSLC%".
    exit /b 1
)

set "CFLAGS=%CFLAGS_BASE% -DUSE_VULKAN -I"%VULKAN_SDK%\Include""
set "LDFLAGS=-L"%VULKAN_SDK%\Lib" -lvulkan-1"

call :do_clean
call :ensure_build_dir

echo.
echo Compiling SPIR-V shaders...
REM Compile every GLSL compute shader to a SPIR-V C uint32 initializer list,
REM included verbatim by iris_vulkan.c (glslc -mfmt=c).
for %%s in (iris_vulkan_*.comp) do (
    echo   GLSLC %%s
    "%GLSLC%" -fshader-stage=compute -O "%%s" -mfmt=c -o "%BUILD_DIR%\%%~ns_spv.h"
    if errorlevel 1 goto :error
)

echo.
echo Compiling sources (Vulkan backend)...
call :compile_objs %SRCS% iris_vulkan.c main.c
if errorlevel 1 goto :error

echo.
echo Linking %TARGET%...
"%CC%" %CFLAGS% -o "%TARGET%" !OBJS! %LDFLAGS%
if errorlevel 1 goto :error

echo.
echo Built with VULKAN backend (GPU GEMM offload + resident VAE decode)
exit /b 0

REM ============================================================================
REM Backend: cpu (pure C, no BLAS)
REM ============================================================================
:cpu
call :setup_msvc
if errorlevel 1 exit /b 1
set "CFLAGS=%CFLAGS_BASE% -DCPU_BUILD"
set "LDFLAGS="

call :do_clean
call :ensure_build_dir

echo.
echo Compiling sources (CPU backend)...
call :compile_objs %SRCS% main.c
if errorlevel 1 goto :error

echo.
echo Linking %TARGET%...
"%CC%" %CFLAGS% -o "%TARGET%" !OBJS! %LDFLAGS%
if errorlevel 1 goto :error

echo.
echo Built with CPU backend (pure C, no BLAS)
echo This will be slow but has zero dependencies.
exit /b 0

REM ============================================================================
REM Static library
REM ============================================================================
:lib
call :setup_msvc
if errorlevel 1 exit /b 1
set "CFLAGS=%CFLAGS_BASE% -DCPU_BUILD"
call :ensure_build_dir

echo.
echo Compiling sources (static library)...
call :compile_objs %SRCS%
if errorlevel 1 goto :error

echo.
echo Archiving %LIBFILE%...
"%AR%" rcs "%LIBFILE%" !OBJS!
if errorlevel 1 goto :error

echo.
echo Built static library %LIBFILE%
exit /b 0

REM ============================================================================
REM Utilities
REM ============================================================================
:clean
call :do_clean
echo Removed %BUILD_DIR%
exit /b 0

:info
echo Platform:  Windows (%PROCESSOR_ARCHITECTURE%)
echo Compiler:  %CC%
echo Build dir: %BUILD_DIR%
if defined VULKAN_SDK (
    echo Vulkan SDK: %VULKAN_SDK%
) else (
    echo Vulkan SDK: ^(VULKAN_SDK not set^)
)
echo.
echo Available backends for this platform:
echo   cpu     - Pure C (always available)
echo   vulkan  - Cross-platform GPU via Vulkan compute
exit /b 0

:help
echo Iris - Build Targets (Windows)
echo.
echo Choose a backend:
echo   make.bat cpu      - Pure C, no dependencies (slow)
echo   make.bat vulkan   - Cross-platform GPU via Vulkan compute (default)
echo.
echo Other targets:
echo   make.bat clean    - Remove build artifacts
echo   make.bat info     - Show build configuration
echo   make.bat lib      - Build static library
echo.
echo Example: make.bat vulkan ^&^& build\iris.exe -d flux-klein-4b -p "a cat" -o cat.png
exit /b 0

REM ============================================================================
REM Subroutines
REM ============================================================================

REM Compile each argument (.c file) to BUILD_DIR\<name>.o and accumulate OBJS.
:compile_objs
set "OBJS="
for %%f in (%*) do (
    echo   CC %%f
    "%CC%" %CFLAGS% -c -o "%BUILD_DIR%\%%~nf.o" "%%f"
    if errorlevel 1 exit /b 1
    set "OBJS=!OBJS! %BUILD_DIR%\%%~nf.o"
)
exit /b 0

:ensure_build_dir
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
exit /b 0

REM Make sure the MSVC x64 toolchain (clang's libs/linker search paths) is on
REM the environment. Skipped if we are already inside a VS developer prompt.
:setup_msvc
if defined VSCMD_VER exit /b 0
set "_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%_VSWHERE%" (
    echo ERROR: vswhere.exe not found at "%_VSWHERE%".
    echo        Install Visual Studio with the "Desktop development with C++"
    echo        workload and the "C++ Clang tools for Windows" component.
    exit /b 1
)
set "_VSPATH="
for /f "usebackq tokens=*" %%i in (`"%_VSWHERE%" -latest -products * -property installationPath`) do set "_VSPATH=%%i"
if not defined _VSPATH (
    echo ERROR: No Visual Studio installation found by vswhere.
    exit /b 1
)
set "_VCVARS=%_VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%_VCVARS%" (
    echo ERROR: vcvars64.bat not found at "%_VCVARS%".
    exit /b 1
)
echo Initializing MSVC x64 environment...
call "%_VCVARS%" >nul 2>&1
if not defined LIB (
    echo ERROR: vcvars64.bat ran but the environment was not initialized.
    exit /b 1
)
exit /b 0

:do_clean
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
exit /b 0

:error
echo.
echo BUILD FAILED.
exit /b 1
