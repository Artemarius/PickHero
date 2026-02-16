@echo off
setlocal enabledelayedexpansion

:: PickHero build script
:: Builds a single-file .exe via PyInstaller
:: Usage:  build.bat           — normal build
::         build.bat --clean   — wipe build/dist dirs then build

set "SPEC=pickhero.spec"
set "EXE=dist\PickHero.exe"

:: ── Pre-flight checks ──────────────────────────────────────────────────────

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

:: ── Handle --clean flag ────────────────────────────────────────────────────

if "%~1"=="--clean" (
    echo Cleaning previous build artifacts...
    if exist build rmdir /s /q build
    if exist dist  rmdir /s /q dist
    echo Done.
    echo.
)

:: ── Install / upgrade PyInstaller ──────────────────────────────────────────

echo Installing/upgrading PyInstaller...
pip install --upgrade pyinstaller >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip install pyinstaller failed. Check your Python environment.
    pause
    exit /b 1
)

:: ── Verify spec file exists ────────────────────────────────────────────────

if not exist "%SPEC%" (
    echo ERROR: %SPEC% not found. Run this script from the project root.
    pause
    exit /b 1
)

:: ── Build ──────────────────────────────────────────────────────────────────

echo.
echo Building PickHero...
echo.
pyinstaller "%SPEC%" --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed. See output above for details.
    pause
    exit /b 1
)

:: ── Post-build summary ─────────────────────────────────────────────────────

echo.
if exist "%EXE%" (
    echo ========================================
    echo  Build succeeded!
    echo  Output: %EXE%
    for %%F in ("%EXE%") do (
        set "SIZE=%%~zF"
        set /a "MB=!SIZE! / 1048576"
        echo  Size:   ~!MB! MB
    )
    echo ========================================
    echo.
    echo Next steps:
    echo   1. Place a "songs" folder next to PickHero.exe containing
    echo      your .gp3/.gp4/.gp5/.gp7/.gp8 tab files.
    echo   2. Run PickHero.exe to launch the app.
    echo.
    echo Tip: You can also download tabs from within the app
    echo      by pressing S on the song selection screen.
) else (
    echo WARNING: Build appeared to succeed but %EXE% was not found.
)

echo.
pause
