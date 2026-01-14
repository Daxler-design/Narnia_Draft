@echo off
REM ========================================
REM Narnia Viewer - Build Portable EXE
REM ========================================
REM
REM This script builds the Narnia Viewer as a portable Windows executable.
REM
REM Prerequisites:
REM   1. Conda environment 'narnia' activated
REM   2. PyInstaller installed: pip install pyinstaller
REM   3. All dependencies installed: pip install -r requirements.txt
REM
REM Usage:
REM   build_exe.bat          - Build exe
REM   build_exe.bat clean    - Clean build artifacts
REM

echo.
echo ========================================
echo Narnia Viewer - EXE Build Script
echo ========================================
echo.

REM Check if clean mode
if "%1"=="clean" goto CLEAN

REM ========================================
REM Build EXE
REM ========================================

echo [1/4] Checking PyInstaller installation...
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller not found
    echo Please install: pip install pyinstaller
    pause
    exit /b 1
)
echo   OK - PyInstaller found

echo.
echo [2/4] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo   OK - Build folders cleaned

echo.
echo [3/4] Building executable with PyInstaller...
echo   This may take 5-10 minutes...
pyinstaller narnia.spec
if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    echo Check the output above for errors
    pause
    exit /b 1
)

echo.
echo [4/4] Build complete!
echo.
echo Output location:
echo   dist\NarniaViewer\NarniaViewer.exe
echo.
echo Test the executable:
echo   cd dist\NarniaViewer
echo   NarniaViewer.exe
echo.

REM Open output folder
explorer dist\NarniaViewer

goto END

REM ========================================
REM Clean Mode
REM ========================================
:CLEAN
echo Cleaning build artifacts...
if exist build (
    rmdir /s /q build
    echo   Removed: build\
)
if exist dist (
    rmdir /s /q dist
    echo   Removed: dist\
)
if exist __pycache__ (
    rmdir /s /q __pycache__
    echo   Removed: __pycache__\
)
echo.
echo Clean complete!
goto END

:END
echo.
pause
