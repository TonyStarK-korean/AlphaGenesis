@echo off
REM AlphaGenesis GitHub Deployment Script
REM Simple version without Korean characters

echo.
echo ==========================================
echo   AlphaGenesis GitHub Deployment
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed
    pause
    exit /b 1
)

REM Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed
    echo Download: https://git-scm.com/
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install paramiko scp >nul 2>&1

echo.
echo Select deployment method:
echo 1. Full deployment (Local - GitHub - Server)
echo 2. Upload to GitHub only
echo 3. Deploy to server only (GitHub - Server)
echo 4. Edit config file
echo.
set /p choice=Select (1-4): 

if "%choice%"=="1" (
    echo.
    set /p message=Commit message (Enter=auto): 
    if "!message!"=="" (
        python git_deploy.py deploy
    ) else (
        python git_deploy.py deploy -m "!message!"
    )
) else if "%choice%"=="2" (
    echo.
    set /p message=Commit message (Enter=auto): 
    if "!message!"=="" (
        python git_deploy.py push
    ) else (
        python git_deploy.py push -m "!message!"
    )
) else if "%choice%"=="3" (
    echo.
    echo Deploying to server...
    python git_deploy.py pull
) else if "%choice%"=="4" (
    echo.
    echo Opening config file...
    if exist "deployment\git_config.json" (
        notepad deployment\git_config.json
    ) else (
        echo Config file not found. Please run deployment once first.
    )
) else (
    echo ERROR: Invalid selection
    pause
    exit /b 1
)

echo.
echo Task completed.
echo Web Dashboard: http://34.47.77.230:9000
echo GitHub: https://github.com/TonyStarK-korean/AlphaGenesis
echo.
pause