@echo off
REM GVS 서버 배포 자동화 스크립트
REM AlphaGenesis 프로젝트를 GVS 서버로 배포

echo.
echo =====================================
echo   AlphaGenesis GVS 서버 배포 도구
echo =====================================
echo.

REM Python 환경 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    pause
    exit /b 1
)

REM 필요한 패키지 설치
echo 📦 필요한 패키지 설치 중...
pip install paramiko scp

REM 배포 메뉴
echo.
echo 다음 중 하나를 선택하세요:
echo 1. 전체 배포 (deploy)
echo 2. 상태 확인 (status)
echo 3. 서비스 재시작 (restart)
echo.
set /p choice=선택 (1-3): 

if "%choice%"=="1" (
    echo.
    echo 🚀 전체 배포를 시작합니다...
    python deployment/gvs_deploy.py deploy
) else if "%choice%"=="2" (
    echo.
    echo 📊 서버 상태를 확인합니다...
    python deployment/gvs_deploy.py status
) else if "%choice%"=="3" (
    echo.
    echo 🔄 서비스를 재시작합니다...
    python deployment/gvs_deploy.py restart
) else (
    echo ❌ 잘못된 선택입니다.
    pause
    exit /b 1
)

echo.
echo 작업이 완료되었습니다.
echo 웹 대시보드 접속: http://34.47.77.230:9000
echo.
pause