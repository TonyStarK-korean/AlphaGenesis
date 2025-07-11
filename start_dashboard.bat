@echo off
REM AlphaGenesis 대시보드 시작 스크립트

echo.
echo ==========================================
echo   AlphaGenesis 대시보드 시작
echo ==========================================
echo.

REM Python 환경 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    pause
    exit /b 1
)

REM 필요한 패키지 설치
echo 📦 필요한 패키지 확인 중...
pip install flask flask-cors pandas numpy ccxt >nul 2>&1

REM 대시보드 시작
echo 🚀 AlphaGenesis 대시보드 시작 중...
echo.
echo 📊 접속 주소:
echo   로컬: http://localhost:9000
echo   GVS: http://34.47.77.230:9000
echo.
echo ⚠️  종료하려면 Ctrl+C를 누르세요.
echo.

REM 대시보드 실행
python dashboard/app.py

echo.
echo 대시보드가 종료되었습니다.
pause