@echo off
REM GitHub 기반 자동 배포 스크립트
REM 로컬 → GitHub → GVS 서버 자동 동기화

echo.
echo ==========================================
echo   AlphaGenesis GitHub 자동 배포
echo ==========================================
echo.

REM Python 환경 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    pause
    exit /b 1
)

REM Git 설치 확인
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git이 설치되지 않았습니다.
    echo 다운로드: https://git-scm.com/
    pause
    exit /b 1
)

REM 필요한 패키지 설치
echo 📦 필요한 패키지 설치 중...
pip install paramiko scp >nul 2>&1

echo.
echo 배포 방법을 선택하세요:
echo 1. 전체 배포 (로컬 → GitHub → 서버)
echo 2. GitHub에만 업로드
echo 3. 서버에만 배포 (GitHub → 서버)
echo 4. 설정 파일 편집
echo.
set /p choice=선택 (1-4): 

if "%choice%"=="1" (
    echo.
    set /p message=커밋 메시지 입력 (Enter=자동): 
    if "!message!"=="" (
        python git_deploy.py deploy
    ) else (
        python git_deploy.py deploy -m "!message!"
    )
) else if "%choice%"=="2" (
    echo.
    set /p message=커밋 메시지 입력 (Enter=자동): 
    if "!message!"=="" (
        python git_deploy.py push
    ) else (
        python git_deploy.py push -m "!message!"
    )
) else if "%choice%"=="3" (
    echo.
    echo 🚀 서버에 배포 중...
    python git_deploy.py pull
) else if "%choice%"=="4" (
    echo.
    echo 📝 설정 파일을 엽니다...
    if exist "deployment\git_config.json" (
        notepad deployment\git_config.json
    ) else (
        echo 설정 파일이 없습니다. 먼저 배포를 한 번 실행해주세요.
    )
) else (
    echo ❌ 잘못된 선택입니다.
    pause
    exit /b 1
)

echo.
echo 작업이 완료되었습니다.
echo 웹 대시보드: http://34.47.77.230:9000
echo GitHub: https://github.com/TonyStarK-korean/AlphaGenesis
echo.
pause