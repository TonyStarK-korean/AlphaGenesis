@echo off
chcp 65001 > nul
echo === 빠른 Git 업로드 시작 ===

echo 1. Git 상태 확인 중...
git status

echo 2. 모든 변경 사항 추가 중...
git add .

echo 3. 커밋 중...
if "%~1"=="" (
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set mydate=%%c-%%a-%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a:%%b
    set commit_msg=코드 업데이트: !mydate! !mytime!
) else (
    set commit_msg=%~1
)
git commit -m "%commit_msg%"

echo 4. GitHub에 푸시 중...
git push origin main

echo 5. 웹서버 동기화 중...
ssh outerwoolf@34.47.77.230 "cd /home/outerwoolf/AlphaGenesis && git pull origin main && echo 서버 동기화 완료"

echo === 업로드 및 동기화 완료! ===
pause 