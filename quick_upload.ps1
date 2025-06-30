# quick_upload.ps1 - 빠른 Git 업로드 및 서버 동기화
param(
    [string]$message = "코드 업데이트: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

Write-Host "=== 빠른 Git 업로드 시작 ===" -ForegroundColor Green

# Git 상태 확인
Write-Host "1. Git 상태 확인 중..." -ForegroundColor Yellow
git status

# 모든 변경 사항 추가
Write-Host "2. 모든 변경 사항 추가 중..." -ForegroundColor Yellow
git add .

# 커밋
Write-Host "3. 커밋 중..." -ForegroundColor Yellow
git commit -m $message

# 푸시
Write-Host "4. GitHub에 푸시 중..." -ForegroundColor Yellow
git push origin main

# 서버 동기화 (SSH 키가 설정된 경우)
Write-Host "5. 웹서버 동기화 중..." -ForegroundColor Yellow
try {
    ssh outerwoolf@34.47.77.230 "cd /home/outerwoolf/AlphaGenesis && git pull origin main && echo '서버 동기화 완료'"
    Write-Host "=== 업로드 및 동기화 완료! ===" -ForegroundColor Green
} catch {
    Write-Host "서버 동기화 실패 (SSH 연결 확인 필요)" -ForegroundColor Red
    Write-Host "=== GitHub 업로드는 완료! ===" -ForegroundColor Green
} 