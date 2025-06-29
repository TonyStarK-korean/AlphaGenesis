# auto_sync.ps1
$projectPath = "C:\Project\AlphaGenesis"
Set-Location $projectPath

# Git 상태 확인
$status = git status --porcelain

if ($status) {
    git add .
    $commitMessage = "Auto sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    git commit -m $commitMessage
    git push origin main

    # GVS 서버에 SSH로 접속해서 자동 pull (SSH 키 필요)
    ssh outerwoolf@34.47.77.230 "cd /home/outerwoolf/AlphaGenesis && git pull origin main"
}