# realtime_git_sync.ps1
$projectPath = "C:\Project\AlphaGenesis"
$syncScript = "$projectPath\auto_sync.ps1"

Write-Host "실시간 Git 동기화 시작: $(Get-Date)"
Write-Host "프로젝트 경로: $projectPath"

# FileSystemWatcher 설정
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $projectPath
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# 변경 감지 이벤트 등록
Register-ObjectEvent $watcher "Changed" -Action {
    $filePath = $Event.SourceEventArgs.FullPath
    $fileName = $Event.SourceEventArgs.Name

    # .git 폴더나 로그 파일은 제외
    if ($fileName -notlike ".git*" -and $fileName -notlike "*.log") {
        Write-Host "파일 변경 감지: $fileName - $(Get-Date)"

        # 2초 대기 (파일 저장 완료 대기)
        Start-Sleep -Seconds 2

        # 동기화 스크립트 실행
        & $syncScript
    }
}

Write-Host "실시간 감시 중... (종료하려면 Ctrl+C)"
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "실시간 동기화 종료"
}