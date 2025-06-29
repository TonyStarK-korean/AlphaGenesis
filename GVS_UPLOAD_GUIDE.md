# GVS 서버 업로드 및 실행 가이드 (AlphaGenesis 프로젝트)

## 📁 프로젝트 구조
```
AlphaGenesis/
├── core/           # 핵심 트레이딩 엔진
├── dashboard/      # 웹 대시보드
├── ml/            # 머신러닝 모델
├── strategies/    # 트레이딩 전략
├── exchange/      # 거래소 연동
├── data/          # 데이터 저장소
├── config/        # 설정 파일
├── utils/         # 유틸리티 함수
├── logs/          # 로그 파일
├── docs/          # 문서
├── requirements.txt
├── run_dashboard.py
├── run_ml_backtest.py
└── README.md
```

## 🚀 GVS 서버 업로드 방법 (Windows)

### 1. 프로젝트 압축 (Windows PowerShell)
```powershell
# 현재 프로젝트 폴더에서 실행
cd C:\Project\AlphaGenesis

# ZIP 파일 생성
Compress-Archive -Path "C:\Project\AlphaGenesis\*" -DestinationPath "AlphaGenesis.zip" -Force

# 또는 Windows 탐색기에서:
# 1. AlphaGenesis 폴더 선택
# 2. 우클릭 → "압축 파일로 보내기" → "압축된 폴더"
```

### 2. GVS 서버 접속 방법

#### 방법 A: WinSCP 사용 (권장)
1. **WinSCP 다운로드**: https://winscp.net/
2. **연결 설정**:
   - 프로토콜: SFTP
   - 호스트명: GVS 서버 IP 주소
   - 포트: 22
   - 사용자명: GVS 계정명
   - 비밀번호: GVS 계정 비밀번호

#### 방법 B: FileZilla 사용
1. **FileZilla 다운로드**: https://filezilla-project.org/
2. **연결 설정**:
   - 호스트: GVS 서버 IP
   - 사용자명: GVS 계정명
   - 비밀번호: GVS 계정 비밀번호
   - 포트: 22

#### 방법 C: 명령줄 (PowerShell)
```powershell
# SCP를 통한 업로드 (OpenSSH 설치 필요)
scp AlphaGenesis.zip username@gvs-server-ip:/home/username/
```

### 3. 파일 업로드
1. **WinSCP/FileZilla에서**:
   - 로컬 창: `AlphaGenesis.zip` 파일 선택
   - 원격 창: `/home/username/` 폴더로 이동
   - 파일을 드래그 앤 드롭으로 업로드

2. **업로드 완료 확인**:
   - 원격 창에서 `AlphaGenesis.zip` 파일이 보이는지 확인

### 4. GVS 서버에서 압축 해제

#### SSH 접속
```bash
# PuTTY 또는 Windows Terminal에서
ssh username@gvs-server-ip
```

#### 압축 해제 및 설정
```bash
# 홈 디렉토리로 이동
cd /home/username

# 압축 해제
unzip AlphaGenesis.zip

# 프로젝트 폴더로 이동
cd AlphaGenesis

# 실행 권한 부여
chmod +x run_dashboard.py
chmod +x run_ml_backtest.py
```

## 🔄 실시간 동기화 설정 (로컬 ↔ 서버)

### 방법 1: WinSCP 동기화 (가장 간단)

#### 1. WinSCP에서 동기화 설정
```powershell
# WinSCP에서:
# 1. 로컬 폴더: C:\Project\AlphaGenesis
# 2. 원격 폴더: /home/username/AlphaGenesis
# 3. 도구 → 동기화 → 설정
```

#### 2. 자동 동기화 스크립트 생성
```powershell
# sync_to_gvs.ps1 생성
$localPath = "C:\Project\AlphaGenesis"
$remotePath = "/home/username/AlphaGenesis"
$serverIP = "GVS-SERVER-IP"
$username = "your-username"

# WinSCP 명령어로 동기화
& "C:\Program Files (x86)\WinSCP\WinSCP.com" /command ^
    "open sftp://$username@$serverIP" ^
    "synchronize local $localPath $remotePath" ^
    "exit"
```

#### 3. 파일 변경 감지 자동 동기화
```powershell
# FileSystemWatcher를 사용한 실시간 동기화
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = "C:\Project\AlphaGenesis"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

Register-ObjectEvent $watcher "Changed" -Action {
    Write-Host "파일 변경 감지: $($Event.SourceEventArgs.FullPath)"
    & "C:\Project\AlphaGenesis\sync_to_gvs.ps1"
}
```

### 방법 2: rsync 사용 (고급)

#### 1. Windows에 rsync 설치
```powershell
# Chocolatey 사용
choco install rsync

# 또는 WSL 사용
wsl --install
```

#### 2. rsync 동기화 스크립트
```bash
# sync_rsync.sh 생성
#!/bin/bash
LOCAL_PATH="/mnt/c/Project/AlphaGenesis"
REMOTE_PATH="/home/username/AlphaGenesis"
SERVER="username@gvs-server-ip"

# 실시간 동기화
rsync -avz --delete \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    $LOCAL_PATH/ $SERVER:$REMOTE_PATH/

echo "동기화 완료: $(date)"
```

#### 3. 자동 실행 설정
```bash
# crontab에 등록 (Linux/WSL)
crontab -e

# 1분마다 동기화
* * * * * /home/username/sync_rsync.sh
```

### 방법 3: Git 기반 동기화 (권장)

#### 1. GitHub 저장소 설정 (로컬 PC에서)
```bash
# 로컬 PC에서 AlphaGenesis 폴더로 이동
cd C:\Project\AlphaGenesis

# Git 초기화 (이미 되어 있다면 생략)
git init

# GitHub 원격 저장소 추가 (본인의 GitHub 저장소 URL로 변경)
git remote add origin https://github.com/your-username/AlphaGenesis.git
# 또는 SSH 방식
git remote add origin git@github.com:your-username/AlphaGenesis.git

# .gitignore 파일 생성 (중요!)
cat > .gitignore << EOF
*.log
__pycache__/
*.pyc
.env
logs/
data/temp/
*.zip
.DS_Store
.vscode/
.idea/
EOF

# 초기 커밋
git add .
git commit -m "Initial commit: AlphaGenesis project"
git push -u origin main
```

#### 2. GVS 서버에 Git 설정
```bash
# GVS 서버에서
cd /home/outerwoolf

# Git 설치 (필요시)
sudo apt-get update
sudo apt-get install git

# Git 사용자 정보 설정
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"

# AlphaGenesis 폴더를 Git 저장소로 클론
git clone https://github.com/your-username/AlphaGenesis.git
# 또는 SSH 방식
git clone git@github.com:your-username/AlphaGenesis.git

cd AlphaGenesis
```

#### 3. 자동 동기화 스크립트 생성 (로컬 PC)
```powershell
# auto_sync.ps1 생성
$projectPath = "C:\Project\AlphaGenesis"
$logFile = "C:\Project\AlphaGenesis\sync.log"

Write-Host "Git 자동 동기화 시작: $(Get-Date)" | Tee-Object -FilePath $logFile

# 프로젝트 폴더로 이동
Set-Location $projectPath

# Git 상태 확인
$status = git status --porcelain

if ($status) {
    Write-Host "변경사항 발견: $($status.Count)개 파일" | Tee-Object -FilePath $logFile -Append
    
    # 변경사항 추가 및 커밋
    git add .
    $commitMessage = "Auto sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    git commit -m $commitMessage
    
    # GitHub에 푸시
    git push origin main
    
    Write-Host "GitHub에 푸시 완료: $(Get-Date)" | Tee-Object -FilePath $logFile -Append
    
    # GVS 서버에서 자동 풀 (SSH를 통한 원격 명령)
    ssh outerwoolf@34.47.77.230 "cd /home/outerwoolf/AlphaGenesis && git pull origin main"
    
    Write-Host "GVS 서버 동기화 완료: $(Get-Date)" | Tee-Object -FilePath $logFile -Append
} else {
    Write-Host "변경사항 없음: $(Get-Date)" | Tee-Object -FilePath $logFile -Append
}
```

#### 4. 파일 변경 감지 자동 동기화 (로컬 PC)
```powershell
# realtime_git_sync.ps1 생성
$projectPath = "C:\Project\AlphaGenesis"
$syncScript = "C:\Project\AlphaGenesis\auto_sync.ps1"

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
Write-Host "변경된 파일이 자동으로 GitHub에 푸시되고 GVS 서버에 반영됩니다."

# 무한 대기
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "실시간 동기화 종료"
}
```

#### 5. GVS 서버에서 자동 풀 스크립트
```bash
# auto_pull.sh 생성 (GVS 서버에서)
cat > /home/outerwoolf/auto_pull.sh << 'EOF'
#!/bin/bash
cd /home/outerwoolf/AlphaGenesis

# Git 상태 확인
git fetch origin

# 원격 저장소에 새로운 커밋이 있는지 확인
if [ "$(git rev-list HEAD...origin/main --count)" != "0" ]; then
    echo "$(date): 새로운 변경사항 발견, 풀링 중..."
    git pull origin main
    
    # 필요시 의존성 업데이트
    if [ -f "requirements.txt" ]; then
        source alphagenesis_env/bin/activate
        pip install -r requirements.txt --upgrade
    fi
    
    echo "$(date): 동기화 완료"
else
    echo "$(date): 변경사항 없음"
fi
EOF

# 실행 권한 부여
chmod +x /home/outerwoolf/auto_pull.sh

# crontab에 등록 (5분마다 체크)
crontab -e
# 아래 줄 추가:
# */5 * * * * /home/outerwoolf/auto_pull.sh >> /home/outerwoolf/auto_pull.log 2>&1
```

#### 6. VS Code Git 통합 (개발자용)
```json
// .vscode/settings.json
{
    "git.autofetch": true,
    "git.autofetchPeriod": 60,
    "git.confirmSync": false,
    "git.enableSmartCommit": true,
    "git.autocommit": false,
    "git.autopush": false
}
```

#### 7. GitHub Actions 자동 배포 (고급)
```yaml
# .github/workflows/deploy.yml
name: Deploy to GVS Server

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to GVS Server
      uses: appleboy/ssh-action@v0.1.4
      with:
        host: ${{ secrets.GVS_HOST }}
        username: ${{ secrets.GVS_USERNAME }}
        key: ${{ secrets.GVS_SSH_KEY }}
        script: |
          cd /home/outerwoolf/AlphaGenesis
          git pull origin main
          source alphagenesis_env/bin/activate
          pip install -r requirements.txt --upgrade
```

#### 8. 사용법 요약

**로컬 PC에서:**
1. **GitHub 저장소 생성** 및 로컬 프로젝트 연결
2. **auto_sync.ps1** 실행으로 수동 동기화
3. **realtime_git_sync.ps1** 실행으로 실시간 자동 동기화

**GVS 서버에서:**
1. **Git 저장소 클론**
2. **auto_pull.sh** 스크립트로 자동 업데이트
3. **crontab** 등록으로 주기적 체크

#### 9. 장점
- ✅ **버전 관리**: 모든 변경사항 추적 가능
- ✅ **백업**: GitHub에 자동 백업
- ✅ **협업**: 여러 개발자 동시 작업 가능
- ✅ **롤백**: 문제 발생 시 이전 버전으로 복구
- ✅ **자동화**: 파일 변경 시 자동 동기화
- ✅ **안전성**: SSH 키 인증으로 보안 강화

#### 10. 문제 해결
```bash
# Git 상태 확인
git status
git log --oneline -5

# 강제 동기화
git reset --hard HEAD
git clean -fd
git pull origin main

# SSH 키 문제
ssh -T git@github.com
```

---

## 🎯 Git 동기화 방법 비교

| 방법 | 장점 | 단점 | 추천도 |
|------|------|------|--------|
| **수동 푸시** | 간단, 안전 | 수동 실행 필요 | ⭐⭐⭐ |
| **자동 푸시** | 실시간 동기화 | 리소스 사용 | ⭐⭐⭐⭐ |
| **GitHub Actions** | 완전 자동화 | 설정 복잡 | ⭐⭐⭐⭐ |
| **Crontab 풀** | 서버 자동 업데이트 | 지연 발생 | ⭐⭐⭐⭐ |

## 🔧 GVS 서버 환경 설정

### 1. Python 환경 확인
```bash
# Python 버전 확인
python3 --version

# pip 설치 확인
pip3 --version

# 가상환경 도구 확인
python3 -m venv --help
```

### 2. 가상환경 생성 및 의존성 설치
```bash
# 가상환경 생성
python3 -m venv alphagenesis_env

# 가상환경 활성화
source alphagenesis_env/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt

# 설치 확인
pip list
```

### 3. 포트 및 방화벽 설정
```bash
# 포트 5000 개방 (관리자 권한 필요)
sudo ufw allow 5000

# 또는 GVS 서버 관리자에게 요청:
# "포트 5000을 외부 접속용으로 개방해주세요"
```

## 🌐 웹 대시보드 실행

### 1. 백그라운드 실행 (권장)
```bash
# 가상환경 활성화
source alphagenesis_env/bin/activate

# 백그라운드 실행
nohup python3 run_dashboard.py > dashboard.log 2>&1 &

# 프로세스 ID 확인
echo $!
```

### 2. Screen 세션 사용 (대안)
```bash
# Screen 설치 (필요시)
sudo apt-get install screen

# 새 세션 생성
screen -S alphagenesis

# 가상환경 활성화 및 실행
source alphagenesis_env/bin/activate
python3 run_dashboard.py

# 세션 분리: Ctrl+A, D
# 세션 재접속: screen -r alphagenesis
```

### 3. 실행 상태 확인
```bash
# 프로세스 확인
ps aux | grep python

# 포트 사용 확인
netstat -tlnp | grep 5000

# 로그 확인
tail -f dashboard.log

# 실시간 로그 모니터링
tail -f dashboard.log | grep -E "(ERROR|WARNING|INFO)"
```

## 🔗 외부 접속 설정

### 1. GVS 서버 IP 확인
```bash
# 서버 IP 확인
hostname -I
# 또는
ip addr show eth0
```

### 2. 웹 브라우저에서 접속
```
http://GVS-SERVER-IP:5000
```

### 3. 접속 테스트
```bash
# 서버에서 로컬 접속 테스트
curl http://localhost:5000

# 또는 wget 사용
wget -qO- http://localhost:5000
```

## 📊 모니터링 및 관리

### 1. 자동 재시작 스크립트 생성
```bash
# restart_script.sh 생성
cat > restart_script.sh << 'EOF'
#!/bin/bash
cd /home/username/AlphaGenesis
source alphagenesis_env/bin/activate

while true; do
    echo "$(date): Starting AlphaGenesis Dashboard..."
    python3 run_dashboard.py
    echo "$(date): Dashboard stopped, restarting in 10 seconds..."
    sleep 10
done
EOF

# 실행 권한 부여
chmod +x restart_script.sh

# 백그라운드 실행
nohup ./restart_script.sh > restart.log 2>&1 &
```

### 2. 로그 관리
```bash
# 로그 파일 크기 확인
ls -lh *.log

# 오래된 로그 정리
find . -name "*.log" -size +100M -delete

# 로그 로테이션 설정 (선택사항)
sudo logrotate -f /etc/logrotate.conf
```

### 3. 프로세스 관리
```bash
# 프로세스 ID 찾기
pgrep -f run_dashboard.py

# 프로세스 종료
pkill -f run_dashboard.py

# 강제 종료
pkill -9 -f run_dashboard.py

# 모든 Python 프로세스 확인
ps aux | grep python
```

## 🔒 보안 설정

### 1. 기본 인증 추가 (선택사항)
```python
# dashboard/app.py에 추가
from functools import wraps
from flask import request, Response

def check_auth(username, password):
    return username == 'admin' and password == 'your_secure_password'

def authenticate():
    return Response('인증이 필요합니다.\n'
                   '올바른 사용자명과 비밀번호로 로그인하세요.', 401,
                   {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# 메인 라우트에 인증 적용
@app.route('/')
@requires_auth
def index():
    return render_template('index.html')
```

### 2. 방화벽 설정
```bash
# 특정 IP만 접속 허용
sudo ufw allow from YOUR_IP_ADDRESS to any port 5000

# 또는 VPN 사용 권장
```

## 🚨 문제 해결

### 1. 포트 충돌
```bash
# 포트 사용 확인
lsof -i :5000

# 다른 포트 사용
python3 run_dashboard.py --port 5001
```

### 2. 메모리 부족
```bash
# 메모리 사용량 확인
free -h

# 프로세스 메모리 확인
ps aux --sort=-%mem | head -10
```

### 3. 디스크 공간 부족
```bash
# 디스크 사용량 확인
df -h

# 큰 파일 찾기
find . -type f -size +100M
```

### 4. Python 모듈 오류
```bash
# 가상환경 재활성화
source alphagenesis_env/bin/activate

# 의존성 재설치
pip install -r requirements.txt --force-reinstall

# 캐시 정리
pip cache purge
```

### 5. 동기화 문제 해결
```bash
# Git 상태 확인
git status
git log --oneline -5

# 강제 동기화
git reset --hard HEAD
git clean -fd

# rsync 강제 동기화
rsync -avz --delete --force $LOCAL_PATH/ $SERVER:$REMOTE_PATH/
```

## 📱 모바일 접속

### 1. 모바일 브라우저에서 접속
```
http://GVS-SERVER-IP:5000
```

### 2. 반응형 디자인 확인
- 대시보드는 이미 모바일 친화적으로 설계됨
- 터치 인터페이스 지원

## 📞 지원 및 연락처

문제 발생 시 다음 순서로 확인:
1. **로그 확인**: `tail -f dashboard.log`
2. **프로세스 상태**: `ps aux | grep python`
3. **네트워크 연결**: `netstat -tlnp | grep 5000`
4. **동기화 상태**: `git status` 또는 `rsync --dry-run`
5. **GVS 서버 관리자에게 문의**

## 🔄 업데이트 방법

### 1. 새 버전 업로드
```bash
# 기존 프로세스 종료
pkill -f run_dashboard.py

# 새 파일 업로드 (WinSCP/FileZilla 사용)
# AlphaGenesis.zip → /home/username/

# 압축 해제
cd /home/username
unzip -o AlphaGenesis.zip

# 의존성 업데이트
cd AlphaGenesis
source alphagenesis_env/bin/activate
pip install -r requirements.txt --upgrade

# 재시작
nohup python3 run_dashboard.py > dashboard.log 2>&1 &
```

### 2. 설정 파일 백업
```bash
# 중요 설정 파일 백업
cp config/settings.py config/settings_backup.py
cp .env .env_backup
```

## 🎯 동기화 방법 비교

| 방법 | 장점 | 단점 | 추천도 |
|------|------|------|--------|
| **WinSCP 동기화** | 간단, GUI 지원 | 수동 실행 필요 | ⭐⭐⭐ |
| **rsync** | 빠름, 자동화 가능 | 설정 복잡 | ⭐⭐⭐⭐ |
| **Git** | 버전 관리, 안전함 | 학습 곡선 | ⭐⭐⭐⭐⭐ |
| **VS Code Remote** | 개발 편의성 | VS Code 필요 | ⭐⭐⭐⭐ |
| **Syncthing** | 완전 자동화 | 리소스 사용 | ⭐⭐⭐⭐ |