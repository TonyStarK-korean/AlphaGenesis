#!/bin/bash
# AlphaGenesis 서버 자동 업데이트 설정 스크립트
# GVS 서버에서 실행

echo "=========================================="
echo "  AlphaGenesis 자동 업데이트 설정"
echo "=========================================="
echo

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 루트 권한 확인
if [ "$EUID" -ne 0 ]; then
    print_error "이 스크립트는 루트 권한으로 실행해야 합니다."
    echo "사용법: sudo $0"
    exit 1
fi

# 1. 시스템 업데이트
print_step "1. 시스템 패키지 업데이트..."
apt update && apt upgrade -y
print_status "시스템 업데이트 완료"
echo

# 2. 필수 패키지 설치
print_step "2. 필수 패키지 설치..."
apt install -y python3 python3-pip git curl wget vim htop
print_status "필수 패키지 설치 완료"
echo

# 3. Python 패키지 설치
print_step "3. Python 패키지 설치..."
pip3 install --upgrade pip
pip3 install requests paramiko scp
print_status "Python 패키지 설치 완료"
echo

# 4. 프로젝트 디렉토리 확인
PROJECT_DIR="/home/outerwoolf/AlphaGenesis"
print_step "4. 프로젝트 디렉토리 확인..."

if [ ! -d "$PROJECT_DIR" ]; then
    print_warning "프로젝트 디렉토리가 없습니다. GitHub에서 클론합니다..."
    cd /home/outerwoolf
    git clone https://github.com/TonyStarK-korean/AlphaGenesis.git
    print_status "프로젝트 클론 완료"
else
    print_status "프로젝트 디렉토리가 이미 존재합니다."
fi
echo

# 5. 설정 파일 생성
print_step "5. 자동 업데이트 설정 파일 생성..."
CONFIG_FILE="$PROJECT_DIR/deployment/server_config.json"
mkdir -p "$PROJECT_DIR/deployment"

cat > "$CONFIG_FILE" << 'EOF'
{
  "github": {
    "repository": "TonyStarK-korean/AlphaGenesis",
    "branch": "main",
    "api_url": "https://api.github.com/repos/TonyStarK-korean/AlphaGenesis",
    "check_interval": 300
  },
  "server": {
    "repo_path": "/root/AlphaGenesis",
    "service_name": "alphagenesis",
    "backup_count": 5,
    "auto_restart": true
  },
  "notification": {
    "enabled": false,
    "webhook_url": "",
    "telegram_bot_token": "",
    "telegram_chat_id": ""
  }
}
EOF

print_status "설정 파일 생성 완료: $CONFIG_FILE"
echo

# 6. 로그 디렉토리 생성
print_step "6. 로그 디렉토리 설정..."
mkdir -p /var/log
touch /var/log/alphagenesis_auto_update.log
chmod 644 /var/log/alphagenesis_auto_update.log
print_status "로그 디렉토리 설정 완료"
echo

# 7. 자동 업데이트 서비스 설치
print_step "7. 자동 업데이트 서비스 설치..."

# systemd 서비스 파일 생성
cat > /etc/systemd/system/alphagenesis-auto-update.service << 'EOF'
[Unit]
Description=AlphaGenesis Auto Update Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /root/AlphaGenesis/deployment/server_auto_update.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# systemd 리로드
systemctl daemon-reload
systemctl enable alphagenesis-auto-update
print_status "자동 업데이트 서비스 설치 완료"
echo

# 8. 메인 서비스 설정
print_step "8. 메인 AlphaGenesis 서비스 설정..."

# 메인 서비스 파일 생성
cat > /etc/systemd/system/alphagenesis.service << 'EOF'
[Unit]
Description=AlphaGenesis Trading Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/AlphaGenesis
Environment=PYTHONPATH=/root/AlphaGenesis
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 /root/AlphaGenesis/dashboard/app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable alphagenesis
print_status "메인 서비스 설정 완료"
echo

# 9. 방화벽 설정
print_step "9. 방화벽 설정..."
ufw allow 22/tcp    # SSH
ufw allow 9000/tcp  # AlphaGenesis Dashboard
ufw --force enable
print_status "방화벽 설정 완료"
echo

# 10. 의존성 설치
print_step "10. AlphaGenesis 의존성 설치..."
cd "$PROJECT_DIR"

# requirements.txt가 있는 경우
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    # 기본 패키지 설치
    pip3 install flask flask-cors pandas numpy ccxt scikit-learn xgboost optuna asyncio aiohttp python-dotenv
fi
print_status "의존성 설치 완료"
echo

# 11. 서비스 시작
print_step "11. 서비스 시작..."

# 메인 서비스 시작
systemctl start alphagenesis
sleep 3

# 자동 업데이트 서비스 시작
systemctl start alphagenesis-auto-update
sleep 3

print_status "서비스 시작 완료"
echo

# 12. 상태 확인
print_step "12. 서비스 상태 확인..."
echo
echo "=== AlphaGenesis 메인 서비스 ==="
systemctl status alphagenesis --no-pager -l
echo
echo "=== 자동 업데이트 서비스 ==="
systemctl status alphagenesis-auto-update --no-pager -l
echo

# 13. 포트 확인
print_step "13. 포트 상태 확인..."
echo "포트 9000 상태:"
netstat -tlnp | grep :9000 || echo "포트 9000이 열려있지 않습니다."
echo

# 14. 완료 메시지
echo "=========================================="
echo -e "${GREEN}    설정 완료!${NC}"
echo "=========================================="
echo
echo "🌐 웹 대시보드: http://$(curl -s ifconfig.me):9000"
echo "📊 로컬 접속: http://localhost:9000"
echo
echo "📝 서비스 관리 명령어:"
echo "  - 메인 서비스 상태: systemctl status alphagenesis"
echo "  - 자동 업데이트 상태: systemctl status alphagenesis-auto-update"
echo "  - 메인 서비스 재시작: systemctl restart alphagenesis"
echo "  - 자동 업데이트 재시작: systemctl restart alphagenesis-auto-update"
echo
echo "📋 로그 확인:"
echo "  - 메인 로그: journalctl -u alphagenesis -f"
echo "  - 업데이트 로그: journalctl -u alphagenesis-auto-update -f"
echo "  - 업데이트 파일 로그: tail -f /var/log/alphagenesis_auto_update.log"
echo
echo "⚙️ 설정 파일:"
echo "  - 자동 업데이트: $CONFIG_FILE"
echo
echo "🔄 자동 업데이트:"
echo "  - GitHub 저장소를 5분마다 체크합니다"
echo "  - 새로운 커밋이 있으면 자동으로 업데이트됩니다"
echo "  - 서비스는 24시간 계속 실행됩니다"
echo
echo "✅ SSH 터미널을 닫아도 서비스는 계속 실행됩니다!"
echo

# 15. 선택적 설정
echo "추가 설정을 원하시나요?"
echo "1. 텔레그램 알림 설정"
echo "2. 웹훅 알림 설정 (Slack/Discord)"
echo "3. 설정 완료"
echo -n "선택 (1-3): "
read choice

case $choice in
    1)
        echo
        echo "텔레그램 알림 설정:"
        echo -n "봇 토큰을 입력하세요: "
        read bot_token
        echo -n "채팅 ID를 입력하세요: "
        read chat_id
        
        # 설정 파일 업데이트
        python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['notification']['enabled'] = True
config['notification']['telegram_bot_token'] = '$bot_token'
config['notification']['telegram_chat_id'] = '$chat_id'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
"
        print_status "텔레그램 알림 설정 완료"
        systemctl restart alphagenesis-auto-update
        ;;
    2)
        echo
        echo "웹훅 알림 설정:"
        echo -n "웹훅 URL을 입력하세요: "
        read webhook_url
        
        # 설정 파일 업데이트
        python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['notification']['enabled'] = True
config['notification']['webhook_url'] = '$webhook_url'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
"
        print_status "웹훅 알림 설정 완료"
        systemctl restart alphagenesis-auto-update
        ;;
    *)
        print_status "설정 완료"
        ;;
esac

echo
echo "🎉 AlphaGenesis 서버 설정이 모두 완료되었습니다!"
echo "이제 GitHub에 코드를 푸시하면 서버에서 자동으로 업데이트됩니다."