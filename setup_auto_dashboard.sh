#!/bin/bash

# AlphaGenesis 24시간 자동 운영 시스템 설치 스크립트
# 한 번 실행으로 모든 설정 완료!

set -e  # 오류 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 헤더 출력
clear
echo -e "${PURPLE}================================${NC}"
echo -e "${PURPLE}🚀 AlphaGenesis 자동 운영 설정${NC}"
echo -e "${PURPLE}================================${NC}"
echo ""
echo -e "${BLUE}📋 설치 내용:${NC}"
echo "   • 24시간 백그라운드 실행"
echo "   • 자동 코드 업데이트"
echo "   • 서비스 자동 재시작"
echo "   • 크론 작업 설정"
echo ""

# 설정 변수
PROJECT_DIR="/home/outerwoolf/AlphaGenesis"
SERVICE_NAME="alphagenesisdashboard"
USER_NAME="outerwoolf"

# 현재 디렉토리 확인
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ 오류: AlphaGenesis 디렉토리를 찾을 수 없습니다!${NC}"
    echo -e "${YELLOW}현재 위치에서 실행해주세요: $PROJECT_DIR${NC}"
    exit 1
fi

cd "$PROJECT_DIR"

echo -e "${BLUE}[1/6]${NC} 필수 패키지 설치 중..."
sudo apt update -qq
sudo apt install -y cron curl git python3-pip python3-venv

echo -e "${BLUE}[2/6]${NC} 가상환경 및 의존성 설치..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q flask flask-cors requests

# 포트 5001 방화벽 열기
echo -e "${BLUE}[2.5/6]${NC} 포트 5001 방화벽 설정..."
sudo ufw allow 5001

echo -e "${BLUE}[3/6]${NC} Systemd 서비스 설정..."
# 서비스 파일을 시스템 디렉토리로 복사
sudo cp alphagenesisdashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

echo -e "${BLUE}[4/6]${NC} 자동 업데이트 스크립트 권한 설정..."
chmod +x auto_update_dashboard.sh

echo -e "${BLUE}[5/6]${NC} 크론 작업 설정 (5분마다 업데이트 확인)..."
# 기존 크론 작업 제거
crontab -l 2>/dev/null | grep -v "auto_update_dashboard.sh" | crontab -

# 새로운 크론 작업 추가
(crontab -l 2>/dev/null; echo "*/5 * * * * cd $PROJECT_DIR && ./auto_update_dashboard.sh >> logs/cron.log 2>&1") | crontab -

echo -e "${BLUE}[6/6]${NC} 서비스 시작..."
sudo systemctl start $SERVICE_NAME

# 잠시 대기 후 상태 확인
sleep 3

echo ""
echo -e "${PURPLE}================================${NC}"
echo -e "${GREEN}🎉 설치 완료!${NC}"
echo -e "${PURPLE}================================${NC}"
echo ""

# 서비스 상태 확인
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✅ 서비스 상태: 정상 실행 중${NC}"
    echo -e "${GREEN}🌐 대시보드 접속: http://34.47.77.230:5001${NC}"
else
    echo -e "${RED}❌ 서비스 상태: 실행 실패${NC}"
    echo -e "${YELLOW}로그 확인: sudo journalctl -u $SERVICE_NAME${NC}"
fi

echo ""
echo -e "${BLUE}📊 관리 명령어:${NC}"
echo "   서비스 상태:    sudo systemctl status $SERVICE_NAME"
echo "   서비스 중지:    sudo systemctl stop $SERVICE_NAME"
echo "   서비스 시작:    sudo systemctl start $SERVICE_NAME"
echo "   서비스 재시작:  sudo systemctl restart $SERVICE_NAME"
echo "   로그 확인:      sudo journalctl -u $SERVICE_NAME -f"
echo "   수동 업데이트:  ./auto_update_dashboard.sh"
echo ""

echo -e "${BLUE}📁 로그 파일:${NC}"
echo "   자동 업데이트:  $PROJECT_DIR/logs/auto_update.log"
echo "   크론 로그:      $PROJECT_DIR/logs/cron.log"
echo "   서비스 로그:    sudo journalctl -u $SERVICE_NAME"
echo ""

echo -e "${GREEN}🔄 자동 기능:${NC}"
echo "   • 5분마다 GitHub에서 코드 업데이트 확인"
echo "   • 새 코드 발견 시 자동 다운로드 및 재시작"
echo "   • 서비스 장애 시 자동 재시작 (3초 후)"
echo "   • 백업 자동 생성 및 7일 후 정리"
echo ""

echo -e "${PURPLE}🎯 이제 로컬 PC에서 코드를 수정하고 GitHub에 push하면${NC}"
echo -e "${PURPLE}   5분 이내에 서버에 자동으로 반영됩니다!${NC}"
echo ""

# 크론 작업 확인
echo -e "${BLUE}📅 설정된 크론 작업:${NC}"
crontab -l | grep auto_update_dashboard.sh

echo ""
echo -e "${GREEN}✨ AlphaGenesis 자동 운영 시스템 설치 완료! ✨${NC}" 