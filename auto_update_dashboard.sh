#!/bin/bash

# AlphaGenesis 자동 업데이트 스크립트
# 작성자: AlphaGenesis Team
# 설명: GitHub에서 최신 코드를 자동으로 받아서 대시보드를 업데이트

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 설정
PROJECT_DIR="/home/outerwoolf/AlphaGenesis"
SERVICE_NAME="alphagenesisdashboard"
LOG_FILE="/home/outerwoolf/AlphaGenesis/logs/auto_update.log"

# 로그 디렉토리 생성
mkdir -p $(dirname "$LOG_FILE")

# 로그 함수 (파일에도 저장)
log_to_file() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_info "🚀 AlphaGenesis 자동 업데이트 시작"
log_to_file "자동 업데이트 시작"

# 프로젝트 디렉토리로 이동
cd "$PROJECT_DIR" || {
    log_error "프로젝트 디렉토리로 이동 실패: $PROJECT_DIR"
    exit 1
}

# 현재 커밋 해시 저장
CURRENT_COMMIT=$(git rev-parse HEAD)
log_info "현재 커밋: $CURRENT_COMMIT"

# Git 업데이트 시도
log_info "GitHub에서 최신 코드 확인 중..."
git fetch origin main

# 최신 커밋 해시 확인
LATEST_COMMIT=$(git rev-parse origin/main)
log_info "최신 커밋: $LATEST_COMMIT"

# 업데이트가 필요한지 확인
if [ "$CURRENT_COMMIT" = "$LATEST_COMMIT" ]; then
    log_info "✅ 코드가 최신 상태입니다. 업데이트 불필요."
    log_to_file "업데이트 불필요 - 최신 상태"
    exit 0
fi

log_warning "📥 새로운 업데이트 발견! 업데이트를 진행합니다..."
log_to_file "새로운 업데이트 발견: $CURRENT_COMMIT -> $LATEST_COMMIT"

# 백업 생성
BACKUP_DIR="/home/outerwoolf/AlphaGenesis_backup_$(date +%Y%m%d_%H%M%S)"
log_info "백업 생성 중: $BACKUP_DIR"
cp -r "$PROJECT_DIR" "$BACKUP_DIR"

# Git pull 실행
log_info "GitHub에서 최신 코드 다운로드 중..."
if git pull origin main; then
    log_success "✅ 코드 업데이트 성공!"
    log_to_file "코드 업데이트 성공"
else
    log_error "❌ 코드 업데이트 실패!"
    log_to_file "코드 업데이트 실패"
    
    # 백업에서 복원
    log_warning "백업에서 복원 중..."
    rm -rf "$PROJECT_DIR"
    mv "$BACKUP_DIR" "$PROJECT_DIR"
    log_warning "백업 복원 완료"
    exit 1
fi

# 가상환경 활성화 및 의존성 업데이트
log_info "Python 의존성 업데이트 중..."
if [ -f "requirements.txt" ]; then
    source venv/bin/activate
    pip install -r requirements.txt --quiet
    log_success "의존성 업데이트 완료"
else
    log_warning "requirements.txt 파일이 없습니다."
fi

# 서비스 재시작
log_info "대시보드 서비스 재시작 중..."
if systemctl is-active --quiet "$SERVICE_NAME"; then
    sudo systemctl restart "$SERVICE_NAME"
    log_success "✅ 서비스 재시작 완료!"
    log_to_file "서비스 재시작 성공"
else
    log_warning "서비스가 실행되지 않았습니다. 시작 시도..."
    sudo systemctl start "$SERVICE_NAME"
    log_success "서비스 시작 완료!"
fi

# 서비스 상태 확인
sleep 3
if systemctl is-active --quiet "$SERVICE_NAME"; then
    log_success "🎉 AlphaGenesis 대시보드 업데이트 및 재시작 완료!"
    log_info "🌐 접속 주소: http://34.47.77.230:5001"
    log_to_file "업데이트 완료 및 서비스 정상 실행"
else
    log_error "❌ 서비스 시작 실패!"
    log_to_file "서비스 시작 실패"
    
    # 로그 확인
    log_error "서비스 로그:"
    sudo journalctl -u "$SERVICE_NAME" --no-pager -n 10
fi

# 오래된 백업 정리 (7일 이상)
log_info "오래된 백업 정리 중..."
find /home/outerwoolf -name "AlphaGenesis_backup_*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true

log_success "🏁 자동 업데이트 스크립트 완료!"
log_to_file "자동 업데이트 스크립트 완료" 