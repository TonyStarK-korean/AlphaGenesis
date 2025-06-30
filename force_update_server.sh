#!/bin/bash
# force_update_server.sh - GitHub에서 웹서버로 강제 업데이트

echo "🔄 GitHub → 웹서버 강제 업데이트 시작..."
echo "================================================"

# 프로젝트 디렉토리로 이동
cd /home/outerwoolf/AlphaGenesis || {
    echo "❌ 오류: AlphaGenesis 디렉토리를 찾을 수 없습니다."
    exit 1
}

echo "📂 현재 디렉토리: $(pwd)"

# 1. 현재 실행 중인 프로세스 중지
echo "🛑 기존 프로세스 중지 중..."
pkill -f "python.*dashboard" 2>/dev/null || true
pkill -f "python.*backtest" 2>/dev/null || true
sleep 2

# 2. Git 상태 확인
echo "📊 Git 상태 확인..."
git status

# 3. 모든 로컬 변경사항 삭제
echo "🗑️ 로컬 변경사항 삭제 중..."
git reset --hard HEAD
git clean -fd

# 4. 원격에서 최신 변경사항 가져오기
echo "⬇️ 원격 저장소에서 최신 변경사항 가져오는 중..."
git fetch origin

# 5. 강제로 원격 main 브랜치로 리셋
echo "🔥 강제 업데이트 실행 중..."
git reset --hard origin/main

# 6. 추가 정리
echo "🧹 추가 정리 작업..."
git clean -fd

# 7. 업데이트 확인
echo "✅ 업데이트 완료! 최신 커밋:"
git log --oneline -5

# 8. Python 패키지 업데이트 (필요시)
echo "📦 Python 패키지 업데이트 확인 중..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt --upgrade
else
    echo "⚠️ requirements.txt 파일이 없습니다."
fi

# 9. 실행 권한 부여
echo "🔐 실행 권한 부여 중..."
chmod +x *.py 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true

# 10. 서비스 재시작
echo "🚀 서비스 재시작 중..."

# 대시보드 백그라운드 실행
nohup python3 dashboard/simple_dashboard.py > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "📊 대시보드 시작됨 (PID: $DASHBOARD_PID)"

# 잠시 대기
sleep 3

# 대시보드 연결 테스트
echo "🔍 대시보드 연결 테스트 중..."
if curl -s http://localhost:5000 > /dev/null; then
    echo "✅ 대시보드 정상 작동 중!"
    echo "🌐 접속 주소:"
    echo "   - 로컬: http://localhost:5000"
    echo "   - 외부: http://34.47.77.230:5000"
else
    echo "❌ 대시보드 연결 실패"
fi

echo "================================================"
echo "✅ 강제 업데이트 완료!"
echo "🎯 백테스트 실행: python3 run_server_backtest.py"
echo "📊 상태 확인: python3 check_dashboard_status.py"
echo "================================================" 