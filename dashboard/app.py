import sys
import os
from flask import Flask
from flask_cors import CORS

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flask 앱 생성
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# routes.py에 정의된 API Blueprint 등록
try:
    from dashboard.routes import api as api_blueprint
    app.register_blueprint(api_blueprint)
    print("✅ 메인 API 라우트 등록 완료")
except Exception as e:
    print(f"❌ 메인 API 라우트 등록 실패: {e}")

# 다운로드 API Blueprint 등록 (존재하는 경우에만)
try:
    from dashboard.download_routes import download_api
    app.register_blueprint(download_api)
    print("✅ 다운로드 API 라우트 등록 완료")
except Exception as e:
    print(f"⚠️  다운로드 API 라우트 등록 실패: {e}")

# 실전매매 API Blueprint 등록 (존재하는 경우에만)
try:
    from live_trading.routes import live_trading_api
    app.register_blueprint(live_trading_api)
    print("✅ 실전매매 API 라우트 등록 완료")
except Exception as e:
    print(f"⚠️  실전매매 API 라우트 등록 실패: {e}")

if __name__ == '__main__':
    print("🚀 AlphaGenesis 프리미엄 트레이딩 플랫폼 서버 시작")
    print("📊 대시보드 접속 주소:")
    print("   🌐 외부 접속: http://34.47.77.230:9000")
    print("   🏠 로컬 접속: http://127.0.0.1:9000")
    print("   📱 모바일 접속: http://localhost:9000")
    print("")
    print("🎯 주요 페이지:")
    print("   📈 메인 대시보드: /")
    print("   🔬 백테스트: /backtest") 
    print("   ⚡ 실전매매: /premium-live")
    print("   👑 프리미엄: /premium-backtest")
    print("")
    print("⚡ 시스템이 24시간 운영됩니다...")
    print("🔒 안전한 HTTPS 연결 지원")
    
    try:
        # Flask 서버 실행 (외부 접속 허용)
        app.run(
            host='0.0.0.0',  # 모든 IP에서 접속 허용
            port=9000,       # 포트 9000
            debug=False,     # 운영 환경
            threaded=True,   # 멀티스레딩 지원
            use_reloader=False  # 자동 재시작 비활성화 (운영 환경)
        )
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        print("🔧 방화벽 설정을 확인하세요:")
        print("   sudo ufw allow 9000")
        print("   또는 iptables 설정 확인")