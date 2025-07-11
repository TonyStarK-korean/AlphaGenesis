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
from dashboard.routes import api as api_blueprint
app.register_blueprint(api_blueprint)

if __name__ == '__main__':
    print("🚀 AlphaGenesis 대시보드 서버 시작")
    print("📊 대시보드 접속 주소:")
    print("   로컬: http://127.0.0.1:9000")
    print("   GVS 서버: http://34.47.77.230:9000")
    print("⚡ 시스템이 24시간 운영됩니다...")
    
    # Flask 서버 실행 (운영 환경에서는 gunicorn/uwsgi 사용 권장)
    app.run(
        host='0.0.0.0',
        port=9000,
        debug=False,    # 운영 환경에서는 False로 설정
        threaded=True
    )
