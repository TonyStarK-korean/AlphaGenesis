"""
AlphaGenesis Dashboard Module
웹 대시보드 및 API 서비스
"""

# Flask 관련 import
from flask import Flask
import os

# 코어 모듈 import (순환 참조 방지)
try:
    import sys
    import os
    # 프로젝트 루트 경로 추가
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from core import config, initialize_core
except ImportError:
    # 최소 기능으로 대체
    config = None
    initialize_core = None

# 버전 정보
__version__ = "3.0.0"
__app_name__ = "AlphaGenesis Dashboard"

def create_app(config_name='development'):
    """Flask 앱 팩토리 함수"""
    app = Flask(__name__)
    
    # 설정 로드 (config가 None인 경우 기본값 사용)
    if config:
        dashboard_config = config.get_config('dashboard')
        app.config.update(dashboard_config)
    else:
        # 기본 설정
        app.config.update({
            'DEBUG': True,
            'TESTING': False
        })
    
    # 보안 설정
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # 블루프린트 등록
    from .routes import api
    app.register_blueprint(api)
    
    # 에러 핸들러
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not Found", "status": 404}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal Server Error", "status": 500}, 500
    
    return app

def get_dashboard_info():
    """대시보드 정보 반환"""
    return {
        'name': __app_name__,
        'version': __version__,
        'endpoints': [
            '/',
            '/backtest',
            '/live-trading',
            '/api/health',
            '/api/strategies',
            '/api/backtest/run'
        ]
    }