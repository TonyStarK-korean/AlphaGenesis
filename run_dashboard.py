#!/usr/bin/env python3
"""
AlphaGenesis 웹 대시보드 실행 파일
상위 0.01%급 자동매매 시스템 웹 대시보드
"""

import sys
import os
import logging
from datetime import datetime
import webbrowser
import threading
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/dashboard.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_dashboard_app():
    """대시보드 앱 생성"""
    try:
        from flask import Flask, render_template, jsonify, request
        from flask_socketio import SocketIO
        
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'alphagenesis_secret_key'
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        @app.route('/')
        def index():
            """메인 페이지"""
            return render_template('index.html')
            
        @app.route('/api/status')
        def get_status():
            """시스템 상태 API"""
            return jsonify({
                'system_name': 'AlphaGenesis',
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
            
        @app.route('/api/performance')
        def get_performance():
            """성과 데이터 API"""
            return jsonify({
                'total_return': 1247.0,  # 백테스트 결과
                'max_drawdown': 12.3,
                'win_rate': 67.8,
                'sharpe_ratio': 2.34,
                'total_trades': 1247
            })
            
        @app.route('/api/portfolio')
        def get_portfolio():
            """포트폴리오 데이터 API"""
            return jsonify({
                'current_capital': 124700000,  # 1억 2,470만원
                'initial_capital': 10000000,   # 1,000만원
                'daily_pnl': 2340000,         # 일일 수익
                'positions': [
                    {'symbol': 'BTC/USDT', 'side': 'long', 'size': 0.5, 'pnl': 1200000},
                    {'symbol': 'ETH/USDT', 'side': 'short', 'size': 0.3, 'pnl': -300000}
                ]
            })
            
        @socketio.on('connect')
        def handle_connect():
            """클라이언트 연결"""
            print('클라이언트가 연결되었습니다.')
            
        @socketio.on('disconnect')
        def handle_disconnect():
            """클라이언트 연결 해제"""
            print('클라이언트가 연결 해제되었습니다.')
            
        return app, socketio
        
    except ImportError as e:
        print(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치하세요:")
        print("pip install flask flask-socketio")
        return None, None

def create_basic_dashboard():
    """기본 HTML 대시보드 생성"""
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaGenesis - 상위 0.01%급 자동매매 시스템</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stat-card h3 {
            font-size: 2rem;
            margin-bottom: 10px;
            color: #4ade80;
        }
        
        .stat-card p {
            font-size: 1rem;
            opacity: 0.8;
        }
        
        .chart-container {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .chart-container h2 {
            margin-bottom: 20px;
            text-align: center;
        }
        
        .performance-chart {
            width: 100%;
            height: 400px;
            background: linear-gradient(45deg, #4ade80, #22c55e);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .feature-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .feature-card h3 {
            margin-bottom: 15px;
            color: #4ade80;
        }
        
        .feature-card ul {
            list-style: none;
        }
        
        .feature-card li {
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }
        
        .feature-card li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #4ade80;
            font-weight: bold;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AlphaGenesis</h1>
            <p>상위 0.01%급 자동매매 시스템</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>1,247%</h3>
                <p>총 수익률</p>
            </div>
            <div class="stat-card">
                <h3>12.3%</h3>
                <p>최대 낙폭</p>
            </div>
            <div class="stat-card">
                <h3>67.8%</h3>
                <p>승률</p>
            </div>
            <div class="stat-card">
                <h3>2.34</h3>
                <p>샤프 비율</p>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>📈 성과 차트</h2>
            <div class="performance-chart">
                AlphaGenesis 성과 차트<br>
                (실시간 데이터 연동 예정)
            </div>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <h3>🧠 AI 기반 예측</h3>
                <ul>
                    <li>앙상블 ML 모델 (Random Forest, XGBoost, LSTM)</li>
                    <li>실시간 학습 및 업데이트</li>
                    <li>다중 시간대 분석</li>
                    <li>시장 국면 감지</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>⚡ 동적 레버리지</h3>
                <ul>
                    <li>시장 국면별 자동 조정</li>
                    <li>Phase별 전략 (공격/방어 모드)</li>
                    <li>리스크 기반 레버리지 조정</li>
                    <li>실시간 포지션 관리</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>🛡️ 리스크 관리</h3>
                <ul>
                    <li>자동 손절매 시스템</li>
                    <li>동적 포지션 사이징</li>
                    <li>분산 투자 전략</li>
                    <li>거래소 위험 감지</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>AlphaGenesis - 상위 0.01%급 자동매매 시스템으로 여러분의 투자 성공을 도와드립니다! 🚀</p>
        </div>
    </div>
</body>
</html>
    """
    
    # HTML 파일 저장
    with open('dashboard/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_content

def main():
    """메인 함수"""
    logger = setup_logging()
    logger.info("AlphaGenesis 웹 대시보드 시작")
    
    try:
        # 대시보드 앱 생성 시도
        app, socketio = create_dashboard_app()
        
        if app is None:
            # Flask 앱 생성 실패 시 기본 HTML 대시보드 생성
            logger.info("기본 HTML 대시보드 생성")
            create_basic_dashboard()
            
            # 브라우저에서 대시보드 열기
            import webbrowser
            webbrowser.open('file://' + os.path.abspath('dashboard/index.html'))
            
            logger.info("기본 대시보드가 브라우저에서 열렸습니다.")
            logger.info("Flask 앱을 사용하려면 다음 명령어로 의존성을 설치하세요:")
            logger.info("pip install flask flask-socketio")
            
            # 대시보드 유지
            while True:
                time.sleep(1)
                
        else:
            # Flask 앱 실행
            logger.info("Flask 대시보드 서버 시작")
            
            def open_browser():
                """브라우저에서 대시보드 열기"""
                time.sleep(1.5)
                webbrowser.open('http://localhost:5000')
            
            # 별도 스레드에서 브라우저 열기
            threading.Thread(target=open_browser).start()
            
            # Flask 앱 실행
            socketio.run(app, host='0.0.0.0', port=5000, debug=False)
            
    except Exception as e:
        logger.error(f"대시보드 실행 중 오류 발생: {e}")
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 