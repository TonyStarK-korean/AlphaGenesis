import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
import glob
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder

# 시스템 모듈 임포트
from config.backtest_config import backtest_config
from data.market_data_downloader import MarketDataDownloader
from core.trading_engine.adaptive_phase_manager import AdaptivePhaseManager
from core.trading_engine.compound_trading_engine import CompoundTradingEngine, CompoundMode

app = Flask(__name__)
CORS(app)  # 외부 접속 허용
app.config['SECRET_KEY'] = 'your-secret-key-here'

RESULTS_DIR = 'dashboard/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

class DashboardManager:
    """대시보드 관리자"""
    
    def __init__(self):
        self.data_downloader = MarketDataDownloader()
        self.phase_manager = AdaptivePhaseManager()
        self.trading_engine = CompoundTradingEngine()
        
        # 실시간 데이터
        self.real_time_data = {
            'current_capital': backtest_config.initial_capital,
            'total_return': 0.0,
            'daily_pnl': 0.0,
            'open_positions': 0,
            'current_phase': 'PHASE1_AGGRESSIVE',
            'market_condition': 'SIDEWAYS',
            'active_exchanges': 1,
            'last_update': datetime.now()
        }
        
        # 백테스트 결과 캐시
        self.backtest_cache = {}
        
        # 실시간 모니터링 스레드
        self.monitoring_thread = None
        self.is_monitoring = False
        
    def start_monitoring(self):
        """실시간 모니터링 시작"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """실시간 모니터링 중지"""
        self.is_monitoring = False
        
    def _monitoring_loop(self):
        """실시간 모니터링 루프"""
        while self.is_monitoring:
            try:
                # 실시간 데이터 업데이트
                self._update_real_time_data()
                time.sleep(5)  # 5초마다 업데이트
                
            except Exception as e:
                print(f"모니터링 오류: {str(e)}")
                time.sleep(10)
                
    def _update_real_time_data(self):
        """실시간 데이터 업데이트"""
        # 시뮬레이션된 실시간 데이터
        self.real_time_data['current_capital'] += np.random.normal(0, 1000000)
        self.real_time_data['total_return'] = (self.real_time_data['current_capital'] - backtest_config.initial_capital) / backtest_config.initial_capital * 100
        self.real_time_data['daily_pnl'] = np.random.normal(500000, 200000)
        self.real_time_data['open_positions'] = np.random.randint(3, 8)
        self.real_time_data['last_update'] = datetime.now()
        
        # Phase 전환 시뮬레이션
        if np.random.random() < 0.01:
            self.real_time_data['current_phase'] = 'PHASE2_DEFENSIVE' if self.real_time_data['current_phase'] == 'PHASE1_AGGRESSIVE' else 'PHASE1_AGGRESSIVE'
            
        # 시장 국면 시뮬레이션
        market_conditions = ['BULL_MARKET', 'BEAR_MARKET', 'SIDEWAYS', 'HIGH_VOLATILITY', 'LOW_VOLATILITY']
        if np.random.random() < 0.005:
            self.real_time_data['market_condition'] = np.random.choice(market_conditions)

# 대시보드 관리자 인스턴스
dashboard_manager = DashboardManager()

@app.route('/')
def index():
    """메인 대시보드"""
    return render_template('index.html')

@app.route('/api/config')
def get_config():
    """설정 정보 API"""
    try:
        config_summary = backtest_config.get_config_summary()
        return jsonify(config_summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-config', methods=['POST'])
def update_config():
    """설정 업데이트 API"""
    try:
        data = request.json
        
        if 'date_range' in data:
            backtest_config.update_date_range(
                data['date_range']['start'],
                data['date_range']['end']
            )
            
        if 'phase_settings' in data:
            for phase, settings in data['phase_settings'].items():
                backtest_config.update_phase_settings(phase, settings)
                
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-data', methods=['POST'])
def download_data():
    """데이터 다운로드 API"""
    try:
        data = request.json
        symbols = data.get('symbols', backtest_config.data_download['symbols'])
        
        # 데이터 다운로드 실행
        all_data = dashboard_manager.data_downloader.download_all_data()
        
        # 데이터 요약 반환
        summary = dashboard_manager.data_downloader.get_data_summary()
        
        return jsonify({
            'status': 'success',
            'downloaded_symbols': len(all_data),
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """백테스트 실행 API"""
    try:
        data = request.json
        
        # 설정 업데이트
        if 'config' in data:
            config_data = data['config']
            if 'date_range' in config_data:
                backtest_config.update_date_range(
                    config_data['date_range']['start'],
                    config_data['date_range']['end']
                )
                
        # 백테스트 실행
        results = dashboard_manager.trading_engine.run_backtest(
            days=(backtest_config.end_date - backtest_config.start_date).days,
            trades_per_day=5
        )
        
        # 결과 캐시
        cache_key = f"{backtest_config.start_date.strftime('%Y%m%d')}_{backtest_config.end_date.strftime('%Y%m%d')}"
        dashboard_manager.backtest_cache[cache_key] = results
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/phase-analysis')
def get_phase_analysis():
    """Phase 분석 API"""
    try:
        # Phase 상태
        phase_status = dashboard_manager.phase_manager.get_phase_status()
        
        # Phase 전환 기록
        phase_history = dashboard_manager.phase_manager.get_phase_history()
        
        # 시장 국면 기록
        market_history = dashboard_manager.phase_manager.get_market_condition_history()
        
        return jsonify({
            'phase_status': phase_status,
            'phase_history': phase_history,
            'market_history': market_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-summary')
def get_data_summary():
    """데이터 요약 API"""
    try:
        summary = dashboard_manager.data_downloader.get_data_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """성과 데이터 API"""
    try:
        # 복리 효과 비교
        compound_comparison = dashboard_manager.trading_engine.get_performance_comparison()
        
        # 실패 분석
        failure_analysis = dashboard_manager.trading_engine.get_failure_analysis()
        
        # 실시간 데이터
        real_time_data = dashboard_manager.real_time_data
        
        return jsonify({
            'compound_comparison': compound_comparison,
            'failure_analysis': failure_analysis,
            'real_time_data': real_time_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def get_system_status():
    """시스템 상태 API"""
    try:
        return jsonify({
            'status': 'RUNNING',
            'uptime': '24h 15m 30s',
            'last_backup': '2024-01-15 14:30:00',
            'active_connections': 5,
            'memory_usage': '45%',
            'cpu_usage': '23%',
            'disk_usage': '67%',
            'alerts': [
                {'level': 'INFO', 'message': '시스템 정상 운영 중'},
                {'level': 'WARNING', 'message': '거래소1 API 응답 지연'},
                {'level': 'INFO', 'message': '백업 완료'}
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """실시간 모니터링 시작"""
    try:
        dashboard_manager.start_monitoring()
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """실시간 모니터링 중지"""
    try:
        dashboard_manager.stop_monitoring()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/real-time-data')
def get_real_time_data():
    """실시간 데이터 API"""
    try:
        return jsonify(dashboard_manager.real_time_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime_log', methods=['POST'])
def receive_realtime_log():
    """실시간 로그 수신 API"""
    try:
        data = request.get_json()
        log_msg = data.get('log', '')
        print(f"[실시간 로그] {log_msg}")
        return jsonify({'status': 'received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['POST'])
def receive_report():
    """백테스트 리포트 수신 API"""
    try:
        data = request.get_json()
        print(f"[백테스트 리포트] {data}")
        return jsonify({'status': 'received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_results', methods=['POST'])
def upload_results():
    data = request.get_json()
    symbol = data.get('symbol', 'unknown')
    out_path = os.path.join(RESULTS_DIR, f"results_{symbol.replace('/', '_')}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({'status': 'ok', 'file': out_path})

@app.route('/dashboard')
def dashboard():
    files = glob.glob(os.path.join(RESULTS_DIR, 'results_*.json'))
    all_results = []
    for file in files:
        with open(file, encoding='utf-8') as f:
            res = json.load(f)
            symbol = os.path.basename(file).replace('results_', '').replace('.json', '').replace('_', '/')
            res['symbol'] = symbol
            all_results.append(res)
    df = pd.DataFrame(all_results)
    # 표와 그래프 생성
    table_html = df.to_html(classes='table table-striped', index=False)
    # 예시: 최종 자본 그래프
    fig = go.Figure()
    for _, row in df.iterrows():
        if 'capital' in row and isinstance(row['capital'], list):
            fig.add_trace(go.Scatter(y=row['capital'], name=row['symbol']))
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template('dashboard.html', table_html=table_html, graph_json=graph_json)

# === Flask 서버 24시간 가동 안내 ===
# 운영 시 아래 명령어로 백그라운드에서 실행하세요:
# tmux new -s dashboard
# python3 dashboard/app.py
# (Ctrl+B, D로 세션 분리)
# 또는
# nohup python3 dashboard/app.py > dashboard.log 2>&1 &

if __name__ == '__main__':
    # 외부 접속을 위해 host를 0.0.0.0으로 설정
    app.run(debug=True, host='0.0.0.0', port=5000) 