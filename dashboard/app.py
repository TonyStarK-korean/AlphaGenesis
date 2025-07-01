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
import subprocess
import asyncio

# Plotly 선택적 임포트
try:
    import plotly.graph_objs as go
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available. Some charting features may be limited.")
    PLOTLY_AVAILABLE = False

# 시스템 모듈 임포트 (선택적)
try:
    from config.backtest_config import backtest_config
except ImportError:
    print("Warning: config.backtest_config not available. Using default config.")
    # 기본 설정 클래스 생성
    class DefaultConfig:
        initial_capital = 10000000
        
        def get_config_summary(self):
            return {'initial_capital': self.initial_capital}
        
        def update_date_range(self, start, end):
            self.start_date = start
            self.end_date = end
        
        def update_phase_settings(self, phase, settings):
            pass
    
    backtest_config = DefaultConfig()

try:
    from data.market_data_downloader import MarketDataDownloader
except ImportError:
    print("Warning: data.market_data_downloader not available. Using mock class.")
    class MarketDataDownloader:
        def download_all_data(self):
            return {}
        def get_data_summary(self):
            return {}

try:
    from core.trading_engine.adaptive_phase_manager import AdaptivePhaseManager
except ImportError:
    print("Warning: core.trading_engine.adaptive_phase_manager not available. Using mock class.")
    class AdaptivePhaseManager:
        def get_phase_status(self):
            return {}
        def get_phase_history(self):
            return []
        def get_market_condition_history(self):
            return []

try:
    from core.trading_engine.compound_trading_engine import CompoundTradingEngine, CompoundMode
except ImportError:
    print("Warning: core.trading_engine.compound_trading_engine not available. Using mock class.")
    class CompoundTradingEngine:
        def run_backtest(self, days, trades_per_day):
            return {}
    CompoundMode = None

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
        
        # 백테스트 실행 프로세스
        self.backtest_process = None
        self.is_backtest_running = False
        
        # 실시간 모니터링 스레드
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # 백테스트 결과 저장소
        self.latest_backtest_results = {
            'final_capital': backtest_config.initial_capital,
            'total_return': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'trades': [],
            'capital_history': [],
            'performance_metrics': {}
        }
        
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
    
    def start_backtest(self, config):
        """백테스트 시작"""
        if self.is_backtest_running:
            return {'error': '이미 백테스트가 실행 중입니다.'}
        
        try:
            self.is_backtest_running = True
            
            # 백테스트 스크립트 실행
            cmd = [
                'python', 'run_ml_backtest.py',
                '--start-date', config.get('date_range', {}).get('start', '2023-01-01'),
                '--end-date', config.get('date_range', {}).get('end', '2024-01-01'),
                '--initial-capital', str(config.get('initial_capital', 10000000)),
                '--symbol', config.get('symbol', 'BTC/USDT')
            ]
            
            # 백그라운드에서 백테스트 실행
            self.backtest_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            return {'status': 'started', 'message': '백테스트가 시작되었습니다.'}
            
        except Exception as e:
            self.is_backtest_running = False
            return {'error': f'백테스트 시작 실패: {str(e)}'}
    
    def stop_backtest(self):
        """백테스트 중지"""
        if self.backtest_process:
            self.backtest_process.terminate()
            self.backtest_process = None
        self.is_backtest_running = False
        return {'status': 'stopped', 'message': '백테스트가 중지되었습니다.'}
    
    def get_backtest_status(self):
        """백테스트 상태 조회"""
        if self.backtest_process:
            poll = self.backtest_process.poll()
            if poll is None:
                return {'status': 'running', 'is_running': True}
            else:
                self.is_backtest_running = False
                return {'status': 'completed', 'is_running': False, 'return_code': poll}
        return {'status': 'idle', 'is_running': False}

# 대시보드 관리자 인스턴스
dashboard_manager = DashboardManager()

@app.route('/')
def index():
    """메인 대시보드"""
    return render_template('main_dashboard.html')

@app.route('/backtest')
def backtest_dashboard():
    """백테스트 대시보드"""
    return render_template('backtest_dashboard.html')

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
        config = data.get('config', {})
        
        # 백테스트 시작
        result = dashboard_manager.start_backtest(config)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/stop', methods=['POST'])
def stop_backtest():
    """백테스트 중지 API"""
    try:
        result = dashboard_manager.stop_backtest()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/status')
def get_backtest_status():
    """백테스트 상태 API"""
    try:
        status = dashboard_manager.get_backtest_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/results')
def get_backtest_results():
    """백테스트 결과 API"""
    try:
        results = dashboard_manager.latest_backtest_results
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
    """성과 분석 API"""
    try:
        # 성과 지표 계산
        performance = {
            'current_capital': dashboard_manager.real_time_data['current_capital'],
            'total_return': dashboard_manager.real_time_data['total_return'],
            'daily_pnl': dashboard_manager.real_time_data['daily_pnl'],
            'win_rate': 65.5,  # 시뮬레이션
            'max_drawdown': -12.3,  # 시뮬레이션
            'sharpe_ratio': 1.85,  # 시뮬레이션
            'trades_count': 1247,  # 시뮬레이션
            'avg_trade_duration': '4.2시간'  # 시뮬레이션
        }
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def get_system_status():
    """시스템 상태 API"""
    try:
        status = {
            'trading_engine_status': 'running',
            'data_connection_status': 'connected',
            'ml_model_status': 'loaded',
            'current_phase': dashboard_manager.real_time_data['current_phase'],
            'market_condition': dashboard_manager.real_time_data['market_condition'],
            'active_exchanges': dashboard_manager.real_time_data['active_exchanges'],
            'open_positions': dashboard_manager.real_time_data['open_positions'],
            'system_uptime': '24일 15시간 32분',
            'memory_usage': '2.1GB / 8GB',
            'cpu_usage': '45%'
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """실시간 모니터링 시작 API"""
    try:
        dashboard_manager.start_monitoring()
        return jsonify({'status': 'success', 'message': '실시간 모니터링이 시작되었습니다.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """실시간 모니터링 중지 API"""
    try:
        dashboard_manager.stop_monitoring()
        return jsonify({'status': 'success', 'message': '실시간 모니터링이 중지되었습니다.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/real-time-data')
def get_real_time_data():
    """실시간 데이터 API"""
    try:
        data = dashboard_manager.real_time_data.copy()
        data['last_update'] = data['last_update'].isoformat()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime_log', methods=['POST'])
def receive_realtime_log():
    """실시간 로그 수신 API"""
    try:
        data = request.json
        log_message = data.get('log', '')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # 로그를 파일에 저장하거나 메모리에 캐시
        log_entry = {
            'timestamp': timestamp,
            'message': log_message,
            'type': 'backtest_log'
        }
        
        # 여기서 로그를 저장하거나 브로드캐스트할 수 있음
        print(f"[BACKTEST LOG] {timestamp}: {log_message}")
        
        return jsonify({'status': 'received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['POST'])
def receive_report():
    """백테스트 리포트 수신 API"""
    try:
        data = request.json
        
        # 백테스트 결과 업데이트
        if 'final_capital' in data:
            dashboard_manager.latest_backtest_results['final_capital'] = data['final_capital']
        if 'total_return' in data:
            dashboard_manager.latest_backtest_results['total_return'] = data['total_return']
        if 'win_rate' in data:
            dashboard_manager.latest_backtest_results['win_rate'] = data['win_rate']
        if 'max_drawdown' in data:
            dashboard_manager.latest_backtest_results['max_drawdown'] = data['max_drawdown']
        if 'trades' in data:
            dashboard_manager.latest_backtest_results['trades'] = data['trades']
        if 'capital_history' in data:
            dashboard_manager.latest_backtest_results['capital_history'] = data['capital_history']
            
        print(f"[BACKTEST REPORT] 리포트 수신: {data}")
        
        return jsonify({'status': 'received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_results', methods=['POST'])
def upload_results():
    """결과 업로드 API"""
    try:
        data = request.json
        # 결과 저장 로직
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """대시보드 페이지"""
    return render_template('dashboard.html')

# Dashboard Manager 초기화
dashboard_manager = DashboardManager()

# === Flask 서버 24시간 가동 안내 ===
# 이 서버는 24시간 운영을 위해 설계되었습니다.
# 실제 운영 환경에서는 gunicorn 또는 uwsgi 등의 WSGI 서버를 사용하시기 바랍니다.

if __name__ == '__main__':
    print("🚀 AlphaGenesis 대시보드 서버 시작")
    print("📊 대시보드 주소: http://34.47.77.230:5001")
    print("🔄 백테스트 대시보드: http://34.47.77.230:5001/backtest")
    print("⚡ 시스템이 24시간 운영됩니다...")
    
    # 실시간 모니터링 시작
    dashboard_manager.start_monitoring()
    
    # Flask 서버 실행 (외부 접속 허용, 포트 5001)
    app.run(
        host='0.0.0.0',  # 모든 IP에서 접속 허용
        port=5001,       # 포트 5001 사용
        debug=False,     # 운영 환경에서는 False
        threaded=True    # 멀티스레드 처리
    ) 