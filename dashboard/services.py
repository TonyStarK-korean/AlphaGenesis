import os
import json
import threading
import queue
import time
from datetime import datetime

# Plotly 선택적 임포트
try:
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    class PlotlyJSONEncoder(json.JSONEncoder):
        def default(self, o):
            return super().default(o)

# 백테스트 엔진 임포트
try:
    from dashboard.backtest_engine import BacktestEngine, create_dummy_data_if_not_exists
    BACKTEST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: backtest_engine not available ({e}). Backtest features will be disabled.")
    BACKTEST_AVAILABLE = False

class DashboardManager:
    """대시보드 핵심 로직 관리자"""
    
    def __init__(self):
        # 백테스트 엔진 초기화
        if BACKTEST_AVAILABLE:
            self.backtest_engine = BacktestEngine()
            create_dummy_data_if_not_exists()
        else:
            self.backtest_engine = None
        
        self.is_backtest_running = False
        self.backtest_stop_flag = None
        self.backtest_log_queue = queue.Queue()
        self.backtest_cache = {}
        
        # 결과 파일 경로 설정
        self._results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(self._results_dir, exist_ok=True)
        self._results_file = os.path.join(self._results_dir, 'latest_backtest_results.json')

    def get_data_info(self):
        """데이터 정보 조회"""
        if not self.backtest_engine:
            return {'error': '백테스트 엔진을 사용할 수 없습니다.'}
        
        symbols = self.backtest_engine.get_available_symbols()
        default_symbol = 'BTC_USDT'
        data_info = self.backtest_engine.get_data_info(default_symbol)
        
        if data_info:
            return {
                'symbols': symbols,
                'start_date': data_info['start_date'],
                'end_date': data_info['end_date'],
            }
        return {
            'symbols': symbols,
            'start_date': '2023-01-01',
            'end_date': datetime.now().strftime('%Y-%m-%d'),
        }

    def download_data(self, symbol, start_date, end_date):
        """데이터 다운로드"""
        if not self.backtest_engine:
            return {'error': '백테스트 엔진을 사용할 수 없습니다.'}
        return self.backtest_engine.download_data(symbol, start_date, end_date, save_csv=True)

    def get_strategies(self):
        """전략 목록 조회"""
        if not self.backtest_engine:
            return {'error': '백테스트 엔진을 사용할 수 없습니다.'}
        return self.backtest_engine.get_strategies()

    def start_backtest(self, config):
        """백테스트 비동기 실행"""
        if self.is_backtest_running:
            return {'error': '이미 백테스트가 실행 중입니다.'}
        
        self.is_backtest_running = True
        self.backtest_log_queue = queue.Queue()
        self.backtest_stop_flag = threading.Event()

        def run_and_stream():
            try:
                def sse_progress_callback(msg):
                    self.backtest_log_queue.put(json.dumps(msg))

                result = self.backtest_engine.run_backtest(
                    config, 
                    progress_callback=sse_progress_callback, 
                    stop_flag=self.backtest_stop_flag
                )
                
                if result.get('success'):
                    # 결과를 파일에 저장하여 대시보드에 반영
                    with open(self._results_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, cls=PlotlyJSONEncoder)
                    self.backtest_log_queue.put(json.dumps({'type': 'log', 'message': f"[INFO] 백테스트 결과 파일 저장 완료: {self._results_file}"}))
                else:
                    self.backtest_log_queue.put(json.dumps({'type': 'error', 'message': result.get('error', '알 수 없는 오류')}))

            except Exception as e:
                self.backtest_log_queue.put(json.dumps({'type': 'error', 'message': f"백테스트 실행 중 심각한 오류: {str(e)}"}))
            finally:
                self.is_backtest_running = False
                self.backtest_log_queue.put("[END]")

        threading.Thread(target=run_and_stream, daemon=True).start()
        return {'status': 'started'}

    def stop_backtest(self):
        """백테스트 중지"""
        if self.backtest_stop_flag:
            self.backtest_stop_flag.set()
        self.is_backtest_running = False
        return {'status': 'stopped'}

    def get_latest_results(self):
        """최신 백테스트 결과 조회"""
        if not os.path.exists(self._results_file):
            return {'error': '백테스트 결과 파일이 없습니다.'}
        try:
            with open(self._results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {'error': f'결과 파일 로드 오류: {str(e)}'}

# 서비스 인스턴스 생성
dashboard_manager = DashboardManager()
