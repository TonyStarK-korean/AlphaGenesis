from flask import Blueprint, jsonify, request, Response, render_template, redirect
from .services import dashboard_manager
import json
import queue
from datetime import datetime

# Flask Blueprint 생성
api = Blueprint('api', __name__)

@api.route('/')
def main_dashboard():
    """메인 대시보드 페이지 렌더링"""
    return render_template('main_dashboard.html')

@api.route('/backtest')
def backtest_dashboard():
    """백테스트 대시보드 페이지 렌더링"""
    return render_template('backtest_dashboard.html')

@api.route('/live-trading')
def live_trading():
    """실전매매 대시보드 페이지 렌더링"""
    return render_template('live_trading.html')

@api.route('/api/data/info', methods=['GET'])
def get_data_info():
    """데이터 정보 반환 API"""
    info = dashboard_manager.get_data_info()
    if 'error' in info:
        return jsonify(info), 500
    return jsonify(info)

@api.route('/api/data/download', methods=['GET'])
def download_data():
    """데이터 다운로드 API (BTC_USDT 고정)"""
    symbol = 'BTC_USDT'
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    if not start_date or not end_date:
        return jsonify({'error': '시작일과 종료일을 지정해주세요.'}), 400
    
    result = dashboard_manager.download_data(symbol, start_date, end_date)
    if 'error' in result:
        return jsonify(result), 500
    return jsonify(result)

@api.route('/api/strategies', methods=['GET'])
def get_strategies():
    """전략 목록 조회 API"""
    strategies = dashboard_manager.get_strategies()
    if 'error' in strategies:
        return jsonify(strategies), 500
    return jsonify({'strategies': strategies})

@api.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """백테스트 실행 API"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '요청 데이터가 없습니다.'}), 400
    
    result = dashboard_manager.start_backtest(data)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

@api.route('/api/backtest/stop', methods=['POST'])
def stop_backtest():
    """백테스트 중지 API"""
    result = dashboard_manager.stop_backtest()
    return jsonify(result)

@api.route('/api/backtest/stream_log')
def stream_backtest_log():
    """백테스트 로그 실시간 스트리밍 (SSE)"""
    def event_stream():
        q = dashboard_manager.backtest_log_queue
        while True:
            try:
                log = q.get(timeout=1)
                if log == "[END]":
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    break
                yield f"data: {log}\n\n"
            except queue.Empty:
                if not dashboard_manager.is_backtest_running:
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    break
                # 연결 유지를 위한 ping
                yield ": ping\n\n"
                
    return Response(event_stream(), mimetype="text/event-stream")

@api.route('/api/backtest/results')
def get_backtest_results():
    """최신 백테스트 결과 조회 API"""
    results = dashboard_manager.get_latest_results()
    if 'error' in results:
        return jsonify(results), 404
    return jsonify(results)

# 실전매매 API
@api.route('/api/live/start', methods=['POST'])
def start_live_trading():
    """실전매매 시작 API"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '요청 데이터가 없습니다.'}), 400
    
    # 실전매매 시작 로직 (추후 구현)
    return jsonify({'status': 'success', 'message': '실전매매가 시작되었습니다.'})

@api.route('/api/live/stop', methods=['POST'])
def stop_live_trading():
    """실전매매 중지 API"""
    # 실전매매 중지 로직 (추후 구현)
    return jsonify({'status': 'success', 'message': '실전매매가 중지되었습니다.'})

@api.route('/api/live/status', methods=['GET'])
def get_live_trading_status():
    """실전매매 상태 조회 API"""
    # 더미 데이터 반환
    status = {
        'status': 'running',
        'start_time': '2025-01-11T10:00:00',
        'current_capital': 10500000,
        'total_pnl': 500000,
        'active_positions': 3,
        'total_trades': 15,
        'winning_trades': 9,
        'losing_trades': 6,
        'win_rate': 0.6,
        'daily_return': 0.05
    }
    return jsonify(status)

@api.route('/api/live/positions', methods=['GET'])
def get_live_positions():
    """활성 포지션 조회 API"""
    # 더미 데이터 반환
    positions = [
        {
            'symbol': 'BTC/USDT',
            'side': 'LONG',
            'size': 0.1,
            'entry_price': 43000,
            'current_price': 43500,
            'unrealized_pnl': 50,
            'pnl_percentage': 1.16
        },
        {
            'symbol': 'ETH/USDT',
            'side': 'SHORT',
            'size': 2.0,
            'entry_price': 2600,
            'current_price': 2580,
            'unrealized_pnl': 40,
            'pnl_percentage': 0.77
        }
    ]
    return jsonify({'positions': positions})

@api.route('/api/live/settings', methods=['GET', 'POST'])
def live_trading_settings():
    """실전매매 설정 API"""
    if request.method == 'GET':
        settings = {
            'initial_capital': 10000000,
            'max_position': 10,
            'leverage': 2,
            'stop_loss': 2,
            'take_profit': 5,
            'daily_max_loss': 5,
            'strategy': 'triple_combo',
            'confidence_threshold': 0.6
        }
        return jsonify(settings)
    
    elif request.method == 'POST':
        data = request.get_json()
        # 설정 업데이트 로직 (추후 구현)
        return jsonify({'status': 'success', 'message': '설정이 업데이트되었습니다.'})

@api.route('/api/market/overview', methods=['GET'])
def get_market_overview():
    """시장 개요 조회 API"""
    # 더미 데이터 반환
    overview = {
        'timestamp': '2025-01-11T14:30:00',
        'major_symbols': {
            'BTC/USDT': {'price': 43250, 'change': 2.34},
            'ETH/USDT': {'price': 2580, 'change': 1.87},
            'BNB/USDT': {'price': 315, 'change': -0.45}
        },
        'market_sentiment': 'bullish',
        'total_volume': 25000000000,
        'trending_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    }
    return jsonify(overview)

@api.route('/api/binance/symbols', methods=['GET'])
def get_binance_symbols():
    """바이낸스 USDT.P 심볼 조회 API"""
    # 더미 데이터 반환
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT',
        'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT',
        'LTC/USDT', 'BCH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT'
    ]
    return jsonify({'symbols': symbols})

@api.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인 API"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
