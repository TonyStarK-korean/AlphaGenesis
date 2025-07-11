from flask import Blueprint, jsonify, request, Response, render_template, redirect
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

@api.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인 API"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@api.route('/api/status', methods=['GET'])
def api_status():
    """API 상태 확인"""
    return jsonify({
        "status": "running",
        "service": "AlphaGenesis",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

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