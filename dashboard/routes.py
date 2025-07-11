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
    try:
        # 실제 바이낸스 USDT.P 심볼 목록 (주요 심볼들)
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT',
            'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT',
            'LTC/USDT', 'BCH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT',
            'ATOM/USDT', 'FTM/USDT', 'NEAR/USDT', 'ALGO/USDT', 'VET/USDT',
            'FIL/USDT', 'THETA/USDT', 'TRX/USDT', 'EOS/USDT', 'XLM/USDT',
            'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'ENJ/USDT', 'CHZ/USDT',
            'GALA/USDT', 'APE/USDT', 'GMT/USDT', 'KSM/USDT', 'FLOW/USDT',
            'ICP/USDT', 'EGLD/USDT', 'XTZ/USDT', 'COMP/USDT', 'SUSHI/USDT',
            'YFI/USDT', 'BAT/USDT', 'ZRX/USDT', 'CRV/USDT', 'KAVA/USDT',
            'WAVES/USDT', 'ZIL/USDT', 'REN/USDT', 'STORJ/USDT', 'QTUM/USDT'
        ]
        
        # 실제 환경에서는 ccxt 라이브러리를 사용해서 동적으로 가져옴
        # import ccxt
        # exchange = ccxt.binance()
        # markets = exchange.load_markets()
        # symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
        
        return jsonify({'symbols': symbols})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/strategies', methods=['GET'])
def get_strategies():
    """전략 목록 조회 API"""
    try:
        strategies = [
            {
                'id': 'triple_combo',
                'name': '트리플 콤보 전략',
                'description': 'RSI, MACD, 볼린저 밴드를 결합한 종합 전략',
                'timeframe': '1h',
                'category': 'technical',
                'risk_level': 'medium'
            },
            {
                'id': 'simple_triple_combo',
                'name': '심플 트리플 콤보',
                'description': '간단한 트리플 콤보 전략',
                'timeframe': '1h',
                'category': 'technical',
                'risk_level': 'medium'
            },
            {
                'id': 'rsi_strategy',
                'name': 'RSI 전략',
                'description': 'RSI 지표를 활용한 역추세 전략',
                'timeframe': '15m',
                'category': 'technical',
                'risk_level': 'low'
            },
            {
                'id': 'macd_strategy',
                'name': 'MACD 전략',
                'description': 'MACD 크로스오버 전략',
                'timeframe': '30m',
                'category': 'technical',
                'risk_level': 'medium'
            },
            {
                'id': 'bollinger_strategy',
                'name': '볼린저 밴드 전략',
                'description': '볼린저 밴드 돌파 전략',
                'timeframe': '1h',
                'category': 'technical',
                'risk_level': 'medium'
            },
            {
                'id': 'momentum_strategy',
                'name': '모멘텀 전략',
                'description': '가격 모멘텀 기반 추세 추종 전략',
                'timeframe': '4h',
                'category': 'trend',
                'risk_level': 'high'
            },
            {
                'id': 'mean_reversion',
                'name': '평균 회귀 전략',
                'description': '가격의 평균 회귀 특성을 활용한 전략',
                'timeframe': '1h',
                'category': 'statistical',
                'risk_level': 'medium'
            },
            {
                'id': 'ml_ensemble',
                'name': 'ML 앙상블 전략',
                'description': '머신러닝 앙상블 모델 기반 예측 전략',
                'timeframe': '1h',
                'category': 'machine_learning',
                'risk_level': 'high'
            },
            {
                'id': 'grid_trading',
                'name': '그리드 트레이딩',
                'description': '격자 매매 전략',
                'timeframe': '5m',
                'category': 'algorithmic',
                'risk_level': 'medium'
            },
            {
                'id': 'arbitrage',
                'name': '차익거래 전략',
                'description': '시장 간 가격 차이를 활용한 무위험 수익 전략',
                'timeframe': '1m',
                'category': 'arbitrage',
                'risk_level': 'low'
            }
        ]
        
        return jsonify({'strategies': strategies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """백테스트 실행 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다.'}), 400
        
        # 백테스트 설정 검증
        required_fields = ['startDate', 'endDate', 'strategy', 'initialCapital']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field}가 누락되었습니다.'}), 400
        
        # 백테스트 ID 생성
        import uuid
        backtest_id = str(uuid.uuid4())
        
        # 백테스트 설정
        backtest_config = {
            'id': backtest_id,
            'start_date': data['startDate'],
            'end_date': data['endDate'],
            'symbol': data.get('symbol', 'BTC/USDT'),
            'symbol_type': data.get('symbolType', 'individual'),
            'strategy': data['strategy'],
            'timeframe': data.get('timeframe', '1h'),
            'initial_capital': float(data['initialCapital']),
            'leverage': int(data.get('leverage', 1)),
            'ml_optimization': data.get('mlOptimization') == 'on'
        }
        
        # 백테스트 실행 (실제로는 백그라운드 프로세스로 실행)
        # 여기서는 성공 응답만 반환
        
        return jsonify({
            'status': 'success',
            'message': '백테스트가 시작되었습니다.',
            'backtest_id': backtest_id,
            'config': backtest_config
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/stop', methods=['POST'])
def stop_backtest():
    """백테스트 중지 API"""
    try:
        # 백테스트 중지 로직 구현
        return jsonify({
            'status': 'success',
            'message': '백테스트가 중지되었습니다.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/results', methods=['GET'])
def get_backtest_results():
    """백테스트 결과 조회 API"""
    try:
        # 더미 백테스트 결과 데이터
        results = [
            {
                'id': 1,
                'strategy_name': '트리플 콤보 전략',
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 12500000,
                'leverage': 2,
                'total_trades': 45,
                'winning_trades': 28,
                'losing_trades': 17,
                'win_rate': 62.22,
                'sharpe_ratio': 1.85,
                'max_drawdown': 8.5,
                'created_at': '2025-01-11T14:30:00',
                'ml_optimized': True,
                'ml_params': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26}
            },
            {
                'id': 2,
                'strategy_name': 'RSI 전략',
                'symbol': 'ETH/USDT',
                'timeframe': '15m',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 10800000,
                'leverage': 1,
                'total_trades': 78,
                'winning_trades': 42,
                'losing_trades': 36,
                'win_rate': 53.85,
                'sharpe_ratio': 1.12,
                'max_drawdown': 12.3,
                'created_at': '2025-01-11T13:15:00',
                'ml_optimized': False
            },
            {
                'id': 3,
                'strategy_name': 'MACD 전략',
                'symbol': 'BNB/USDT',
                'timeframe': '30m',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 11200000,
                'leverage': 3,
                'total_trades': 32,
                'winning_trades': 19,
                'losing_trades': 13,
                'win_rate': 59.38,
                'sharpe_ratio': 1.45,
                'max_drawdown': 15.2,
                'created_at': '2025-01-11T12:00:00',
                'ml_optimized': True,
                'ml_params': {'macd_fast': 8, 'macd_slow': 21, 'signal': 9}
            }
        ]
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/stream_log')
def stream_backtest_log():
    """백테스트 로그 실시간 스트리밍 (SSE)"""
    def generate_log_stream():
        import time
        import json
        
        # 더미 로그 데이터
        log_messages = [
            "백테스트 시작...",
            "데이터 로딩 중...",
            "BTC/USDT 데이터 로드 완료",
            "전략 초기화 중...",
            "트리플 콤보 전략 설정 완료",
            "백테스트 진행 중... 10%",
            "첫 번째 매수 신호 발생",
            "포지션 진입: BTC/USDT LONG",
            "백테스트 진행 중... 25%",
            "이익 실현: +3.2%",
            "백테스트 진행 중... 50%",
            "새로운 매수 신호 발생",
            "포지션 진입: BTC/USDT LONG",
            "백테스트 진행 중... 75%",
            "손실 제한: -1.5%",
            "백테스트 진행 중... 90%",
            "백테스트 완료!",
            "최종 수익률: +25.0%",
            "총 거래 횟수: 45회",
            "승률: 62.22%"
        ]
        
        for i, message in enumerate(log_messages):
            progress = int((i + 1) / len(log_messages) * 100)
            log_data = {
                'message': message,
                'timestamp': time.time(),
                'progress': progress
            }
            yield f"data: {json.dumps(log_data)}\n\n"
            time.sleep(0.5)  # 0.5초 간격으로 로그 전송
        
        # 완료 신호
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
    
    return Response(generate_log_stream(), mimetype='text/event-stream')

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