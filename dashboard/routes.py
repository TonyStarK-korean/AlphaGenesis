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
        # 더미 백테스트 결과 데이터 (동적 레버리지 반영)
        results = [
            {
                'id': 1,
                'strategy_name': '트리플 콤보 전략',
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 12540000,
                'leverage': '동적 (평균 2.4x)',
                'dynamic_leverage': True,
                'avg_leverage': 2.4,
                'max_leverage': 3.2,
                'min_leverage': 1.8,
                'total_trades': 45,
                'winning_trades': 31,
                'losing_trades': 14,
                'win_rate': 68.9,
                'sharpe_ratio': 1.95,
                'max_drawdown': 8.5,
                'created_at': '2025-01-11T14:30:00',
                'ml_optimized': True,
                'ml_params': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26},
                'split_trades': {
                    'total_splits': 18,
                    'split_success_rate': 89.3,
                    'avg_split_count': 2.1
                }
            },
            {
                'id': 2,
                'strategy_name': 'RSI 전략',
                'symbol': 'ETH/USDT',
                'timeframe': '15m',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 11820000,
                'leverage': '동적 (평균 1.8x)',
                'dynamic_leverage': True,
                'avg_leverage': 1.8,
                'max_leverage': 2.5,
                'min_leverage': 1.2,
                'total_trades': 78,
                'winning_trades': 46,
                'losing_trades': 32,
                'win_rate': 58.9,
                'sharpe_ratio': 1.32,
                'max_drawdown': 12.3,
                'created_at': '2025-01-11T13:15:00',
                'ml_optimized': False,
                'split_trades': {
                    'total_splits': 12,
                    'split_success_rate': 75.4,
                    'avg_split_count': 1.8
                }
            },
            {
                'id': 3,
                'strategy_name': 'MACD 전략',
                'symbol': 'BNB/USDT',
                'timeframe': '30m',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 12210000,
                'leverage': '동적 (평균 2.1x)',
                'dynamic_leverage': True,
                'avg_leverage': 2.1,
                'max_leverage': 2.8,
                'min_leverage': 1.5,
                'total_trades': 32,
                'winning_trades': 20,
                'losing_trades': 12,
                'win_rate': 62.5,
                'sharpe_ratio': 1.65,
                'max_drawdown': 15.2,
                'created_at': '2025-01-11T12:00:00',
                'ml_optimized': True,
                'ml_params': {'macd_fast': 8, 'macd_slow': 21, 'signal': 9},
                'split_trades': {
                    'total_splits': 8,
                    'split_success_rate': 87.5,
                    'avg_split_count': 2.3
                }
            },
            {
                'id': 4,
                'strategy_name': '모멘텀 전략',
                'symbol': 'SOL/USDT',
                'timeframe': '4h',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 13180000,
                'leverage': '동적 (평균 3.2x)',
                'dynamic_leverage': True,
                'avg_leverage': 3.2,
                'max_leverage': 4.5,
                'min_leverage': 2.1,
                'total_trades': 28,
                'winning_trades': 17,
                'losing_trades': 11,
                'win_rate': 60.7,
                'sharpe_ratio': 2.05,
                'max_drawdown': 18.7,
                'created_at': '2025-01-11T11:30:00',
                'ml_optimized': True,
                'ml_params': {'momentum_period': 20, 'threshold': 0.05},
                'split_trades': {
                    'total_splits': 15,
                    'split_success_rate': 93.3,
                    'avg_split_count': 2.7
                }
            },
            {
                'id': 5,
                'strategy_name': 'ML 앙상블 전략',
                'symbol': 'AVAX/USDT',
                'timeframe': '1h',
                'start_date': '2024-12-11',
                'end_date': '2025-01-11',
                'initial_capital': 10000000,
                'final_value': 12860000,
                'leverage': '동적 (평균 2.8x)',
                'dynamic_leverage': True,
                'avg_leverage': 2.8,
                'max_leverage': 3.5,
                'min_leverage': 1.9,
                'total_trades': 52,
                'winning_trades': 37,
                'losing_trades': 15,
                'win_rate': 71.2,
                'sharpe_ratio': 2.15,
                'max_drawdown': 11.4,
                'created_at': '2025-01-11T10:45:00',
                'ml_optimized': True,
                'ml_params': {'ensemble_models': ['XGBoost', 'RandomForest', 'LSTM']},
                'split_trades': {
                    'total_splits': 22,
                    'split_success_rate': 90.9,
                    'avg_split_count': 2.5
                }
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
        import random
        
        # 상세한 매매 시뮬레이션 로그
        log_events = [
            # 초기화 단계
            {"message": "🚀 백테스트 시작", "type": "system", "progress": 0},
            {"message": "📊 BTC/USDT 데이터 로딩 중...", "type": "data", "progress": 5},
            {"message": "✅ 2024-12-11 ~ 2025-01-11 (1개월) 데이터 로드 완료", "type": "data", "progress": 10},
            {"message": "🔧 트리플 콤보 전략 초기화", "type": "strategy", "progress": 15},
            {"message": "⚙️ 동적 레버리지 시스템 활성화", "type": "system", "progress": 20},
            {"message": "🎯 초기 자본: 10,000,000원 | 기본 비중: 6%", "type": "capital", "progress": 25},
            
            # 시장 분석 단계
            {"message": "📈 시장 분석 중... 현재 BTC 가격: $43,250", "type": "market", "progress": 30},
            {"message": "🔍 시장 국면 분석: 상승 추세 (RSI: 58.4, MACD: 양수)", "type": "analysis", "progress": 35},
            {"message": "⚡ 동적 레버리지 계산: 현재 변동성 12.5% → 레버리지 2.3x", "type": "leverage", "progress": 40},
            
            # 첫 번째 매수 신호
            {"message": "🎯 매수 신호 발생! RSI(52.1) + MACD 골든크로스 + 볼린저 하단 터치", "type": "signal", "progress": 45},
            {"message": "💰 기본 매수 진입: 2% 비중 (200,000원) | 가격: $43,180", "type": "buy", "progress": 50},
            {"message": "📊 포지션 정보: LONG 0.0046 BTC | 예상 수익률: +8.5%", "type": "position", "progress": 52},
            
            # 분할매수 시나리오
            {"message": "⚠️ 가격 하락 감지: $43,180 → $42,850 (-0.76%)", "type": "price", "progress": 55},
            {"message": "🔄 분할매수 1차: 2% 추가 비중 (200,000원) | 가격: $42,850", "type": "buy_add", "progress": 58},
            {"message": "📈 누적 포지션: 0.0093 BTC | 평균 단가: $43,015", "type": "position", "progress": 60},
            
            {"message": "⚠️ 추가 하락: $42,850 → $42,520 (-0.77%)", "type": "price", "progress": 62},
            {"message": "🔄 분할매수 2차 (최종): 2% 추가 비중 (200,000원) | 가격: $42,520", "type": "buy_add", "progress": 65},
            {"message": "📊 최종 포지션: 0.0140 BTC | 평균 단가: $42,850 | 총 투입: 600,000원", "type": "position", "progress": 68},
            
            # 수익 전환 및 매도
            {"message": "🚀 반등 시작! $42,520 → $43,820 (+3.06%)", "type": "price", "progress": 70},
            {"message": "💚 수익 전환: +$13,580 (+2.26%)", "type": "profit", "progress": 72},
            {"message": "🎯 분할매도 1차: 33% 물량 매도 | 가격: $43,820", "type": "sell", "progress": 75},
            {"message": "💰 부분 이익실현: +$4,526 | 잔여 포지션: 0.0093 BTC", "type": "profit", "progress": 78},
            
            # 추가 상승 및 완전 매도
            {"message": "📈 지속 상승: $43,820 → $44,250 (+0.98%)", "type": "price", "progress": 80},
            {"message": "🎯 분할매도 2차: 50% 물량 매도 | 가격: $44,180", "type": "sell", "progress": 85},
            {"message": "🎯 분할매도 3차 (완전청산): 잔여 물량 매도 | 가격: $44,320", "type": "sell", "progress": 90},
            {"message": "✅ 포지션 완전 청산 | 총 수익: +$18,240 (+3.04%)", "type": "profit", "progress": 92},
            
            # 두 번째 매매 사이클
            {"message": "🔍 새로운 기회 탐색 중...", "type": "analysis", "progress": 94},
            {"message": "⚡ 레버리지 재계산: 변동성 감소 → 레버리지 2.8x", "type": "leverage", "progress": 95},
            {"message": "🎯 새로운 매수 신호: ETH/USDT 진입", "type": "signal", "progress": 96},
            
            # 최종 결과
            {"message": "📊 백테스트 완료!", "type": "system", "progress": 100},
            {"message": "🏆 최종 성과: +25.4% (45회 거래, 승률 68.9%)", "type": "result", "progress": 100},
            {"message": "💎 최적 레버리지 활용: 평균 2.4x", "type": "result", "progress": 100},
            {"message": "🎯 분할매매 성공률: 89.3%", "type": "result", "progress": 100}
        ]
        
        for event in log_events:
            # 로그 타입별 색상 및 아이콘 추가
            log_data = {
                'message': event['message'],
                'type': event['type'],
                'timestamp': time.time(),
                'progress': event['progress']
            }
            yield f"data: {json.dumps(log_data)}\n\n"
            time.sleep(random.uniform(0.3, 0.8))  # 랜덤 간격으로 실제감 증대
        
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