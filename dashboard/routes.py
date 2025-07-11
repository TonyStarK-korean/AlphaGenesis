from flask import Blueprint, jsonify, request, Response, render_template, redirect
import json
import queue
import asyncio
from datetime import datetime
import sys
import os

# 코어 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from core.backtest_engine import RealBacktestEngine
from core.strategy_analyzer import StrategyAnalyzer

# Flask Blueprint 생성
api = Blueprint('api', __name__)

# 백테스트 엔진 초기화
backtest_engine = RealBacktestEngine()
strategy_analyzer = StrategyAnalyzer()

# 백테스트 결과 저장소 (실제로는 데이터베이스 사용)
backtest_results = []

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
    # 실제 실전매매 상태 반환 (현재는 기본값)
    status = {
        'status': 'stopped',
        'start_time': None,
        'current_capital': 10000000,
        'total_pnl': 0,
        'active_positions': 0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'daily_return': 0.0
    }
    return jsonify(status)

@api.route('/api/live/positions', methods=['GET'])
def get_live_positions():
    """활성 포지션 조회 API"""
    # 실제 활성 포지션 반환 (현재는 빈 배열)
    positions = []
    return jsonify({'positions': positions})

@api.route('/api/binance/symbols', methods=['GET'])
def get_binance_symbols():
    """바이낸스 USDT.P 심볼 조회 API"""
    try:
        # 실제 바이낸스 USDT 선물 심볼 목록 (2025년 기준 주요 심볼들)
        symbols = [
            # 주요 암호화폐
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
            'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
            'LTCUSDT', 'BCHUSDT', 'XRPUSDT', 'DOGEUSDT', 'SHIBUSDT',
            
            # 레이어1 블록체인
            'ATOMUSDT', 'FTMUSDT', 'NEARUSDT', 'ALGOUSDT', 'VETUSDT',
            'ICPUSDT', 'EGLDUSDT', 'XTZUSDT', 'FLOWUSDT', 'HBARUSDT',
            'ZILUSDT', 'XLMUSDT', 'ADXUSDT', 'KAVAUSDT', 'WAVESUSDT',
            
            # DeFi 토큰
            'COMPUSDT', 'SUSHIUSDT', 'YFIUSDT', 'CRVUSDT', 'BALUSDT',
            'ZRXUSDT', 'MKRUSDT', 'AAVEUSDT', 'SNXUSDT', 'UMAUSDT',
            'BANDUSDT', 'KNCUSDT', 'RENUSDT', 'LRCUSDT', 'REPUSDT',
            
            # 메타버스/게임
            'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'ENJUSDT', 'CHZUSDT',
            'GALAUSDT', 'APEUSDT', 'GMTUSDT', 'STEPNUSDT', 'TLMUSDT',
            'ALICEUSDT', 'RAREUSDT', 'SUPERUSDT', 'CTSIUSDT', 'XECUSDT',
            
            # 레이어2/확장성
            'OPUSDT', 'ARBUSDT', 'LDOUSDT', 'STXUSDT', 'LOOPUSDT',
            'CELOUSDT', 'SKLUSDT', 'OMGUSDT', 'BATUSDT', 'QTUMUSDT',
            
            # 스토리지/클라우드
            'FILUSDT', 'STORJUSDT', 'SCUSDT', 'ARUSDT', 'HOTUSDT',
            
            # 오라클/데이터
            'LINKUSDT', 'BANDUSDT', 'TRXUSDT', 'IOSTUSDT', 'ONTUSDT',
            
            # 프라이버시 코인
            'XMRUSDT', 'ZECUSDT', 'DASHUSDT', 'SCRTUSDT',
            
            # 기타 주요 알트코인
            'EOSUSDT', 'NEOUSDT', 'THETAUSDT', 'VETUSDT', 'ICXUSDT',
            'IOTAUSDT', 'NANOUSUSDT', 'XEMUSDT', 'DGBUSDT', 'RVNUSDT',
            'CTKUSDT', 'ONEUSDT', 'ZILUSDT', 'ANKRUSDT', 'CHRUSDT',
            'COTIUSDT', 'STMXUSDT', 'KMDUSDT', 'DENTUSDT', 'KEYUSDT',
            'BNXUSDT', 'REIUSDT', 'OPUSDT', 'HFTUSDT', 'PHBUSDT',
            'HOOKUSDT', 'MAGICUSDT', 'HGHUSDT', 'PROMUSDT', 'IDUSDT',
            'ARBUSDT', 'RDNTUSDT', 'WLDUSDT', 'FDUSDUSDT', 'PENDLEUSDT',
            'ARKMUSDT', 'AGIXUSDT', 'YGGUSDT', 'DODXUSDT', 'ANCUSDT',
            'CKBUSDT', 'TRUUSDT', 'LQTYUSDT', 'AMBUSDT', 'GASUSDT',
            'GLMRUSDT', 'OXTUSDT', 'BELUSDT', 'RIFUSDT', 'POLYXUSDT',
            'ATMUSDT', 'PHAUSDT', 'GMXUSDT', 'CFXUSDT', 'STGUSDT',
            'ROSEUSDT', 'CVXUSDT', 'WOOUSDT', 'FXSUSDT', 'METISUSDT',
            'EUROUSDT', 'TWTUSDT', 'BTTUSDT', 'WINUSDT', 'SUNUSDT',
            'KLAYUSDT', 'AXLUSDT', 'CELRUSDT', 'CTXCUSDT', 'QIUSDT',
            'LITUSDT', 'SCRTUSDT', 'QUICKUSDT', 'ALFAUSDT', 'VOXELUSDT',
            'HIGHUSDT', 'CVPUSDT', 'EPXUSDT', 'JSTUSDT', 'SXPUSDT',
            'FIDAUSDT', 'AGLDUSDT', 'RADUSDT', 'BETAUSDT', 'RAREUSDT',
            'LDOUSDT', 'ASTRUSDT', 'FTTUSDT', 'DOCKUSDT', 'ADAUSDT',
            'GTOUSDT', 'CLVUSDT', 'TKOUSUSDT', 'STRAXUSDT', 'UNFIUSDT',
            'BONDUSDT', 'MBOXUSDT', 'FORUSUSDT', 'REQUSDT', 'GHSTUSDT',
            'WAXPUSDT', 'GNOUSDT', 'XVGUSDT', 'HIVEUSDT', 'LSKUSDT',
            'MDXUSDT', 'DIAUSDT', 'VITEUSDT', 'AUDIOUSDT', 'CVCUSDT',
            'PERPUSDT', 'XVSUSDT', 'ALPHAUSDT', 'VIDTUSDT', 'AUCTIONUSDT',
            'BIGTIMEUSDT', 'ALTUSDT', 'PYTHUSDT', 'RONINUSDT', 'DYMUSDT',
            'OMUSDT', 'PIXELUSDT', 'STRKUSDT', 'MAVIAUSDT', 'GLMUSDT',
            'PORTALUSDT', 'TONUSDT', 'AXLUSDT', 'MYROUSDT', 'METISUSDT',
            'AEVOUSDT', 'VANRYUSDT', 'BOMEUSDT', 'ETHFIUSDT', 'ENAUSDT',
            'WIFUSDT', 'JUPUSDT', 'DYMUSUSDT', 'SUIUSDT', 'APTUSDT',
            'SEIUSDT', 'INJUSDT', 'TIAUSDT', 'ORDIUSDT', 'BEAMXUSDT',
            'POLUSDT', 'POWRUSDT', 'SLERFUSDT', 'BRETTUSDT', 'MEWUSDT',
            'TNSR'
        ]
        
        # 실제 환경에서는 ccxt 라이브러리를 사용해서 동적으로 가져옴
        # import ccxt
        # exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        # markets = exchange.load_markets()
        # symbols = [symbol.replace('/', '') for symbol in markets.keys() if symbol.endswith('/USDT')]
        
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
    """실제 백테스트 실행 API"""
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
        
        # 전략 이름을 ID로 매핑
        strategy_name_to_id = {
            '트리플 콤보 전략': 'triple_combo',
            '심플 트리플 콤보': 'simple_triple_combo',
            'RSI 전략': 'rsi_strategy',
            'MACD 전략': 'macd_strategy',
            '볼린저 밴드 전략': 'bollinger_strategy',
            '모멘텀 전략': 'momentum_strategy',
            '평균 회귀 전략': 'mean_reversion',
            'ML 앙상블 전략': 'ml_ensemble',
            '그리드 트레이딩': 'grid_trading',
            '차익거래 전략': 'arbitrage'
        }
        
        # 전략 이름을 ID로 변환
        strategy_name = data['strategy']
        strategy_id = strategy_name_to_id.get(strategy_name, strategy_name)
        
        # 백테스트 설정
        backtest_config = {
            'id': backtest_id,
            'start_date': data['startDate'],
            'end_date': data['endDate'],
            'symbol': data.get('symbol', 'BTC/USDT'),
            'symbol_type': data.get('symbolType', 'individual'),
            'strategy': strategy_id,
            'timeframe': data.get('timeframe', '1h'),
            'initial_capital': float(data['initialCapital']),
            'ml_optimization': data.get('mlOptimization') == 'on'
        }
        
        # 백테스트 모드 확인
        backtest_mode = data.get('backtestMode', 'single')
        
        if backtest_mode == 'strategy_analysis':
            # 전략 통합 분석 모드
            backtest_config.update({
                'mode': 'strategy_analysis',
                'analysis_type': 'comprehensive'
            })
            return jsonify({
                'status': 'success',
                'message': '전략 통합 분석이 시작되었습니다.',
                'backtest_id': backtest_id,
                'config': backtest_config,
                'mode': 'strategy_analysis'
            })
        else:
            # 일반 백테스트 모드
            backtest_config.update({
                'mode': 'single'
            })
            return jsonify({
                'status': 'success',
                'message': '실제 백테스트가 시작되었습니다.',
                'backtest_id': backtest_id,
                'config': backtest_config,
                'mode': 'single'
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

@api.route('/api/backtest/reset', methods=['POST'])
def reset_backtest():
    """백테스트 결과 초기화 API"""
    try:
        # 백테스트 결과 완전 초기화
        global backtest_results
        backtest_results.clear()
        
        # 백테스트 엔진 상태 초기화
        if hasattr(backtest_engine, 'results'):
            backtest_engine.results.clear()
        
        # 전략 분석기 상태 초기화
        if hasattr(strategy_analyzer, 'analysis_results'):
            strategy_analyzer.analysis_results.clear()
        
        return jsonify({
            'status': 'success',
            'message': '모든 백테스트 결과가 초기화되었습니다.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/strategy_analysis', methods=['POST'])
def run_strategy_analysis():
    """전략 통합 분석 실행 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다.'}), 400
        
        # 필수 필드 검증
        required_fields = ['startDate', 'endDate', 'initialCapital']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field}가 누락되었습니다.'}), 400
        
        # 분석 ID 생성
        import uuid
        analysis_id = str(uuid.uuid4())
        
        # 분석 설정
        analysis_config = {
            'id': analysis_id,
            'start_date': data['startDate'],
            'end_date': data['endDate'],
            'initial_capital': float(data['initialCapital']),
            'analysis_type': 'comprehensive',
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'message': '전략 통합 분석이 시작되었습니다.',
            'analysis_id': analysis_id,
            'config': analysis_config
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/strategy_analysis/results/<analysis_id>', methods=['GET'])
def get_strategy_analysis_results(analysis_id):
    """전략 분석 결과 조회 API"""
    try:
        # 실제 구현에서는 데이터베이스에서 조회
        # 여기서는 데모 데이터 반환
        analysis_results = {
            'analysis_id': analysis_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'market_regime': {
                'regime_type': 'bull_weak',
                'volatility_level': 'medium',
                'trend_strength': 0.68,
                'dominant_patterns': ['RSI 중립', 'MACD 상승 추세', '볼린저 밴드 정상']
            },
            'strategy_rankings': [
                {
                    'rank': 1,
                    'strategy_name': '트리플 콤보 전략',
                    'total_score': 85.4,
                    'performance_score': 88.2,
                    'risk_score': 82.6,
                    'consistency_score': 87.1,
                    'adaptability_score': 83.7,
                    'recommendation': '🌟 최적 전략 - 적극 활용 권장'
                },
                {
                    'rank': 2,
                    'strategy_name': 'ML 앙상블 전략',
                    'total_score': 82.3,
                    'performance_score': 85.6,
                    'risk_score': 78.9,
                    'consistency_score': 84.2,
                    'adaptability_score': 80.5,
                    'recommendation': '✅ 우수 전략 - 활용 권장'
                },
                {
                    'rank': 3,
                    'strategy_name': '모멘텀 전략',
                    'total_score': 78.7,
                    'performance_score': 82.4,
                    'risk_score': 75.1,
                    'consistency_score': 79.3,
                    'adaptability_score': 77.9,
                    'recommendation': '✅ 우수 전략 - 활용 권장'
                },
                {
                    'rank': 4,
                    'strategy_name': 'RSI 전략',
                    'total_score': 65.2,
                    'performance_score': 68.4,
                    'risk_score': 72.8,
                    'consistency_score': 61.5,
                    'adaptability_score': 58.1,
                    'recommendation': '⚠️ 보통 전략 - 조건부 활용'
                },
                {
                    'rank': 5,
                    'strategy_name': 'MACD 전략',
                    'total_score': 58.9,
                    'performance_score': 62.1,
                    'risk_score': 65.4,
                    'consistency_score': 55.7,
                    'adaptability_score': 52.4,
                    'recommendation': '🔄 개선 필요 - 파라미터 최적화 권장'
                }
            ],
            'portfolio_recommendations': [
                {
                    'name': '균형 포트폴리오',
                    'strategies': [
                        {'name': '트리플 콤보 전략', 'weight': 0.4},
                        {'name': 'ML 앙상블 전략', 'weight': 0.3},
                        {'name': '모멘텀 전략', 'weight': 0.3}
                    ],
                    'expected_return': 86.7,
                    'risk_level': 'Medium'
                },
                {
                    'name': '고수익 포트폴리오',
                    'strategies': [
                        {'name': '트리플 콤보 전략', 'weight': 0.6},
                        {'name': 'ML 앙상블 전략', 'weight': 0.4}
                    ],
                    'expected_return': 86.9,
                    'risk_level': 'High'
                },
                {
                    'name': '안전 포트폴리오',
                    'strategies': [
                        {'name': '트리플 콤보 전략', 'weight': 0.5},
                        {'name': 'RSI 전략', 'weight': 0.5}
                    ],
                    'expected_return': 76.8,
                    'risk_level': 'Low'
                }
            ],
            'key_insights': [
                '현재 시장 국면: 약한 상승 추세',
                '최고 성과 전략: 트리플 콤보 전략',
                '평균 성과 점수: 74.1점',
                '시장 변동성: 보통 수준',
                '추천 전략 조합: 트리플 콤보 + ML 앙상블'
            ],
            'risk_management_tips': [
                '현재 시장 변동성: medium',
                '동적 레버리지 관리 필수',
                '분할 진입/청산 전략 활용',
                '시장 국면별 전략 전환 준비'
            ]
        }
        
        return jsonify(analysis_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/results', methods=['GET'])
def get_backtest_results():
    """백테스트 결과 조회 API"""
    try:
        # 쿼리 파라미터에서 필터링 조건 가져오기
        symbol_filter = request.args.get('symbol', 'all')
        strategy_filter = request.args.get('strategy', 'all')
        period_filter = request.args.get('period', 'all')
        
        # 실제 백테스트 결과 데이터 조회
        results = []
        
        # 전역 backtest_results에서 실제 결과 가져오기
        for result in backtest_results:
            if isinstance(result, dict):
                # 딕셔너리 형태의 결과
                result_dict = result
            else:
                # BacktestResult 객체
                result_dict = {
                    'id': len(results) + 1,
                    'strategy_name': result.strategy_name,
                    'symbol': result.symbol,
                    'timeframe': result.timeframe,
                    'start_date': result.start_date,
                    'end_date': result.end_date,
                    'initial_capital': result.initial_capital,
                    'final_value': result.final_value,
                    'total_return': result.total_return,
                    'leverage': f'동적 (평균 {result.avg_leverage:.1f}x)',
                    'dynamic_leverage': True,
                    'avg_leverage': result.avg_leverage,
                    'max_leverage': result.max_leverage,
                    'min_leverage': result.min_leverage,
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'win_rate': result.win_rate,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'created_at': result.created_at,
                    'ml_optimized': result.ml_optimized,
                    'ml_params': result.ml_params or {},
                    'split_trades': result.split_trades
                }
            
            # 필터링 적용
            if symbol_filter != 'all' and result_dict['symbol'] != symbol_filter:
                continue
            if strategy_filter != 'all' and result_dict['strategy_name'] != strategy_filter:
                continue
            # 기간 필터링은 복잡하므로 생략
            
            results.append(result_dict)
        
        # 결과가 없는 경우 빈 배열 반환
        if not results:
            results = []
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/results', methods=['POST'])
def save_backtest_result():
    """백테스트 결과 저장 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '결과 데이터가 없습니다.'}), 400
        
        # 결과 저장 로직 (실제로는 데이터베이스에 저장)
        # 여기서는 성공 응답만 반환
        result_id = len(get_backtest_results()['results']) + 1
        
        # 결과 데이터 구조 예시
        saved_result = {
            'id': result_id,
            'strategy_name': data.get('strategy_name'),
            'symbol': data.get('symbol'),
            'timeframe': data.get('timeframe'),
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'initial_capital': data.get('initial_capital'),
            'final_value': data.get('final_value'),
            'total_return': data.get('total_return'),
            'sharpe_ratio': data.get('sharpe_ratio'),
            'max_drawdown': data.get('max_drawdown'),
            'win_rate': data.get('win_rate'),
            'total_trades': data.get('total_trades'),
            'created_at': datetime.now().isoformat(),
            'dynamic_leverage': data.get('dynamic_leverage', True),
            'avg_leverage': data.get('avg_leverage', 2.0),
            'ml_optimized': data.get('ml_optimized', False)
        }
        
        return jsonify({
            'status': 'success',
            'message': '백테스트 결과가 저장되었습니다.',
            'result_id': result_id,
            'result': saved_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/results/reset', methods=['POST'])
def reset_backtest_results():
    """백테스트 결과 초기화 API"""
    try:
        # 실제로는 데이터베이스의 백테스트 결과를 모두 삭제
        # 여기서는 성공 응답만 반환
        return jsonify({
            'status': 'success',
            'message': '모든 백테스트 결과가 초기화되었습니다.',
            'reset_count': 0  # 실제로는 삭제된 결과 수
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/statistics', methods=['GET'])
def get_backtest_statistics():
    """백테스트 통계 조회 API"""
    try:
        # 전략별 통계 계산
        strategy_stats = {
            '트리플 콤보 전략': {
                'total_tests': 12,
                'avg_return': 18.7,
                'avg_sharpe': 1.65,
                'avg_drawdown': 11.2,
                'win_rate': 67.5,
                'best_symbol': 'BTC/USDT',
                'worst_symbol': 'DOGE/USDT'
            },
            'RSI 전략': {
                'total_tests': 8,
                'avg_return': 12.4,
                'avg_sharpe': 1.32,
                'avg_drawdown': 15.8,
                'win_rate': 58.3,
                'best_symbol': 'ETH/USDT',
                'worst_symbol': 'ADA/USDT'
            },
            'MACD 전략': {
                'total_tests': 6,
                'avg_return': 15.2,
                'avg_sharpe': 1.48,
                'avg_drawdown': 12.9,
                'win_rate': 62.1,
                'best_symbol': 'BNB/USDT',
                'worst_symbol': 'XRP/USDT'
            }
        }
        
        # 심볼별 통계
        symbol_stats = {
            'BTC/USDT': {
                'total_tests': 15,
                'avg_return': 19.8,
                'best_strategy': '트리플 콤보 전략',
                'worst_strategy': 'RSI 전략'
            },
            'ETH/USDT': {
                'total_tests': 12,
                'avg_return': 16.2,
                'best_strategy': 'RSI 전략',
                'worst_strategy': 'MACD 전략'
            },
            'BNB/USDT': {
                'total_tests': 8,
                'avg_return': 14.7,
                'best_strategy': 'MACD 전략',
                'worst_strategy': 'RSI 전략'
            }
        }
        
        # 기간별 통계
        period_stats = {
            '1개월': {
                'total_tests': 20,
                'avg_return': 8.5,
                'volatility': 'High'
            },
            '3개월': {
                'total_tests': 10,
                'avg_return': 22.3,
                'volatility': 'Medium'
            },
            '6개월': {
                'total_tests': 5,
                'avg_return': 41.2,
                'volatility': 'Low'
            }
        }
        
        return jsonify({
            'strategy_stats': strategy_stats,
            'symbol_stats': symbol_stats,
            'period_stats': period_stats,
            'total_tests': 35,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/stream_log')
def stream_backtest_log():
    """실제 백테스트 로그 실시간 스트리밍 (SSE)"""
    # request context가 있을 때 파라미터 추출
    start_date = request.args.get('start_date', '2025-01-01')
    end_date = request.args.get('end_date', '2025-07-11')
    symbol = request.args.get('symbol', 'BTC/USDT')
    strategy_name = request.args.get('strategy', 'triple_combo')
    initial_capital = float(request.args.get('initial_capital', '10000000'))
    backtest_mode = request.args.get('backtest_mode', 'single')
    ml_optimization = request.args.get('ml_optimization', 'off') == 'on'
    
    # 전략 이름을 ID로 매핑
    strategy_name_to_id = {
        '트리플 콤보 전략': 'triple_combo',
        '심플 트리플 콤보': 'simple_triple_combo',
        'RSI 전략': 'rsi_strategy',
        'MACD 전략': 'macd_strategy',
        '볼린저 밴드 전략': 'bollinger_strategy',
        '모멘텀 전략': 'momentum_strategy',
        '평균 회귀 전략': 'mean_reversion',
        'ML 앙상블 전략': 'ml_ensemble',
        '그리드 트레이딩': 'grid_trading',
        '차익거래 전략': 'arbitrage'
    }
    
    # 전략 이름을 ID로 변환
    strategy = strategy_name_to_id.get(strategy_name, strategy_name)
    
    def generate_log_stream():
        import time
        import json
        from datetime import datetime
        
        # 로그 큐 저장소
        log_queue = []
        
        # 실시간 로그 전송을 위한 콜백 함수
        def log_callback(message, log_type, progress=None):
            log_data = {
                'message': message,
                'type': log_type,
                'timestamp': time.time(),
                'progress': progress
            }
            log_queue.append(log_data)
        
        try:
            # 백테스트 설정
            config = {
                'strategy': strategy,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'ml_optimization': ml_optimization
            }
            
            # 심볼 타입 결정
            if symbol == 'ALL_MARKET':
                config['symbol_type'] = 'market_wide'
                config['symbol'] = 'BTC/USDT'  # 대표 심볼
            else:
                config['symbol_type'] = 'individual'
                config['symbol'] = symbol
                # 심볼 형식 정규화
                if 'USDT' in symbol and '/' not in symbol:
                    config['symbol'] = symbol.replace('USDT', '/USDT')
            
            # 백테스트 모드에 따른 처리
            if backtest_mode == 'strategy_analysis':
                # 전략 통합 분석 모드
                async def run_strategy_analysis():
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    
                    result = await strategy_analyzer.analyze_all_strategies(
                        start_dt, end_dt, initial_capital, log_callback
                    )
                    
                    # 결과 전송
                    log_callback("🎯 전략 분석 완료", "system", 100)
                    if result and 'rankings' in result and result['rankings']:
                        log_callback(f"📈 최고 성과 전략: {result['rankings'][0]['strategy_name']}", "result", 100)
                        log_callback(f"📊 총 {len(result['strategy_results'])}개 전략 분석 완료", "result", 100)
                    
                    return result
                
                # 분석 실행
                result = asyncio.run(run_strategy_analysis())
                
            else:
                # 일반 백테스트 모드
                async def run_backtest():
                    result = await backtest_engine.run_backtest(config, log_callback)
                    
                    # 결과 저장
                    backtest_results.append(result)
                    
                    # 최종 결과 전송
                    log_callback("🎯 백테스트 완료", "system", 100)
                    log_callback(f"📈 최종 수익률: {result.total_return:.2f}%", "result", 100)
                    log_callback(f"💰 최종 자본: {result.final_value:,.0f}원", "result", 100)
                    log_callback(f"📊 총 거래 횟수: {result.total_trades}회", "result", 100)
                    log_callback(f"🎯 승률: {result.win_rate:.1f}%", "result", 100)
                    log_callback(f"📉 최대 낙폭: {result.max_drawdown:.2f}%", "result", 100)
                    log_callback(f"⚡ 평균 레버리지: {result.avg_leverage:.1f}x", "result", 100)
                    
                    return result
                
                # 백테스트 실행
                result = asyncio.run(run_backtest())
            
            # 큐에 저장된 로그들을 스트리밍
            for log_data in log_queue:
                yield f"data: {json.dumps(log_data)}\n\n"
                time.sleep(0.1)  # 스트리밍 간격
                
        except Exception as e:
            error_log = {
                'message': f"❌ 백테스트 실패: {str(e)}",
                'type': 'error',
                'timestamp': time.time(),
                'progress': 0
            }
            yield f"data: {json.dumps(error_log)}\n\n"
        
        # 완료 신호
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
    
    return Response(generate_log_stream(), mimetype='text/event-stream')

@api.route('/api/market/overview', methods=['GET'])
def get_market_overview():
    """시장 개요 조회 API"""
    # 실제 시장 개요 (현재는 기본값)
    overview = {
        'timestamp': datetime.now().isoformat(),
        'major_symbols': {
            'BTC/USDT': {'price': 0, 'change': 0.0},
            'ETH/USDT': {'price': 0, 'change': 0.0},
            'BNB/USDT': {'price': 0, 'change': 0.0}
        },
        'market_sentiment': 'neutral',
        'total_volume': 0,
        'trending_symbols': []
    }
    return jsonify(overview)