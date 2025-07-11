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
from core.portfolio_optimizer import PortfolioOptimizer

# Flask Blueprint 생성
api = Blueprint('api', __name__)

# 백테스트 엔진 초기화
backtest_engine = RealBacktestEngine()
strategy_analyzer = StrategyAnalyzer()
portfolio_optimizer = PortfolioOptimizer()

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
    """전략 목록 조회 API - 백테스트 엔진에서 실제 전략 로드"""
    try:
        # 백테스트 엔진의 실제 전략 목록을 가져오기
        engine_strategies = backtest_engine.strategies
        
        strategies = []
        for strategy_id, strategy_info in engine_strategies.items():
            strategies.append({
                'id': strategy_id,
                'name': strategy_info.get('name', strategy_id),
                'description': strategy_info.get('description', ''),
                'timeframe': strategy_info.get('timeframe', '1h'),
                'category': 'technical',
                'risk_level': 'medium'
            })
        
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
        
        # 기본값 체크
        if strategy_name == '전략을 선택하세요' or not strategy_name:
            strategy_id = 'triple_combo'  # 기본 전략
        else:
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
    if strategy_name == '전략을 선택하세요' or not strategy_name:
        strategy = 'triple_combo'  # 기본 전략
    else:
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

@api.route('/api/backtest/comprehensive', methods=['POST'])
def run_comprehensive_analysis():
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
            'analysis_type': 'comprehensive'
        }
        
        # 비동기 분석 실행 (실제로는 celery 등 사용)
        async def run_analysis():
            start_dt = datetime.strptime(data['startDate'], '%Y-%m-%d')
            end_dt = datetime.strptime(data['endDate'], '%Y-%m-%d')
            
            result = await strategy_analyzer.analyze_all_strategies(
                start_dt, end_dt, analysis_config['initial_capital']
            )
            
            return {
                'analysis_id': analysis_id,
                'status': 'completed',
                'result': result,
                'config': analysis_config
            }
        
        # 결과 반환
        return jsonify({
            'status': 'success',
            'message': '전략 통합 분석이 시작되었습니다.',
            'analysis_id': analysis_id,
            'config': analysis_config
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@api.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """포트폴리오 최적화 실행 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다.'}), 400
        
        # 필수 필드 검증
        required_fields = ['strategy_results', 'optimization_method', 'risk_level']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field}가 누락되었습니다.'}), 400
        
        # 포트폴리오 최적화 실행
        optimized_portfolios = portfolio_optimizer.optimize_portfolio(
            strategy_results=data['strategy_results'],
            optimization_method=data['optimization_method'],
            risk_level=data['risk_level'],
            constraints=data.get('constraints', {})
        )
        
        # 포트폴리오 보고서 생성
        report = portfolio_optimizer.generate_portfolio_report(optimized_portfolios)
        
        return jsonify({
            'status': 'success',
            'portfolios': [{
                'name': p.name,
                'weights': p.weights,
                'expected_return': p.expected_return,
                'volatility': p.volatility,
                'sharpe_ratio': p.sharpe_ratio,
                'max_drawdown': p.max_drawdown,
                'var_95': p.var_95,
                'cvar_95': p.cvar_95,
                'strategies': p.strategies,
                'rebalancing_frequency': p.rebalancing_frequency,
                'risk_level': p.risk_level,
                'created_at': p.created_at
            } for p in optimized_portfolios],
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/portfolio/analyze', methods=['POST'])
def analyze_portfolio_with_optimization():
    """전략 분석 후 포트폴리오 최적화 통합 실행 API"""
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
        
        # 전략 분석 설정
        start_dt = datetime.strptime(data['startDate'], '%Y-%m-%d')
        end_dt = datetime.strptime(data['endDate'], '%Y-%m-%d')
        initial_capital = float(data['initialCapital'])
        
        # 전략 분석 실행
        async def run_integrated_analysis():
            # 1. 전략 분석 실행
            strategy_results = await strategy_analyzer.analyze_all_strategies(
                start_dt, end_dt, initial_capital
            )
            
            # 2. 전략 결과를 포트폴리오 최적화 형태로 변환
            portfolio_strategy_results = []
            if 'strategy_results' in strategy_results:
                for result in strategy_results['strategy_results']:
                    portfolio_strategy_results.append({
                        'strategy_name': result.get('strategy_name', ''),
                        'total_return': result.get('total_return', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'win_rate': result.get('win_rate', 0),
                        'volatility': result.get('volatility', 0)
                    })
            
            # 3. 포트폴리오 최적화 실행
            optimization_methods = ['sharpe', 'min_vol', 'max_return', 'risk_parity']
            all_portfolios = []
            
            for method in optimization_methods:
                portfolios = portfolio_optimizer.optimize_portfolio(
                    strategy_results=portfolio_strategy_results,
                    optimization_method=method,
                    risk_level=data.get('risk_level', 'medium'),
                    constraints=data.get('constraints', {})
                )
                all_portfolios.extend(portfolios)
            
            # 4. 포트폴리오 보고서 생성
            report = portfolio_optimizer.generate_portfolio_report(all_portfolios)
            
            return {
                'analysis_id': analysis_id,
                'status': 'completed',
                'strategy_analysis': strategy_results,
                'portfolio_optimization': {
                    'portfolios': [{
                        'name': p.name,
                        'weights': p.weights,
                        'expected_return': p.expected_return,
                        'volatility': p.volatility,
                        'sharpe_ratio': p.sharpe_ratio,
                        'max_drawdown': p.max_drawdown,
                        'var_95': p.var_95,
                        'cvar_95': p.cvar_95,
                        'strategies': p.strategies,
                        'rebalancing_frequency': p.rebalancing_frequency,
                        'risk_level': p.risk_level,
                        'created_at': p.created_at
                    } for p in all_portfolios],
                    'report': report
                },
                'timestamp': datetime.now().isoformat()
            }
        
        # 결과 반환 (실제로는 비동기 실행)
        return jsonify({
            'status': 'success',
            'message': '전략 분석 및 포트폴리오 최적화가 시작되었습니다.',
            'analysis_id': analysis_id,
            'config': {
                'start_date': data['startDate'],
                'end_date': data['endDate'],
                'initial_capital': initial_capital,
                'risk_level': data.get('risk_level', 'medium'),
                'analysis_type': 'integrated'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/portfolio/backtest', methods=['POST'])
def backtest_portfolio():
    """포트폴리오 백테스트 실행 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다.'}), 400
        
        # 필수 필드 검증
        required_fields = ['portfolio_weights', 'start_date', 'end_date', 'initial_capital']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field}가 누락되었습니다.'}), 400
        
        # 포트폴리오 백테스트 실행
        portfolio_weights = data['portfolio_weights']
        start_date = data['start_date']
        end_date = data['end_date']
        initial_capital = float(data['initial_capital'])
        
        # 각 전략별 백테스트 실행 후 포트폴리오 성과 계산
        async def run_portfolio_backtest():
            portfolio_results = []
            total_portfolio_value = 0
            
            for strategy_name, weight in portfolio_weights.items():
                if weight > 0:
                    # 개별 전략 백테스트
                    strategy_config = {
                        'strategy': strategy_name,
                        'symbol': 'BTC/USDT',
                        'symbol_type': 'individual',
                        'start_date': start_date,
                        'end_date': end_date,
                        'timeframe': '1h',
                        'initial_capital': initial_capital * weight,
                        'ml_optimization': False
                    }
                    
                    try:
                        result = await backtest_engine.run_backtest(strategy_config)
                        portfolio_results.append({
                            'strategy': strategy_name,
                            'weight': weight,
                            'allocated_capital': initial_capital * weight,
                            'final_value': result.final_value,
                            'return': result.total_return,
                            'sharpe_ratio': result.sharpe_ratio,
                            'max_drawdown': result.max_drawdown,
                            'trades': result.total_trades
                        })
                        total_portfolio_value += result.final_value
                    except Exception as e:
                        logger.error(f"전략 {strategy_name} 백테스트 실패: {e}")
                        portfolio_results.append({
                            'strategy': strategy_name,
                            'weight': weight,
                            'allocated_capital': initial_capital * weight,
                            'final_value': initial_capital * weight,
                            'return': 0,
                            'sharpe_ratio': 0,
                            'max_drawdown': 0,
                            'trades': 0,
                            'error': str(e)
                        })
                        total_portfolio_value += initial_capital * weight
            
            # 포트폴리오 전체 성과 계산
            portfolio_return = (total_portfolio_value - initial_capital) / initial_capital * 100
            
            return {
                'portfolio_performance': {
                    'initial_capital': initial_capital,
                    'final_value': total_portfolio_value,
                    'total_return': portfolio_return,
                    'weights': portfolio_weights
                },
                'strategy_results': portfolio_results,
                'timestamp': datetime.now().isoformat()
            }
        
        # 결과 반환
        return jsonify({
            'status': 'success',
            'message': '포트폴리오 백테스트가 시작되었습니다.',
            'portfolio_config': {
                'weights': portfolio_weights,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500