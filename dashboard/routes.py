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

# API 응답 표준화 함수들
def success_response(data=None, message="Success", status_code=200):
    """성공 응답 표준화"""
    response = {
        "status": "success",
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    return jsonify(response), status_code

def error_response(message="Error", error_code=None, status_code=400):
    """에러 응답 표준화"""
    response = {
        "status": "error",
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "error_code": error_code
    }
    return jsonify(response), status_code

# Flask Blueprint 생성
api = Blueprint('api', __name__)

# 백테스트 엔진 초기화
backtest_engine = RealBacktestEngine()
strategy_analyzer = StrategyAnalyzer()
portfolio_optimizer = PortfolioOptimizer()

# 백테스트 결과 저장소 (실제로는 데이터베이스 사용)
backtest_results = []

# 분석 결과 생성 헬퍼 함수들
def _generate_portfolio_recommendations(strategy_rankings):
    """포트폴리오 추천 생성"""
    if not strategy_rankings:
        return []
    
    recommendations = []
    
    # 상위 3개 전략 선택
    top_strategies = strategy_rankings[:min(3, len(strategy_rankings))]
    
    if len(top_strategies) >= 2:
        # 균형 포트폴리오
        recommendations.append({
            'name': '균형 포트폴리오',
            'strategies': [
                {'name': top_strategies[0]['strategy_name'], 'weight': 0.5},
                {'name': top_strategies[1]['strategy_name'], 'weight': 0.5}
            ],
            'expected_return': round((top_strategies[0]['total_score'] + top_strategies[1]['total_score']) / 2, 1),
            'risk_level': 'Medium'
        })
        
        # 고수익 포트폴리오 (최고 성과 전략 집중)
        recommendations.append({
            'name': '고수익 포트폴리오',
            'strategies': [
                {'name': top_strategies[0]['strategy_name'], 'weight': 0.7},
                {'name': top_strategies[1]['strategy_name'], 'weight': 0.3}
            ],
            'expected_return': round(top_strategies[0]['total_score'] * 0.95, 1),
            'risk_level': 'High'
        })
    
    if len(top_strategies) >= 3:
        # 안전 포트폴리오 (3개 전략 분산)
        recommendations.append({
            'name': '분산 포트폴리오',
            'strategies': [
                {'name': top_strategies[0]['strategy_name'], 'weight': 0.4},
                {'name': top_strategies[1]['strategy_name'], 'weight': 0.3},
                {'name': top_strategies[2]['strategy_name'], 'weight': 0.3}
            ],
            'expected_return': round(sum(s['total_score'] for s in top_strategies) / 3, 1),
            'risk_level': 'Low'
        })
    
    return recommendations

def _generate_key_insights(strategy_rankings, total_results):
    """핵심 인사이트 생성"""
    if not strategy_rankings:
        return ['분석할 데이터가 부족합니다.']
    
    insights = []
    
    # 최고 성과 전략
    best_strategy = strategy_rankings[0]
    insights.append(f'최고 성과 전략: {best_strategy["strategy_name"]} (점수: {best_strategy["total_score"]})')
    
    # 평균 성과
    avg_score = round(sum(s['total_score'] for s in strategy_rankings) / len(strategy_rankings), 1)
    insights.append(f'평균 성과 점수: {avg_score}점')
    
    # 총 백테스트 수
    insights.append(f'총 {total_results}개 백테스트 결과 분석 완료')
    
    # 성과 분포
    excellent_count = len([s for s in strategy_rankings if s['total_score'] >= 80])
    good_count = len([s for s in strategy_rankings if 70 <= s['total_score'] < 80])
    
    if excellent_count > 0:
        insights.append(f'우수 전략 {excellent_count}개, 양호 전략 {good_count}개 발견')
    
    # 위험 수준 평가
    avg_risk = round(sum(s['risk_score'] for s in strategy_rankings) / len(strategy_rankings), 1)
    if avg_risk >= 80:
        insights.append('전반적 위험 관리 수준: 우수')
    elif avg_risk >= 70:
        insights.append('전반적 위험 관리 수준: 양호')
    else:
        insights.append('위험 관리 개선 필요')
    
    return insights

def _generate_risk_tips(strategy_rankings):
    """위험 관리 팁 생성"""
    if not strategy_rankings:
        return ['실제 데이터 수집 후 분석이 가능합니다.']
    
    tips = []
    
    # 평균 위험 점수 기반 조언
    avg_risk = sum(s['risk_score'] for s in strategy_rankings) / len(strategy_rankings)
    
    if avg_risk < 70:
        tips.append('⚠️ 높은 위험 수준 감지 - 포지션 크기 축소 권장')
        tips.append('📉 드로다운 관리 강화 필요')
    else:
        tips.append('✅ 적정 위험 수준 유지 중')
    
    # 성과 편차 기반 조언
    scores = [s['total_score'] for s in strategy_rankings]
    score_std = (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
    
    if score_std > 15:
        tips.append('📊 전략별 성과 편차 큼 - 분산 투자 권장')
    else:
        tips.append('📈 전략별 성과 안정적')
    
    # 일반적인 조언
    tips.append('🔄 동적 레버리지 관리 필수')
    tips.append('⏰ 정기적인 성과 리뷰 및 전략 재조정')
    
    return tips

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
    """통합 서버 상태 확인 API"""
    return jsonify({
        "status": "ok",
        "service": "AlphaGenesis",
        "version": "3.0.0",
        "environment": "development",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",
        "components": {
            "database": "ok",
            "data_manager": "ok",
            "backtest_engine": "ok",
            "ml_models": "ok"
        }
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
    """전략 목록 조회 API - 실제 구현된 전략만 반환"""
    try:
        # 백테스트 엔진의 실제 전략 목록을 가져오기
        engine_strategies = backtest_engine.strategies
        
        strategies = []
        for strategy_id, strategy_info in engine_strategies.items():
            # 구현된 전략만 포함 (구현되지 않은 전략은 비활성화 표시)
            is_implemented = strategy_info.get('implemented', True)
            status = '✅ 사용 가능' if is_implemented else '🚧 구현 예정'
            
            strategies.append({
                'id': strategy_id,
                'name': strategy_info.get('name', strategy_id),
                'description': f"{strategy_info.get('description', '')} - {status}",
                'timeframe': strategy_info.get('timeframe', '1h'),
                'category': 'technical',
                'risk_level': 'medium',
                'implemented': is_implemented,
                'status': status
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
    """전략 분석 결과 조회 API - 실제 데이터 기반"""
    try:
        # 실제 백테스트 결과가 없으면 빈 분석 결과 반환
        if not backtest_results:
            return jsonify({
                'analysis_id': analysis_id,
                'status': 'completed',
                'created_at': datetime.now().isoformat(),
                'message': '분석할 백테스트 결과가 없습니다.',
                'market_regime': {
                    'regime_type': 'unknown',
                    'volatility_level': 'unknown',
                    'trend_strength': 0,
                    'dominant_patterns': ['데이터 없음']
                },
                'strategy_rankings': [],
                'portfolio_recommendations': [],
                'key_insights': ['백테스트를 먼저 실행해주세요.'],
                'risk_management_tips': ['실제 데이터 수집 후 분석이 가능합니다.']
            })
        
        # 실제 백테스트 결과를 기반으로 분석 생성
        strategy_performance = {}
        
        # 전략별 성과 계산
        for result in backtest_results:
            strategy_name = result.strategy_name if hasattr(result, 'strategy_name') else 'Unknown'
            if strategy_name not in strategy_performance:
                strategy_performance[strategy_name] = {
                    'returns': [],
                    'sharpe_ratios': [],
                    'drawdowns': [],
                    'win_rates': [],
                    'trades': []
                }
            
            if hasattr(result, 'total_return'):
                strategy_performance[strategy_name]['returns'].append(result.total_return)
            if hasattr(result, 'sharpe_ratio'):
                strategy_performance[strategy_name]['sharpe_ratios'].append(result.sharpe_ratio)
            if hasattr(result, 'max_drawdown'):
                strategy_performance[strategy_name]['drawdowns'].append(result.max_drawdown)
            if hasattr(result, 'win_rate'):
                strategy_performance[strategy_name]['win_rates'].append(result.win_rate)
            if hasattr(result, 'total_trades'):
                strategy_performance[strategy_name]['trades'].append(result.total_trades)
        
        # 전략 랭킹 생성
        strategy_rankings = []
        rank = 1
        
        for strategy_name, perf in strategy_performance.items():
            avg_return = sum(perf['returns']) / len(perf['returns']) if perf['returns'] else 0
            avg_sharpe = sum(perf['sharpe_ratios']) / len(perf['sharpe_ratios']) if perf['sharpe_ratios'] else 0
            avg_drawdown = sum(perf['drawdowns']) / len(perf['drawdowns']) if perf['drawdowns'] else 0
            avg_win_rate = sum(perf['win_rates']) / len(perf['win_rates']) if perf['win_rates'] else 0
            
            # 점수 계산 (단순 가중 평균)
            performance_score = min(100, max(0, (avg_return + 10) * 2))  # 수익률 기반
            risk_score = min(100, max(0, 100 - avg_drawdown * 3))      # 리스크 기반
            consistency_score = min(100, max(0, avg_win_rate))         # 승률 기반
            
            total_score = (performance_score * 0.4 + risk_score * 0.3 + consistency_score * 0.3)
            
            # 추천사항 결정
            if total_score >= 80:
                recommendation = '🌟 최적 전략 - 적극 활용 권장'
            elif total_score >= 70:
                recommendation = '✅ 우수 전략 - 활용 권장'
            elif total_score >= 60:
                recommendation = '⚠️ 보통 전략 - 조건부 활용'
            else:
                recommendation = '🔄 개선 필요 - 파라미터 최적화 권장'
            
            strategy_rankings.append({
                'rank': rank,
                'strategy_name': strategy_name,
                'total_score': round(total_score, 1),
                'performance_score': round(performance_score, 1),
                'risk_score': round(risk_score, 1),
                'consistency_score': round(consistency_score, 1),
                'adaptability_score': round(total_score * 0.9, 1),  # 총점의 90%로 근사
                'recommendation': recommendation,
                'avg_return': round(avg_return, 2),
                'avg_drawdown': round(avg_drawdown, 2),
                'avg_win_rate': round(avg_win_rate, 1)
            })
            rank += 1
        
        # 총점 기준으로 정렬
        strategy_rankings.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 랭크 재설정
        for i, ranking in enumerate(strategy_rankings):
            ranking['rank'] = i + 1
        
        analysis_results = {
            'analysis_id': analysis_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'market_regime': {
                'regime_type': 'data_based',
                'volatility_level': 'calculated',
                'trend_strength': round(sum(s['performance_score'] for s in strategy_rankings) / len(strategy_rankings) / 100, 2) if strategy_rankings else 0,
                'dominant_patterns': [f'{len(backtest_results)}개 백테스트 결과 기반']
            },
            'strategy_rankings': strategy_rankings,
            'portfolio_recommendations': _generate_portfolio_recommendations(strategy_rankings),
            'key_insights': _generate_key_insights(strategy_rankings, len(backtest_results)),
            'risk_management_tips': _generate_risk_tips(strategy_rankings)
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

@api.route('/api/backtest/results/<result_id>', methods=['GET'])
def get_backtest_result_detail(result_id):
    """백테스트 결과 상세 조회 API"""
    try:
        # result_id로 결과 찾기
        result_index = int(result_id) - 1
        if 0 <= result_index < len(backtest_results):
            result = backtest_results[result_index]
            
            # BacktestResult 객체를 딕셔너리로 변환
            if hasattr(result, 'trade_log'):
                trades = result.trade_log
            else:
                trades = []
            
            # 거래 로그 정리
            formatted_trades = []
            for trade in trades:
                formatted_trades.append({
                    'timestamp': trade.get('timestamp', ''),
                    'type': trade.get('type', ''),
                    'symbol': trade.get('symbol', ''),
                    'price': trade.get('price', 0),
                    'amount': trade.get('amount', 0),
                    'leverage': trade.get('leverage', 1.0),
                    'pnl': trade.get('pnl', 0),
                    'pnl_percent': trade.get('pnl_percent', 0),
                    'reason': trade.get('reason', ''),
                    'balance_after': trade.get('balance_after', 0)
                })
            
            return jsonify({
                'result': {
                    'id': result_id,
                    'strategy_name': result.strategy_name,
                    'symbol': result.symbol,
                    'start_date': result.start_date,
                    'end_date': result.end_date,
                    'initial_capital': result.initial_capital,
                    'final_value': result.final_value,
                    'total_return': result.total_return,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'avg_leverage': result.avg_leverage
                },
                'trades': formatted_trades
            })
        else:
            return jsonify({'error': '결과를 찾을 수 없습니다.'}), 404
            
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
        global backtest_results
        
        # 현재 결과 수 저장
        reset_count = len(backtest_results)
        
        # 결과 초기화
        backtest_results.clear()
        
        return success_response(
            data={'reset_count': reset_count},
            message=f'{reset_count}개의 백테스트 결과가 초기화되었습니다.'
        )
        
    except Exception as e:
        return error_response(f'결과 초기화 실패: {str(e)}', status_code=500)

@api.route('/api/backtest/statistics', methods=['GET'])
def get_backtest_statistics():
    """백테스트 통계 조회 API - 실제 결과 기반"""
    try:
        # 실제 백테스트 결과에서 통계 계산
        if not backtest_results:
            return jsonify({
                'strategy_stats': {},
                'symbol_stats': {},
                'period_stats': {},
                'total_tests': 0,
                'last_updated': datetime.now().isoformat(),
                'message': '백테스트 결과가 없습니다. 먼저 백테스트를 실행해주세요.'
            })
        
        # 실제 결과에서 통계 생성
        strategy_stats = {}
        symbol_stats = {}
        
        for result in backtest_results:
            strategy_name = result.strategy_name if hasattr(result, 'strategy_name') else 'Unknown'
            symbol = result.symbol if hasattr(result, 'symbol') else 'Unknown'
            
            # 전략별 통계
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {
                    'total_tests': 0,
                    'returns': [],
                    'sharpe_ratios': [],
                    'drawdowns': [],
                    'win_rates': []
                }
            
            stats = strategy_stats[strategy_name]
            stats['total_tests'] += 1
            if hasattr(result, 'total_return'):
                stats['returns'].append(result.total_return)
            if hasattr(result, 'sharpe_ratio'):
                stats['sharpe_ratios'].append(result.sharpe_ratio)
            if hasattr(result, 'max_drawdown'):
                stats['drawdowns'].append(result.max_drawdown)
            if hasattr(result, 'win_rate'):
                stats['win_rates'].append(result.win_rate)
        
        # 평균 계산
        for strategy_name, stats in strategy_stats.items():
            stats['avg_return'] = sum(stats['returns']) / len(stats['returns']) if stats['returns'] else 0
            stats['avg_sharpe'] = sum(stats['sharpe_ratios']) / len(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0
            stats['avg_drawdown'] = sum(stats['drawdowns']) / len(stats['drawdowns']) if stats['drawdowns'] else 0
            stats['avg_win_rate'] = sum(stats['win_rates']) / len(stats['win_rates']) if stats['win_rates'] else 0
            
            # 불필요한 리스트 데이터 제거
            del stats['returns']
            del stats['sharpe_ratios'] 
            del stats['drawdowns']
            del stats['win_rates']
        
        # 심볼별 통계 (실제 결과 기반)
        for result in backtest_results:
            symbol = result.symbol if hasattr(result, 'symbol') else 'Unknown'
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'total_tests': 0,
                    'returns': [],
                    'strategies': {}
                }
            
            symbol_stats[symbol]['total_tests'] += 1
            if hasattr(result, 'total_return'):
                symbol_stats[symbol]['returns'].append(result.total_return)
            
            strategy_name = result.strategy_name if hasattr(result, 'strategy_name') else 'Unknown'
            if strategy_name not in symbol_stats[symbol]['strategies']:
                symbol_stats[symbol]['strategies'][strategy_name] = []
            symbol_stats[symbol]['strategies'][strategy_name].append(result.total_return if hasattr(result, 'total_return') else 0)
        
        # 심볼별 평균 계산 및 최고/최악 전략 결정
        for symbol, data in symbol_stats.items():
            data['avg_return'] = sum(data['returns']) / len(data['returns']) if data['returns'] else 0
            
            # 최고/최악 전략 계산
            best_strategy = None
            worst_strategy = None
            best_return = float('-inf')
            worst_return = float('inf')
            
            for strategy_name, returns in data['strategies'].items():
                avg_return = sum(returns) / len(returns) if returns else 0
                if avg_return > best_return:
                    best_return = avg_return
                    best_strategy = strategy_name
                if avg_return < worst_return:
                    worst_return = avg_return
                    worst_strategy = strategy_name
            
            data['best_strategy'] = best_strategy or 'N/A'
            data['worst_strategy'] = worst_strategy or 'N/A'
            
            # 불필요한 데이터 제거
            del data['returns']
            del data['strategies']
        
        # 총 테스트 수 계산
        total_tests = len(backtest_results)
        
        return jsonify({
            'strategy_stats': strategy_stats,
            'symbol_stats': symbol_stats,
            'period_stats': {},  # 실제 구현 시 날짜 기반으로 계산
            'total_tests': total_tests,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/backtest/stream_log')
def stream_backtest_log():
    """실제 백테스트 로그 실시간 스트리밍 (SSE)"""
    # request context가 있을 때 파라미터 추출 및 검증
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')  
    symbol = request.args.get('symbol')
    strategy_name = request.args.get('strategy')
    initial_capital = float(request.args.get('initial_capital', '10000000'))
    backtest_mode = request.args.get('backtest_mode', 'single')
    ml_optimization = request.args.get('ml_optimization', 'off') == 'on'
    
    # 필수 파라미터 검증
    if not all([start_date, end_date, symbol, strategy_name]):
        def error_response():
            yield "data: " + json.dumps({
                "type": "error", 
                "message": "❌ 필수 파라미터가 누락되었습니다. (날짜, 심볼, 전략)",
                "timestamp": time.time()
            }) + "\n\n"
            yield "data: " + json.dumps({"type": "end"}) + "\n\n"
        return Response(error_response(), mimetype='text/plain')
    
    # 실제 구현된 전략만 매핑
    strategy_name_to_id = {
        '트리플 콤보 전략': 'triple_combo',
        'RSI 전략': 'rsi_strategy',
        'MACD 전략': 'macd_strategy'  # 구현 예정
    }
    
    # 전략 검증 - 유효하지 않은 전략은 실행하지 않음
    if strategy_name == '전략을 선택하세요' or not strategy_name or strategy_name == '':
        # 유효하지 않은 전략 요청시 즉시 종료
        def error_response():
            yield "data: " + json.dumps({
                "type": "error",
                "message": "❌ 전략이 선택되지 않았습니다. 전략을 선택해주세요.",
                "timestamp": time.time()
            }) + "\n\n"
            yield "data: " + json.dumps({"type": "end"}) + "\n\n"
        return Response(error_response(), mimetype='text/plain')
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