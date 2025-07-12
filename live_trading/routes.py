"""
실전매매 API 라우트
"""

from flask import Blueprint, jsonify, request, Response
import json
import asyncio
from datetime import datetime
import logging
import os

from live_trading.binance_testnet import BinanceTestnetTrader
from ml.models.risk_manager import DynamicRiskManager

logger = logging.getLogger(__name__)

live_trading_api = Blueprint('live_trading', __name__)

# 전역 인스턴스
trader_instance = None
risk_manager_instance = None

def get_trader():
    """트레이더 인스턴스 가져오기"""
    global trader_instance
    if trader_instance is None:
        trader_instance = BinanceTestnetTrader()
    return trader_instance

def get_risk_manager():
    """리스크 매니저 인스턴스 가져오기"""
    global risk_manager_instance
    if risk_manager_instance is None:
        risk_manager_instance = DynamicRiskManager()
    return risk_manager_instance

@live_trading_api.route('/api/live-trading/status', methods=['GET'])
async def get_trading_status():
    """실전매매 상태 조회"""
    try:
        trader = get_trader()
        
        async with trader:
            status = await trader.get_trading_status()
        
        return jsonify({
            'status': 'success',
            'data': status,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"거래 상태 조회 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/start', methods=['POST'])
async def start_trading():
    """실전매매 시작"""
    try:
        data = request.get_json() or {}
        
        # 환경변수 확인
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        secret_key = os.getenv('BINANCE_TESTNET_SECRET_KEY')
        
        if not api_key or not secret_key:
            return jsonify({
                'status': 'error',
                'message': '바이낸스 테스트넷 API 키가 설정되지 않았습니다. 환경변수를 확인하세요.',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        trader = get_trader()
        
        # 비동기로 거래 시작
        asyncio.create_task(start_trading_async(trader))
        
        return jsonify({
            'status': 'success',
            'message': '실전매매가 시작되었습니다',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"실전매매 시작 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

async def start_trading_async(trader):
    """비동기 거래 시작"""
    try:
        async with trader:
            await trader.start_live_trading()
    except Exception as e:
        logger.error(f"비동기 거래 실행 실패: {e}")

@live_trading_api.route('/api/live-trading/stop', methods=['POST'])
def stop_trading():
    """실전매매 중지"""
    try:
        trader = get_trader()
        trader.stop_live_trading()
        
        return jsonify({
            'status': 'success',
            'message': '실전매매가 중지되었습니다',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"실전매매 중지 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/positions', methods=['GET'])
async def get_positions():
    """현재 포지션 조회"""
    try:
        trader = get_trader()
        
        async with trader:
            positions = await trader.get_positions()
        
        return jsonify({
            'status': 'success',
            'data': {
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'leverage': pos.leverage,
                        'margin': pos.margin
                    }
                    for pos in positions
                ],
                'total_positions': len(positions)
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"포지션 조회 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/balance', methods=['GET'])
async def get_balance():
    """잔고 조회"""
    try:
        trader = get_trader()
        
        async with trader:
            balance = await trader.get_balance()
        
        return jsonify({
            'status': 'success',
            'data': {
                'balance': balance,
                'currency': 'USDT'
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"잔고 조회 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/close-position', methods=['POST'])
async def close_position():
    """포지션 청산"""
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                'status': 'error',
                'message': '심볼이 필요합니다',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        symbol = data['symbol']
        trader = get_trader()
        
        async with trader:
            success = await trader.close_position(symbol)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'{symbol} 포지션이 청산되었습니다',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'{symbol} 포지션 청산에 실패했습니다',
                'timestamp': datetime.now().isoformat()
            }), 400
    
    except Exception as e:
        logger.error(f"포지션 청산 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/risk-analysis', methods=['GET'])
async def get_risk_analysis():
    """리스크 분석 조회"""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        
        risk_manager = get_risk_manager()
        
        # 시장 데이터 기반 리스크 파라미터 계산
        risk_params = await risk_manager.calculate_optimal_risk_params(symbol, 'triple_combo')
        
        return jsonify({
            'status': 'success',
            'data': {
                'symbol': symbol,
                'risk_parameters': {
                    'max_position_size': risk_params.max_position_size,
                    'stop_loss_pct': risk_params.stop_loss_pct,
                    'take_profit_pct': risk_params.take_profit_pct,
                    'max_leverage': risk_params.max_leverage,
                    'risk_score': risk_params.risk_score,
                    'recommended_allocation': risk_params.recommended_allocation
                }
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"리스크 분석 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/train-risk-models', methods=['POST'])
async def train_risk_models():
    """ML 리스크 모델 훈련"""
    try:
        data = request.get_json() or {}
        lookback_days = data.get('lookback_days', 365)
        
        risk_manager = get_risk_manager()
        
        # 비동기로 모델 훈련 시작
        training_result = await risk_manager.train_risk_models(lookback_days)
        
        return jsonify({
            'status': 'success',
            'message': 'ML 리스크 모델 훈련이 완료되었습니다',
            'data': training_result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"ML 모델 훈련 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/market-signals', methods=['GET'])
async def get_market_signals():
    """시장 신호 분석"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        
        trader = get_trader()
        
        async with trader:
            signals = await trader.analyze_market_signals(symbol)
        
        return jsonify({
            'status': 'success',
            'data': signals,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"시장 신호 분석 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@live_trading_api.route('/api/live-trading/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'success',
        'message': '실전매매 서비스가 정상 작동 중입니다',
        'timestamp': datetime.now().isoformat()
    })

@live_trading_api.route('/api/live-trading/config', methods=['GET'])
def get_config():
    """실전매매 설정 조회"""
    try:
        trader = get_trader()
        
        config = {
            'trading_symbols': trader.trading_symbols,
            'max_positions': trader.max_positions,
            'position_size_ratio': trader.position_size_ratio,
            'base_url': trader.base_url,
            'api_configured': bool(trader.api_key and trader.secret_key)
        }
        
        return jsonify({
            'status': 'success',
            'data': config,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"설정 조회 실패: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500