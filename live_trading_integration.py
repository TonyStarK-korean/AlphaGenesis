#!/usr/bin/env python3
"""
🚀 실전 매매 시스템 통합
새로운 4가지 전략을 실전 매매에 안전하게 연동
"""

import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import warnings
import json
import logging

# ccxt를 optional로 만들기
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enhanced_strategies import (
    ComprehensiveStrategySystem,
    EnhancedStrategy1,
    EnhancedStrategy2,
    HourlyTradingStrategy
)

class LiveTradingManager:
    """
    실전 매매 관리자
    - 4가지 전략 통합 관리
    - 실시간 신호 생성
    - 리스크 관리
    - 포지션 관리
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.exchange = None
        self.strategies = {}
        self.positions = {}
        self.is_trading = False
        self.last_signal_time = {}
        
        # 리스크 관리 설정
        self.max_position_size = self.config.get('max_position_size', 0.02)  # 자본의 2%
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 일일 최대 손실 5%
        self.min_signal_interval = self.config.get('min_signal_interval', 3600)  # 1시간
        
        self._initialize_strategies()
        self._initialize_exchange()
    
    def _default_config(self):
        """기본 설정"""
        return {
            'api_key': '',
            'api_secret': '',
            'sandbox': True,  # 기본적으로 샌드박스 모드
            'initial_capital': 10000,
            'max_position_size': 0.02,
            'max_daily_loss': 0.05,
            'min_signal_interval': 3600,
            'enabled_strategies': ['strategy1_alpha', 'strategy2_alpha'],  # 알파 전략만 사용
            'trading_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'timeframe': '1h'
        }
    
    def _initialize_strategies(self):
        """전략 초기화"""
        try:
            # 4가지 전략 모두 초기화
            self.strategies = {
                'strategy1_basic': HourlyTradingStrategy(),
                'strategy1_alpha': EnhancedStrategy1(),
                'strategy2_basic': HourlyTradingStrategy(),
                'strategy2_alpha': EnhancedStrategy2()
            }
            
            # 각 전략별 마지막 신호 시간 초기화
            for strategy_name in self.strategies:
                self.last_signal_time[strategy_name] = {}
            
            logger.info("전략 초기화 완료")
            
        except Exception as e:
            logger.error(f"전략 초기화 실패: {e}")
            raise
    
    def _initialize_exchange(self):
        """거래소 연결 초기화"""
        try:
            if not self.config.get('api_key') or not self.config.get('api_secret'):
                logger.warning("API 키가 설정되지 않음 - 시뮬레이션 모드로 실행")
                return
            
            if not CCXT_AVAILABLE:
                logger.warning("CCXT 모듈이 없음 - 시뮬레이션 모드로 실행")
                return
            
            self.exchange = ccxt.binance({
                'apiKey': self.config['api_key'],
                'secret': self.config['api_secret'],
                'sandbox': self.config.get('sandbox', True),
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}  # 선물 거래
            })
            
            # 계정 정보 확인
            balance = self.exchange.fetch_balance()
            logger.info(f"거래소 연결 성공 - 잔고: {balance['USDT']['total']} USDT")
            
        except Exception as e:
            logger.error(f"거래소 연결 실패: {e}")
            self.exchange = None
    
    def get_latest_data(self, symbol, timeframe='1h', limit=200):
        """최신 시장 데이터 가져오기"""
        try:
            if not self.exchange:
                # 거래소 연결이 없으면 더미 데이터 반환
                return self._generate_dummy_data(limit)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"{symbol} 데이터 가져오기 실패: {e}")
            return None
    
    def _generate_dummy_data(self, limit=200):
        """테스트용 더미 데이터 생성"""
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1H')
        
        # 랜덤 워크 기반 가격 데이터
        np.random.seed(42)
        price = 45000
        prices = []
        
        for _ in range(limit):
            change = np.random.normal(0, 0.02) * price
            price += change
            prices.append(price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000, 10000) for _ in range(limit)]
        }, index=dates)
        
        return df
    
    def generate_live_signals(self, symbol):
        """실시간 신호 생성"""
        try:
            # 최신 데이터 가져오기
            df = self.get_latest_data(symbol)
            if df is None or len(df) < 200:
                return {}
            
            signals = {}
            current_time = datetime.now()
            
            # 활성화된 전략들에 대해서만 신호 생성
            enabled_strategies = self.config.get('enabled_strategies', list(self.strategies.keys()))
            
            for strategy_name in enabled_strategies:
                if strategy_name not in self.strategies:
                    continue
                
                try:
                    strategy = self.strategies[strategy_name]
                    
                    # 신호 생성 간격 확인
                    last_signal = self.last_signal_time[strategy_name].get(symbol, datetime.min)
                    if (current_time - last_signal).total_seconds() < self.min_signal_interval:
                        continue
                    
                    # 전략별 신호 생성
                    if strategy_name == 'strategy1_basic':
                        strategy_signals = strategy.strategy1_early_surge(df)
                    elif strategy_name == 'strategy2_basic':
                        strategy_signals = strategy.strategy2_pullback_surge(df)
                    else:
                        strategy_signals = strategy.generate_signals(df)
                    
                    # 최신 신호 확인
                    if (len(strategy_signals) > 0 and 
                        strategy_signals['signal'].iloc[-1] == 1 and
                        strategy_signals['confidence'].iloc[-1] > 0.7):  # 높은 신뢰도만
                        
                        signals[strategy_name] = {
                            'signal': strategy_signals['signal'].iloc[-1],
                            'confidence': strategy_signals['confidence'].iloc[-1],
                            'timestamp': current_time,
                            'strategy': strategy_signals.get('strategy', {}).iloc[-1] if 'strategy' in strategy_signals.columns else strategy_name
                        }
                        
                        # 마지막 신호 시간 업데이트
                        self.last_signal_time[strategy_name][symbol] = current_time
                        
                except Exception as e:
                    logger.error(f"{strategy_name} 신호 생성 실패: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"{symbol} 신호 생성 실패: {e}")
            return {}
    
    def check_risk_limits(self, symbol, signal):
        """리스크 제한 확인"""
        try:
            # 일일 손실 한도 확인
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl < -self.max_daily_loss:
                logger.warning(f"일일 손실 한도 초과: {daily_pnl:.2%}")
                return False
            
            # 포지션 크기 확인
            current_exposure = self._calculate_total_exposure()
            if current_exposure > 0.8:  # 총 노출도 80% 제한
                logger.warning(f"총 노출도 한도 초과: {current_exposure:.2%}")
                return False
            
            # 신호 신뢰도 확인
            if signal.get('confidence', 0) < 0.7:
                logger.info(f"신호 신뢰도 부족: {signal.get('confidence', 0):.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"리스크 검사 실패: {e}")
            return False
    
    def _calculate_daily_pnl(self):
        """일일 손익 계산"""
        # 실제로는 포지션 기록에서 계산
        # 여기서는 더미 값 반환
        return 0.0
    
    def _calculate_total_exposure(self):
        """총 노출도 계산"""
        # 실제로는 모든 포지션의 크기 합계
        # 여기서는 더미 값 반환
        return len(self.positions) * 0.02
    
    def execute_trade(self, symbol, signal):
        """거래 실행"""
        try:
            if not self.exchange:
                logger.info(f"[시뮬레이션] {symbol} 매수 신호: {signal}")
                return True
            
            # 포지션 크기 계산
            balance = self.exchange.fetch_balance()
            available_balance = balance['USDT']['free']
            position_size = available_balance * self.max_position_size
            
            # 현재 가격 가져오기
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            quantity = position_size / current_price
            
            # 최소 주문 크기 확인
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            if quantity < min_amount:
                logger.warning(f"주문 크기가 최소 한도보다 작음: {quantity} < {min_amount}")
                return False
            
            # 시장가 매수 주문
            order = self.exchange.create_market_buy_order(symbol, quantity)
            
            # 포지션 기록
            self.positions[symbol] = {
                'side': 'long',
                'size': quantity,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'strategy': signal.get('strategy', 'unknown'),
                'order_id': order['id']
            }
            
            logger.info(f"매수 주문 성공: {symbol} {quantity:.6f} @ {current_price}")
            return True
            
        except Exception as e:
            logger.error(f"거래 실행 실패: {e}")
            return False
    
    def monitor_positions(self):
        """포지션 모니터링 및 관리"""
        try:
            for symbol, position in self.positions.copy().items():
                # 현재 가격 가져오기
                ticker = self.exchange.fetch_ticker(symbol) if self.exchange else None
                current_price = ticker['last'] if ticker else position['entry_price'] * 1.05
                
                # 수익률 계산
                entry_price = position['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                # 손절/익절 조건 확인
                should_close = False
                close_reason = ""
                
                if pnl_pct <= -5:  # 5% 손절
                    should_close = True
                    close_reason = "stop_loss"
                elif pnl_pct >= 10:  # 10% 익절
                    should_close = True
                    close_reason = "take_profit"
                
                # 포지션 보유 시간 확인 (최대 24시간)
                hold_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
                if hold_time >= 24:
                    should_close = True
                    close_reason = "time_limit"
                
                if should_close:
                    self._close_position(symbol, close_reason, current_price, pnl_pct)
                    
        except Exception as e:
            logger.error(f"포지션 모니터링 실패: {e}")
    
    def _close_position(self, symbol, reason, current_price, pnl_pct):
        """포지션 청산"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            if self.exchange:
                # 실제 매도 주문
                order = self.exchange.create_market_sell_order(symbol, position['size'])
                logger.info(f"포지션 청산: {symbol} ({reason}) PnL: {pnl_pct:.2f}%")
            else:
                logger.info(f"[시뮬레이션] 포지션 청산: {symbol} ({reason}) PnL: {pnl_pct:.2f}%")
            
            # 포지션 기록에서 제거
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"포지션 청산 실패: {e}")
    
    def start_trading(self):
        """실전 매매 시작"""
        if self.is_trading:
            logger.warning("이미 거래가 진행 중입니다")
            return
        
        self.is_trading = True
        logger.info("실전 매매 시작")
        
        # 메인 트레이딩 루프
        def trading_loop():
            while self.is_trading:
                try:
                    # 각 심볼에 대해 신호 확인
                    for symbol in self.config.get('trading_symbols', ['BTC/USDT']):
                        signals = self.generate_live_signals(symbol)
                        
                        # 신호가 있고 리스크 조건을 만족하면 거래 실행
                        for strategy_name, signal in signals.items():
                            if (signal['signal'] == 1 and 
                                self.check_risk_limits(symbol, signal) and
                                symbol not in self.positions):
                                
                                logger.info(f"매수 신호 발생: {symbol} ({strategy_name}) 신뢰도: {signal['confidence']:.2f}")
                                self.execute_trade(symbol, signal)
                    
                    # 포지션 모니터링
                    self.monitor_positions()
                    
                    # 1분 대기
                    time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"트레이딩 루프 오류: {e}")
                    time.sleep(60)
        
        # 백그라운드 스레드에서 실행
        self.trading_thread = threading.Thread(target=trading_loop, daemon=True)
        self.trading_thread.start()
    
    def stop_trading(self):
        """실전 매매 중지"""
        if not self.is_trading:
            return
        
        self.is_trading = False
        logger.info("실전 매매 중지 요청")
        
        # 모든 포지션 청산
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, "manual_stop", 0, 0)
        
        logger.info("실전 매매 완전 중지")
    
    def get_status(self):
        """현재 상태 조회"""
        return {
            'is_trading': self.is_trading,
            'positions': len(self.positions),
            'position_details': self.positions,
            'enabled_strategies': self.config.get('enabled_strategies', []),
            'trading_symbols': self.config.get('trading_symbols', []),
            'last_update': datetime.now().isoformat()
        }

# 전역 인스턴스
live_trading_manager = None

def get_live_trading_manager(config=None):
    """라이브 트레이딩 매니저 싱글톤 인스턴스"""
    global live_trading_manager
    if live_trading_manager is None:
        live_trading_manager = LiveTradingManager(config)
    return live_trading_manager

if __name__ == "__main__":
    print("🚀 실전 매매 시스템 통합")
    
    # 테스트 설정
    config = {
        'sandbox': True,
        'initial_capital': 10000,
        'enabled_strategies': ['strategy1_alpha', 'strategy2_alpha'],
        'trading_symbols': ['BTC/USDT'],
        'max_position_size': 0.01  # 1% 리스크
    }
    
    # 매니저 생성
    manager = LiveTradingManager(config)
    
    # 상태 확인
    print("매니저 상태:", manager.get_status())
    
    # 시뮬레이션 신호 테스트
    signals = manager.generate_live_signals('BTC/USDT')
    print("생성된 신호:", signals)