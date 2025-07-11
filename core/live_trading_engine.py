#!/usr/bin/env python3
"""
실전매매 엔진
실시간 자동 트레이딩 시스템
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 프로젝트 모듈
from exchange.binance_futures_api import BinanceFuturesAPI
from ml.models.price_prediction_model import PricePredictionModel
from config.unified_config import config
from core.risk_management import RiskManager
from core.position_management import PositionManager
from notification.telegram_notification_bot import TelegramBot

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingStatus(Enum):
    """거래 상태"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"

@dataclass
class TradingSignal:
    """거래 신호 데이터 클래스"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    strategy: str
    price: float
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """포지션 데이터 클래스"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Trade:
    """거래 데이터 클래스"""
    symbol: str
    side: str
    order_type: OrderType
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    order_id: str = ""
    status: str = "filled"

class LiveTradingEngine:
    """실전매매 엔진 클래스"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        초기화
        
        Args:
            api_key: 바이낸스 API 키
            api_secret: 바이낸스 API 시크릿
            testnet: 테스트넷 사용 여부
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # 거래 상태
        self.status = TradingStatus.STOPPED
        self.start_time = None
        self.last_update = None
        
        # 거래 설정
        self.trading_config = config.get_config('trading')
        self.initial_capital = self.trading_config['initial_capital']
        self.current_capital = self.initial_capital
        
        # 거래 컴포넌트
        self.exchange = BinanceFuturesAPI(api_key, api_secret, testnet)
        self.ml_model = PricePredictionModel()
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager()
        self.telegram_bot = TelegramBot()
        
        # 거래 데이터
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.active_orders: Dict[str, Dict] = {}
        
        # 성능 지표
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'daily_pnl': []
        }
        
        # 심볼 및 전략 설정
        self.active_symbols = []
        self.strategy_settings = {
            'confidence_threshold': 0.6,
            'max_positions': 5,
            'position_size_ratio': 0.1,
            'stop_loss_ratio': 0.02,
            'take_profit_ratio': 0.05
        }
        
        # 비동기 태스크
        self.running_tasks = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"실전매매 엔진 초기화 완료 (테스트넷: {testnet})")
    
    async def initialize(self):
        """
        엔진 초기화
        """
        try:
            logger.info("실전매매 엔진 초기화 중...")
            
            # 거래소 연결 확인
            await self.exchange.get_usdt_perpetual_symbols()
            
            # ML 모델 로드
            await self.ml_model.load_latest_model()
            
            # 텔레그램 봇 초기화
            await self.telegram_bot.initialize()
            
            # 기존 포지션 로드
            await self.load_existing_positions()
            
            logger.info("실전매매 엔진 초기화 완료")
            
        except Exception as e:
            logger.error(f"엔진 초기화 실패: {e}")
            raise
    
    async def start_trading(self, symbols: List[str], strategy_config: Dict = None):
        """
        거래 시작
        
        Args:
            symbols: 거래할 심볼 리스트
            strategy_config: 전략 설정
        """
        try:
            if self.status == TradingStatus.RUNNING:
                logger.warning("이미 거래가 실행 중입니다.")
                return
            
            logger.info("실전매매 시작...")
            
            # 설정 업데이트
            self.active_symbols = symbols
            if strategy_config:
                self.strategy_settings.update(strategy_config)
            
            # 상태 업데이트
            self.status = TradingStatus.RUNNING
            self.start_time = datetime.now()
            
            # 거래 루프 시작
            await self.start_trading_loops()
            
            # 텔레그램 알림
            await self.telegram_bot.send_message(
                f"🚀 실전매매 시작\n"
                f"심볼: {', '.join(symbols)}\n"
                f"초기 자본: {self.initial_capital:,.0f}원"
            )
            
        except Exception as e:
            logger.error(f"거래 시작 실패: {e}")
            self.status = TradingStatus.ERROR
            raise
    
    async def stop_trading(self):
        """
        거래 중지
        """
        try:
            logger.info("실전매매 중지 중...")
            
            # 상태 업데이트
            self.status = TradingStatus.STOPPED
            
            # 실행 중인 태스크 정리
            for task in self.running_tasks:
                task.cancel()
            
            # 모든 포지션 청산 (옵션)
            # await self.close_all_positions()
            
            # 텔레그램 알림
            await self.telegram_bot.send_message(
                f"⏹️ 실전매매 중지\n"
                f"총 거래 수: {self.performance_metrics['total_trades']}\n"
                f"총 손익: {self.performance_metrics['total_pnl']:,.0f}원"
            )
            
            logger.info("실전매매 중지 완료")
            
        except Exception as e:
            logger.error(f"거래 중지 실패: {e}")
    
    async def start_trading_loops(self):
        """
        거래 루프 시작
        """
        try:
            # 메인 거래 루프
            main_loop = asyncio.create_task(self.main_trading_loop())
            self.running_tasks.append(main_loop)
            
            # 위험 관리 루프
            risk_loop = asyncio.create_task(self.risk_management_loop())
            self.running_tasks.append(risk_loop)
            
            # 포지션 모니터링 루프
            position_loop = asyncio.create_task(self.position_monitoring_loop())
            self.running_tasks.append(position_loop)
            
            # 성능 업데이트 루프
            performance_loop = asyncio.create_task(self.performance_update_loop())
            self.running_tasks.append(performance_loop)
            
            logger.info("모든 거래 루프 시작 완료")
            
        except Exception as e:
            logger.error(f"거래 루프 시작 실패: {e}")
            raise
    
    async def main_trading_loop(self):
        """
        메인 거래 루프
        """
        try:
            while self.status == TradingStatus.RUNNING:
                try:
                    # 각 심볼에 대해 거래 신호 생성
                    for symbol in self.active_symbols:
                        if self.status != TradingStatus.RUNNING:
                            break
                        
                        # 시장 데이터 조회
                        market_data = await self.exchange.get_ohlcv_data(symbol, '1h', 100)
                        
                        if market_data.empty:
                            continue
                        
                        # 거래 신호 생성
                        signal = await self.generate_trading_signal(symbol, market_data)
                        
                        if signal and signal.action != 'HOLD':
                            # 거래 실행
                            await self.execute_signal(signal)
                        
                        # API 제한 방지
                        await asyncio.sleep(1)
                    
                    # 메인 루프 대기 (30초)
                    await asyncio.sleep(30)
                    
                except Exception as e:
                    logger.error(f"메인 거래 루프 오류: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("메인 거래 루프 종료")
        except Exception as e:
            logger.error(f"메인 거래 루프 실패: {e}")
            self.status = TradingStatus.ERROR
    
    async def generate_trading_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        거래 신호 생성
        
        Args:
            symbol: 심볼
            market_data: 시장 데이터
            
        Returns:
            거래 신호 또는 None
        """
        try:
            # ML 예측
            prediction = await self.ml_model.predict(market_data)
            
            # 기술적 분석
            current_price = market_data['close'].iloc[-1]
            rsi = market_data['rsi'].iloc[-1]
            macd = market_data['macd'].iloc[-1]
            macd_signal = market_data['macd_signal'].iloc[-1]
            
            # 트리플 콤보 전략
            confidence = 0.0
            action = 'HOLD'
            
            # 상승 신호
            if (prediction > 0.01 and rsi < 70 and macd > macd_signal):
                confidence = 0.7
                action = 'BUY'
            
            # 하락 신호
            elif (prediction < -0.01 and rsi > 30 and macd < macd_signal):
                confidence = 0.7
                action = 'SELL'
            
            # 신뢰도 임계값 확인
            if confidence < self.strategy_settings['confidence_threshold']:
                action = 'HOLD'
            
            # 거래 신호 생성
            if action != 'HOLD':
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    strategy='triple_combo',
                    price=current_price,
                    quantity=self.calculate_position_size(symbol, current_price),
                    stop_loss=self.calculate_stop_loss(current_price, action),
                    take_profit=self.calculate_take_profit(current_price, action)
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"{symbol} 거래 신호 생성 실패: {e}")
            return None
    
    async def execute_signal(self, signal: TradingSignal):
        """
        거래 신호 실행
        
        Args:
            signal: 거래 신호
        """
        try:
            # 위험 관리 검사
            if not self.risk_manager.check_signal(signal, self.positions, self.current_capital):
                logger.warning(f"{signal.symbol} 거래 신호 위험 관리 실패")
                return
            
            # 기존 포지션 확인
            existing_position = self.positions.get(signal.symbol)
            
            if signal.action == 'BUY':
                if existing_position and existing_position.side == PositionSide.SHORT:
                    # 숏 포지션 청산
                    await self.close_position(signal.symbol)
                
                # 롱 포지션 오픈
                await self.open_position(signal, PositionSide.LONG)
                
            elif signal.action == 'SELL':
                if existing_position and existing_position.side == PositionSide.LONG:
                    # 롱 포지션 청산
                    await self.close_position(signal.symbol)
                
                # 숏 포지션 오픈
                await self.open_position(signal, PositionSide.SHORT)
            
            # 성능 지표 업데이트
            self.performance_metrics['total_trades'] += 1
            
            logger.info(f"{signal.symbol} 거래 신호 실행 완료: {signal.action}")
            
        except Exception as e:
            logger.error(f"거래 신호 실행 실패: {e}")
    
    async def open_position(self, signal: TradingSignal, side: PositionSide):
        """
        포지션 오픈
        
        Args:
            signal: 거래 신호
            side: 포지션 방향
        """
        try:
            # 시장가 주문 실행 (테스트넷에서는 시뮬레이션)
            if self.testnet:
                # 테스트넷 시뮬레이션
                order_result = {
                    'orderId': f"test_{int(time.time())}",
                    'symbol': signal.symbol,
                    'side': 'BUY' if side == PositionSide.LONG else 'SELL',
                    'executedQty': signal.quantity,
                    'price': signal.price,
                    'commission': signal.price * signal.quantity * 0.001  # 0.1% 수수료
                }
            else:
                # 실제 주문 실행
                order_result = await self.exchange.create_market_order(
                    signal.symbol,
                    'BUY' if side == PositionSide.LONG else 'SELL',
                    signal.quantity
                )
            
            # 포지션 생성
            position = Position(
                symbol=signal.symbol,
                side=side,
                size=float(order_result['executedQty']),
                entry_price=float(order_result['price']),
                current_price=float(order_result['price']),
                unrealized_pnl=0.0
            )
            
            self.positions[signal.symbol] = position
            
            # 거래 기록
            trade = Trade(
                symbol=signal.symbol,
                side=order_result['side'],
                order_type=OrderType.MARKET,
                quantity=float(order_result['executedQty']),
                price=float(order_result['price']),
                commission=float(order_result.get('commission', 0)),
                timestamp=datetime.now(),
                order_id=order_result['orderId']
            )
            
            self.trades.append(trade)
            
            # 스탑로스, 테이크프로핏 주문 설정
            if signal.stop_loss > 0:
                await self.set_stop_loss(signal.symbol, signal.stop_loss)
            
            if signal.take_profit > 0:
                await self.set_take_profit(signal.symbol, signal.take_profit)
            
            # 텔레그램 알림
            await self.telegram_bot.send_message(
                f"📈 포지션 오픈\n"
                f"심볼: {signal.symbol}\n"
                f"방향: {side.value}\n"
                f"수량: {signal.quantity}\n"
                f"가격: {signal.price:,.2f}\n"
                f"신뢰도: {signal.confidence:.2f}"
            )
            
        except Exception as e:
            logger.error(f"포지션 오픈 실패: {e}")
    
    async def close_position(self, symbol: str):
        """
        포지션 청산
        
        Args:
            symbol: 심볼
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            # 반대 방향 시장가 주문
            side = 'SELL' if position.side == PositionSide.LONG else 'BUY'
            
            if self.testnet:
                # 테스트넷 시뮬레이션
                current_price = await self.exchange.get_realtime_price(symbol)
                order_result = {
                    'orderId': f"test_close_{int(time.time())}",
                    'symbol': symbol,
                    'side': side,
                    'executedQty': position.size,
                    'price': current_price['price'],
                    'commission': current_price['price'] * position.size * 0.001
                }
            else:
                # 실제 주문 실행
                order_result = await self.exchange.create_market_order(
                    symbol,
                    side,
                    position.size
                )
            
            # 손익 계산
            exit_price = float(order_result['price'])
            if position.side == PositionSide.LONG:
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
            
            # 수수료 차감
            commission = float(order_result.get('commission', 0))
            net_pnl = pnl - commission
            
            # 거래 기록
            trade = Trade(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position.size,
                price=exit_price,
                commission=commission,
                timestamp=datetime.now(),
                order_id=order_result['orderId']
            )
            
            self.trades.append(trade)
            
            # 성능 지표 업데이트
            self.performance_metrics['total_pnl'] += net_pnl
            self.current_capital += net_pnl
            
            if net_pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
            
            # 포지션 제거
            del self.positions[symbol]
            
            # 텔레그램 알림
            await self.telegram_bot.send_message(
                f"📉 포지션 청산\n"
                f"심볼: {symbol}\n"
                f"방향: {position.side.value}\n"
                f"진입가: {position.entry_price:,.2f}\n"
                f"청산가: {exit_price:,.2f}\n"
                f"손익: {net_pnl:,.0f}원"
            )
            
        except Exception as e:
            logger.error(f"포지션 청산 실패: {e}")
    
    async def risk_management_loop(self):
        """
        위험 관리 루프
        """
        try:
            while self.status == TradingStatus.RUNNING:
                try:
                    # 전체 포트폴리오 위험 검사
                    await self.check_portfolio_risk()
                    
                    # 개별 포지션 위험 검사
                    for symbol, position in self.positions.items():
                        await self.check_position_risk(symbol, position)
                    
                    # 위험 관리 루프 대기 (10초)
                    await asyncio.sleep(10)
                    
                except Exception as e:
                    logger.error(f"위험 관리 루프 오류: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("위험 관리 루프 종료")
        except Exception as e:
            logger.error(f"위험 관리 루프 실패: {e}")
    
    async def position_monitoring_loop(self):
        """
        포지션 모니터링 루프
        """
        try:
            while self.status == TradingStatus.RUNNING:
                try:
                    # 포지션 가격 업데이트
                    for symbol, position in self.positions.items():
                        await self.update_position_price(symbol, position)
                    
                    # 포지션 모니터링 루프 대기 (5초)
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"포지션 모니터링 루프 오류: {e}")
                    await asyncio.sleep(2)
                    
        except asyncio.CancelledError:
            logger.info("포지션 모니터링 루프 종료")
        except Exception as e:
            logger.error(f"포지션 모니터링 루프 실패: {e}")
    
    async def performance_update_loop(self):
        """
        성능 지표 업데이트 루프
        """
        try:
            while self.status == TradingStatus.RUNNING:
                try:
                    # 성능 지표 업데이트
                    await self.update_performance_metrics()
                    
                    # 성능 업데이트 루프 대기 (60초)
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"성능 업데이트 루프 오류: {e}")
                    await asyncio.sleep(30)
                    
        except asyncio.CancelledError:
            logger.info("성능 업데이트 루프 종료")
        except Exception as e:
            logger.error(f"성능 업데이트 루프 실패: {e}")
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            symbol: 심볼
            price: 가격
            
        Returns:
            포지션 크기
        """
        try:
            # 현재 자본의 일정 비율로 포지션 크기 계산
            position_value = self.current_capital * self.strategy_settings['position_size_ratio']
            quantity = position_value / price
            
            # 최소/최대 수량 제한
            min_quantity = 0.001  # 최소 수량
            max_quantity = self.current_capital * 0.2 / price  # 최대 20%
            
            return max(min_quantity, min(quantity, max_quantity))
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 실패: {e}")
            return 0.001
    
    def calculate_stop_loss(self, price: float, action: str) -> float:
        """
        손절가 계산
        
        Args:
            price: 현재 가격
            action: 거래 방향
            
        Returns:
            손절가
        """
        try:
            stop_loss_ratio = self.strategy_settings['stop_loss_ratio']
            
            if action == 'BUY':
                return price * (1 - stop_loss_ratio)
            else:  # SELL
                return price * (1 + stop_loss_ratio)
                
        except Exception as e:
            logger.error(f"손절가 계산 실패: {e}")
            return price
    
    def calculate_take_profit(self, price: float, action: str) -> float:
        """
        익절가 계산
        
        Args:
            price: 현재 가격
            action: 거래 방향
            
        Returns:
            익절가
        """
        try:
            take_profit_ratio = self.strategy_settings['take_profit_ratio']
            
            if action == 'BUY':
                return price * (1 + take_profit_ratio)
            else:  # SELL
                return price * (1 - take_profit_ratio)
                
        except Exception as e:
            logger.error(f"익절가 계산 실패: {e}")
            return price
    
    async def get_status(self) -> Dict[str, Any]:
        """
        현재 상태 반환
        
        Returns:
            상태 정보
        """
        try:
            return {
                'status': self.status.value,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'current_capital': self.current_capital,
                'active_symbols': self.active_symbols,
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'performance_metrics': self.performance_metrics,
                'strategy_settings': self.strategy_settings
            }
            
        except Exception as e:
            logger.error(f"상태 조회 실패: {e}")
            return {}
    
    async def update_settings(self, new_settings: Dict[str, Any]):
        """
        설정 업데이트
        
        Args:
            new_settings: 새로운 설정
        """
        try:
            self.strategy_settings.update(new_settings)
            logger.info(f"설정 업데이트 완료: {new_settings}")
            
        except Exception as e:
            logger.error(f"설정 업데이트 실패: {e}")

# 테스트 코드
async def test_live_trading():
    """실전매매 엔진 테스트"""
    try:
        print("🚀 실전매매 엔진 테스트 시작")
        
        # 엔진 생성 (테스트넷)
        engine = LiveTradingEngine("", "", testnet=True)
        
        # 초기화
        await engine.initialize()
        
        # 거래 시작
        test_symbols = ['BTC/USDT', 'ETH/USDT']
        await engine.start_trading(test_symbols)
        
        # 5분 후 중지
        await asyncio.sleep(300)
        await engine.stop_trading()
        
        print("🎉 실전매매 엔진 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    # 비동기 테스트 실행
    asyncio.run(test_live_trading())