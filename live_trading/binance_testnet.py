"""
바이낸스 테스트넷 실전매매 시스템
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import hmac
import hashlib
import time
import aiohttp

from core.data_manager import DataManager
from core.dynamic_leverage import DynamicLeverageManager
from ml.models.risk_manager import DynamicRiskManager

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float
    margin: float
    timestamp: datetime

@dataclass
class Trade:
    """거래 정보"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    type: str  # 'MARKET' or 'LIMIT'
    quantity: float
    price: float
    leverage: float
    timestamp: datetime
    order_id: str
    status: str

class BinanceTestnetTrader:
    """바이낸스 테스트넷 실전매매 시스템"""
    
    def __init__(self):
        # 환경변수에서 API 키 로드
        self.api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        self.secret_key = os.getenv('BINANCE_TESTNET_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("바이낸스 테스트넷 API 키가 설정되지 않았습니다. 환경변수를 확인하세요.")
        
        # 테스트넷 URL
        self.base_url = "https://testnet.binancefuture.com"
        self.api_url = f"{self.base_url}/fapi/v1"
        
        # 핵심 모듈 초기화
        self.data_manager = DataManager()
        self.leverage_manager = DynamicLeverageManager()
        self.risk_manager = DynamicRiskManager()
        
        # 거래 상태
        self.positions: Dict[str, Position] = {}
        self.orders: List[Trade] = []
        self.is_trading = False
        self.balance = 10000.0  # 초기 잔고 (USDT)
        
        # 거래 설정
        self.trading_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.max_positions = 3
        self.position_size_ratio = 0.1  # 포지션당 잔고의 10%
        
        # 세션
        self.session = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params: str) -> str:
        """API 서명 생성"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self) -> Dict[str, str]:
        """API 헤더 생성"""
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """API 요청 실행"""
        url = f"{self.api_url}/{endpoint}"
        headers = self._get_headers()
        
        if params is None:
            params = {}
        
        if signed:
            # 타임스탬프 추가
            params['timestamp'] = int(time.time() * 1000)
            
            # 쿼리 스트링 생성
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            
            # 서명 생성
            signature = self._generate_signature(query_string)
            params['signature'] = signature
        
        try:
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await response.json()
            elif method == 'POST':
                async with self.session.post(url, json=params, headers=headers) as response:
                    return await response.json()
            elif method == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"API 요청 실패: {e}")
            return {'error': str(e)}
    
    async def get_account_info(self) -> Dict:
        """계정 정보 조회"""
        return await self._make_request('GET', 'account', signed=True)
    
    async def get_balance(self) -> float:
        """USDT 잔고 조회"""
        account_info = await self.get_account_info()
        
        if 'assets' in account_info:
            for asset in account_info['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
        
        return self.balance
    
    async def get_positions(self) -> List[Position]:
        """현재 포지션 조회"""
        response = await self._make_request('GET', 'positionRisk', signed=True)
        
        positions = []
        if isinstance(response, list):
            for pos_data in response:
                if float(pos_data['positionAmt']) != 0:
                    position = Position(
                        symbol=pos_data['symbol'],
                        side='LONG' if float(pos_data['positionAmt']) > 0 else 'SHORT',
                        size=abs(float(pos_data['positionAmt'])),
                        entry_price=float(pos_data['entryPrice']),
                        current_price=float(pos_data['markPrice']),
                        unrealized_pnl=float(pos_data['unRealizedProfit']),
                        leverage=float(pos_data['leverage']),
                        margin=float(pos_data['isolatedMargin']),
                        timestamp=datetime.now()
                    )
                    positions.append(position)
        
        return positions
    
    async def get_market_price(self, symbol: str) -> float:
        """현재 시장가 조회"""
        response = await self._make_request('GET', 'ticker/price', {'symbol': symbol})
        
        if 'price' in response:
            return float(response['price'])
        
        return 0.0
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """레버리지 설정"""
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        
        response = await self._make_request('POST', 'leverage', params, signed=True)
        
        return 'leverage' in response
    
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = 'MARKET', price: float = None) -> Optional[Trade]:
        """주문 실행"""
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        if order_type == 'LIMIT' and price:
            params['price'] = price
            params['timeInForce'] = 'GTC'
        
        response = await self._make_request('POST', 'order', params, signed=True)
        
        if 'orderId' in response:
            trade = Trade(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=float(response.get('price', price or 0)),
                leverage=1.0,  # 현재 레버리지로 업데이트 필요
                timestamp=datetime.now(),
                order_id=str(response['orderId']),
                status=response.get('status', 'NEW')
            )
            
            self.orders.append(trade)
            logger.info(f"주문 실행: {symbol} {side} {quantity} at {trade.price}")
            
            return trade
        
        logger.error(f"주문 실패: {response}")
        return None
    
    async def close_position(self, symbol: str) -> bool:
        """포지션 청산"""
        positions = await self.get_positions()
        
        for position in positions:
            if position.symbol == symbol:
                # 반대 주문으로 청산
                side = 'SELL' if position.side == 'LONG' else 'BUY'
                
                trade = await self.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=position.size,
                    order_type='MARKET'
                )
                
                return trade is not None
        
        return False
    
    async def analyze_market_signals(self, symbol: str) -> Dict[str, Any]:
        """시장 신호 분석"""
        try:
            # 최근 데이터 로드
            data = self.data_manager.load_data(symbol.replace('USDT', '/USDT'), '1h')
            
            if data.empty or len(data) < 50:
                return {'signal': 'HOLD', 'strength': 0.0, 'reason': '데이터 부족'}
            
            # 기술적 지표 계산
            signals = self._calculate_technical_signals(data)
            
            # 동적 레버리지 계산
            optimal_leverage = self.leverage_manager.calculate_leverage(data)
            
            # ML 기반 리스크 평가
            risk_score = await self.risk_manager.assess_risk(data, symbol)
            
            return {
                'signal': signals['main_signal'],
                'strength': signals['signal_strength'],
                'leverage': optimal_leverage,
                'risk_score': risk_score,
                'technical_signals': signals,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"시장 신호 분석 실패 ({symbol}): {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'reason': f'분석 실패: {e}'}
    
    def _calculate_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기술적 지표 기반 신호 계산"""
        close = data['close']
        
        # RSI 계산
        rsi = self._calculate_rsi(close, 14)
        current_rsi = rsi.iloc[-1]
        
        # MACD 계산
        macd_line, macd_signal, macd_histogram = self._calculate_macd(close)
        macd_cross = macd_line.iloc[-1] > macd_signal.iloc[-1]
        
        # 볼린저 밴드 계산
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20)
        current_price = close.iloc[-1]
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # 신호 종합
        signals = []
        
        # RSI 신호
        if current_rsi < 30:
            signals.append(('BUY', 0.7))
        elif current_rsi > 70:
            signals.append(('SELL', 0.7))
        
        # MACD 신호
        if macd_cross and macd_line.iloc[-1] > 0:
            signals.append(('BUY', 0.6))
        elif not macd_cross and macd_line.iloc[-1] < 0:
            signals.append(('SELL', 0.6))
        
        # 볼린저 밴드 신호
        if bb_position < 0.2:
            signals.append(('BUY', 0.5))
        elif bb_position > 0.8:
            signals.append(('SELL', 0.5))
        
        # 신호 집계
        buy_strength = sum([strength for signal, strength in signals if signal == 'BUY'])
        sell_strength = sum([strength for signal, strength in signals if signal == 'SELL'])
        
        if buy_strength > sell_strength and buy_strength > 0.5:
            main_signal = 'BUY'
            signal_strength = buy_strength
        elif sell_strength > buy_strength and sell_strength > 0.5:
            main_signal = 'SELL'
            signal_strength = sell_strength
        else:
            main_signal = 'HOLD'
            signal_strength = 0.0
        
        return {
            'main_signal': main_signal,
            'signal_strength': signal_strength,
            'rsi': current_rsi,
            'macd_cross': macd_cross,
            'bb_position': bb_position,
            'individual_signals': signals
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    async def execute_trading_cycle(self) -> Dict[str, Any]:
        """거래 사이클 실행"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'actions': [],
            'errors': []
        }
        
        try:
            # 잔고 확인
            current_balance = await self.get_balance()
            
            # 현재 포지션 확인
            current_positions = await self.get_positions()
            
            # 각 심볼별 신호 분석
            for symbol in self.trading_symbols:
                signal_data = await self.analyze_market_signals(symbol)
                results['signals'][symbol] = signal_data
                
                # 거래 실행 판단
                if signal_data['signal'] != 'HOLD' and signal_data['strength'] > 0.6:
                    await self._execute_trade_signal(symbol, signal_data, current_balance, results)
            
            # 기존 포지션 관리
            for position in current_positions:
                await self._manage_existing_position(position, results)
            
        except Exception as e:
            error_msg = f"거래 사이클 실행 중 오류: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def _execute_trade_signal(self, symbol: str, signal_data: Dict, balance: float, results: Dict):
        """거래 신호 실행"""
        try:
            # 포지션 크기 계산
            position_value = balance * self.position_size_ratio
            
            # 현재 가격 조회
            current_price = await self.get_market_price(symbol)
            if current_price == 0:
                return
            
            # 수량 계산
            quantity = round(position_value / current_price, 6)
            
            # 레버리지 설정
            leverage = min(int(signal_data.get('leverage', 1)), 10)  # 최대 10배
            await self.set_leverage(symbol, leverage)
            
            # 주문 실행
            side = signal_data['signal']  # 'BUY' or 'SELL'
            
            trade = await self.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type='MARKET'
            )
            
            if trade:
                action = {
                    'symbol': symbol,
                    'action': f"{side} {quantity} at {current_price}",
                    'leverage': leverage,
                    'signal_strength': signal_data['strength'],
                    'timestamp': datetime.now().isoformat()
                }
                results['actions'].append(action)
                logger.info(f"거래 실행: {action}")
        
        except Exception as e:
            error_msg = f"거래 신호 실행 실패 ({symbol}): {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
    
    async def _manage_existing_position(self, position: Position, results: Dict):
        """기존 포지션 관리"""
        try:
            # 손절/익절 조건 확인
            pnl_ratio = position.unrealized_pnl / (position.size * position.entry_price)
            
            should_close = False
            reason = ""
            
            # 손절 조건 (-5%)
            if pnl_ratio < -0.05:
                should_close = True
                reason = "손절"
            
            # 익절 조건 (+10%)
            elif pnl_ratio > 0.10:
                should_close = True
                reason = "익절"
            
            # 포지션 유지 시간 확인 (24시간 초과시 강제 청산)
            elif (datetime.now() - position.timestamp).hours > 24:
                should_close = True
                reason = "시간 초과"
            
            if should_close:
                success = await self.close_position(position.symbol)
                
                if success:
                    action = {
                        'symbol': position.symbol,
                        'action': f"포지션 청산 ({reason})",
                        'pnl': position.unrealized_pnl,
                        'pnl_ratio': f"{pnl_ratio:.2%}",
                        'timestamp': datetime.now().isoformat()
                    }
                    results['actions'].append(action)
                    logger.info(f"포지션 청산: {action}")
        
        except Exception as e:
            error_msg = f"포지션 관리 실패 ({position.symbol}): {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
    
    async def start_live_trading(self):
        """실전매매 시작"""
        logger.info("🚀 바이낸스 테스트넷 실전매매 시작")
        
        self.is_trading = True
        
        while self.is_trading:
            try:
                # 거래 사이클 실행
                cycle_result = await self.execute_trading_cycle()
                
                # 결과 로깅
                if cycle_result['actions']:
                    logger.info(f"거래 사이클 완료: {len(cycle_result['actions'])}개 액션 실행")
                
                if cycle_result['errors']:
                    logger.warning(f"거래 사이클 오류: {cycle_result['errors']}")
                
                # 1분 대기
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"실전매매 루프 오류: {e}")
                await asyncio.sleep(30)  # 오류 발생시 30초 대기
    
    def stop_live_trading(self):
        """실전매매 중지"""
        logger.info("🛑 바이낸스 테스트넷 실전매매 중지")
        self.is_trading = False
    
    async def get_trading_status(self) -> Dict[str, Any]:
        """거래 상태 조회"""
        try:
            balance = await self.get_balance()
            positions = await self.get_positions()
            
            return {
                'is_trading': self.is_trading,
                'balance': balance,
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'leverage': pos.leverage
                    }
                    for pos in positions
                ],
                'total_orders': len(self.orders),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"거래 상태 조회 실패: {e}")
            return {
                'is_trading': self.is_trading,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }