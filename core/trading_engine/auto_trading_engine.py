import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum

class OrderType(Enum):
    """주문 타입"""
    BUY = "BUY"
    SELL = "SELL"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    CLOSE_ALL = "CLOSE_ALL"

class PositionStatus(Enum):
    """포지션 상태"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"

class AutoTradingEngine:
    """
    상위 0.01%급 자동매매 엔진
    - 실시간 매수/매도 신호 처리
    - 동적 익절/손절 관리
    - 포지션 사이징 최적화
    - 리스크 관리
    """
    
    def __init__(self, 
                 initial_capital: float = 100_000_000,
                 max_position_size: float = 0.1,
                 default_stop_loss: float = 0.02,
                 default_take_profit: float = 0.05,
                 trailing_stop: bool = True,
                 trailing_stop_distance: float = 0.01):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.default_stop_loss = default_stop_loss
        self.default_take_profit = default_take_profit
        self.trailing_stop = trailing_stop
        self.trailing_stop_distance = trailing_stop_distance
        
        # 포지션 관리
        self.positions = {}  # {symbol: position_info}
        self.order_history = []
        self.trade_history = []
        
        # 성과 추적
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def process_signal(self, 
                      symbol: str, 
                      signal: Dict, 
                      current_price: float,
                      market_data: pd.DataFrame) -> Optional[Dict]:
        """매매 신호 처리"""
        
        signal_type = signal.get('signal', 0)
        confidence = signal.get('confidence', 0.0)
        strength = signal.get('strength', 0.0)
        
        # 신뢰도가 낮으면 거래하지 않음
        if confidence < 0.6:
            self.logger.info(f"신뢰도 부족: {symbol} - {confidence:.2f}")
            return None
            
        # 현재 포지션 확인
        current_position = self.positions.get(symbol, None)
        
        if signal_type == 1:  # 매수 신호
            return self._process_buy_signal(symbol, signal, current_price, market_data)
        elif signal_type == -1:  # 매도 신호
            return self._process_sell_signal(symbol, signal, current_price, market_data)
        elif signal_type == 0:  # 청산 신호
            return self._process_close_signal(symbol, current_price)
            
        return None
        
    def _process_buy_signal(self, 
                           symbol: str, 
                           signal: Dict, 
                           current_price: float,
                           market_data: pd.DataFrame) -> Optional[Dict]:
        """매수 신호 처리"""
        
        # 이미 포지션이 있으면 추가 매수 고려
        current_position = self.positions.get(symbol, None)
        
        if current_position and current_position['status'] == PositionStatus.OPEN:
            # 추가 매수 조건: 강한 신호이고 손실 중일 때
            if signal.get('strength', 0) > 0.8 and current_position['unrealized_pnl'] < 0:
                return self._execute_buy_order(symbol, signal, current_price, "ADDITIONAL_BUY")
            else:
                self.logger.info(f"이미 포지션 보유 중: {symbol}")
                return None
        
        # 새 포지션 진입
        return self._execute_buy_order(symbol, signal, current_price, "NEW_POSITION")
        
    def _process_sell_signal(self, 
                            symbol: str, 
                            signal: Dict, 
                            current_price: float,
                            market_data: pd.DataFrame) -> Optional[Dict]:
        """매도 신호 처리"""
        
        current_position = self.positions.get(symbol, None)
        
        if not current_position or current_position['status'] != PositionStatus.OPEN:
            self.logger.info(f"매도할 포지션 없음: {symbol}")
            return None
            
        # 매도 실행
        return self._execute_sell_order(symbol, signal, current_price, "SIGNAL_SELL")
        
    def _process_close_signal(self, symbol: str, current_price: float) -> Optional[Dict]:
        """청산 신호 처리"""
        
        current_position = self.positions.get(symbol, None)
        
        if not current_position or current_position['status'] != PositionStatus.OPEN:
            return None
            
        return self._execute_sell_order(symbol, {}, current_price, "CLOSE_ALL")
        
    def _execute_buy_order(self, 
                          symbol: str, 
                          signal: Dict, 
                          current_price: float,
                          order_reason: str) -> Dict:
        """매수 주문 실행"""
        
        # 포지션 크기 계산
        confidence = signal.get('confidence', 0.5)
        strength = signal.get('strength', 0.5)
        
        # 동적 포지션 사이징
        position_size_ratio = min(
            self.max_position_size * confidence * strength,
            self.max_position_size
        )
        
        available_capital = self.current_capital * 0.95  # 5% 여유자금
        position_value = available_capital * position_size_ratio
        quantity = position_value / current_price
        
        if quantity <= 0:
            self.logger.warning(f"매수 수량이 0: {symbol}")
            return None
            
        # 주문 실행
        order = {
            'symbol': symbol,
            'order_type': OrderType.BUY,
            'quantity': quantity,
            'price': current_price,
            'timestamp': datetime.now(),
            'reason': order_reason,
            'confidence': confidence,
            'strength': strength
        }
        
        # 포지션 정보 업데이트
        position_info = {
            'symbol': symbol,
            'entry_price': current_price,
            'quantity': quantity,
            'entry_time': datetime.now(),
            'status': PositionStatus.OPEN,
            'unrealized_pnl': 0.0,
            'stop_loss': current_price * (1 - self.default_stop_loss),
            'take_profit': current_price * (1 + self.default_take_profit),
            'trailing_stop': current_price * (1 - self.trailing_stop_distance),
            'order_history': [order]
        }
        
        self.positions[symbol] = position_info
        self.order_history.append(order)
        
        # 자본 차감
        self.current_capital -= position_value
        
        self.logger.info(f"매수 주문 실행: {symbol} - {quantity:.4f} @ {current_price:.2f}")
        
        return order
        
    def _execute_sell_order(self, 
                           symbol: str, 
                           signal: Dict, 
                           current_price: float,
                           order_reason: str) -> Dict:
        """매도 주문 실행"""
        
        current_position = self.positions.get(symbol, None)
        
        if not current_position:
            return None
            
        # 실현 손익 계산
        unrealized_pnl = (current_price - current_position['entry_price']) * current_position['quantity']
        realized_pnl = unrealized_pnl
        
        # 주문 실행
        order = {
            'symbol': symbol,
            'order_type': OrderType.SELL,
            'quantity': current_position['quantity'],
            'price': current_price,
            'timestamp': datetime.now(),
            'reason': order_reason,
            'realized_pnl': realized_pnl
        }
        
        # 포지션 종료
        current_position['status'] = PositionStatus.CLOSED
        current_position['exit_price'] = current_price
        current_position['exit_time'] = datetime.now()
        current_position['realized_pnl'] = realized_pnl
        current_position['order_history'].append(order)
        
        # 자본 복원
        position_value = current_position['quantity'] * current_price
        self.current_capital += position_value
        
        # 성과 업데이트
        self.total_trades += 1
        self.total_pnl += realized_pnl
        
        if realized_pnl > 0:
            self.winning_trades += 1
            
        # 거래 기록
        trade_record = {
            'symbol': symbol,
            'entry_price': current_position['entry_price'],
            'exit_price': current_price,
            'quantity': current_position['quantity'],
            'entry_time': current_position['entry_time'],
            'exit_time': datetime.now(),
            'pnl': realized_pnl,
            'pnl_percent': (realized_pnl / (current_position['entry_price'] * current_position['quantity'])) * 100,
            'reason': order_reason
        }
        
        self.trade_history.append(trade_record)
        self.order_history.append(order)
        
        self.logger.info(f"매도 주문 실행: {symbol} - {realized_pnl:.2f} ({trade_record['pnl_percent']:.2f}%)")
        
        return order
        
    def update_positions(self, market_data: pd.DataFrame):
        """포지션 업데이트 (익절/손절 체크)"""
        
        for symbol, position in self.positions.items():
            if position['status'] != PositionStatus.OPEN:
                continue
                
            current_price = market_data.loc[market_data.index[-1], 'Close']
            
            # 손절 체크
            if current_price <= position['stop_loss']:
                self._execute_sell_order(symbol, {}, current_price, "STOP_LOSS")
                continue
                
            # 익절 체크
            if current_price >= position['take_profit']:
                self._execute_sell_order(symbol, {}, current_price, "TAKE_PROFIT")
                continue
                
            # 트레일링 스탑 업데이트
            if self.trailing_stop and current_price > position['entry_price']:
                new_trailing_stop = current_price * (1 - self.trailing_stop_distance)
                if new_trailing_stop > position['trailing_stop']:
                    position['trailing_stop'] = new_trailing_stop
                    
            # 트레일링 스탑 체크
            if self.trailing_stop and current_price <= position['trailing_stop']:
                self._execute_sell_order(symbol, {}, current_price, "TRAILING_STOP")
                continue
                
            # 미실현 손익 업데이트
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
            
    def get_performance_metrics(self) -> Dict:
        """성과 지표 반환"""
        
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0,
                'avg_pnl_per_trade': 0.0
            }
            
        win_rate = (self.winning_trades / self.total_trades) * 100
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        avg_pnl_per_trade = self.total_pnl / self.total_trades
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'total_return': total_return,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'current_capital': self.current_capital,
            'open_positions': len([p for p in self.positions.values() if p['status'] == PositionStatus.OPEN])
        }
        
    def get_open_positions(self) -> List[Dict]:
        """오픈 포지션 반환"""
        return [
            {
                'symbol': symbol,
                'entry_price': pos['entry_price'],
                'quantity': pos['quantity'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'entry_time': pos['entry_time']
            }
            for symbol, pos in self.positions.items()
            if pos['status'] == PositionStatus.OPEN
        ]
        
    def emergency_close_all(self, current_prices: Dict[str, float]):
        """긴급 전체 청산"""
        self.logger.warning("긴급 전체 청산 실행!")
        
        for symbol, position in self.positions.items():
            if position['status'] == PositionStatus.OPEN:
                current_price = current_prices.get(symbol, position['entry_price'])
                self._execute_sell_order(symbol, {}, current_price, "EMERGENCY_CLOSE")
                
        self.logger.info("긴급 전체 청산 완료") 