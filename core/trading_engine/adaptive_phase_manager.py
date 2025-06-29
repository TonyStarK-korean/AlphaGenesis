import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum
import threading
import time

# 설정 파일 임포트
import sys
sys.path.append('../../../')
from config.backtest_config import backtest_config

class PhaseType(Enum):
    """Phase 타입"""
    PHASE1_AGGRESSIVE = "PHASE1_AGGRESSIVE"  # 공격 모드
    PHASE2_DEFENSIVE = "PHASE2_DEFENSIVE"    # 방어 모드

class MarketCondition(Enum):
    """시장 국면"""
    BULL_MARKET = "BULL_MARKET"      # 상승장
    BEAR_MARKET = "BEAR_MARKET"      # 하락장
    SIDEWAYS = "SIDEWAYS"            # 횡보장
    HIGH_VOLATILITY = "HIGH_VOLATILITY"  # 고변동성
    LOW_VOLATILITY = "LOW_VOLATILITY"    # 저변동성

class AdaptivePhaseManager:
    """적응형 Phase 관리자"""
    
    def __init__(self):
        self.current_phase = PhaseType.PHASE1_AGGRESSIVE
        self.current_market_condition = MarketCondition.SIDEWAYS
        self.phase_history = []
        self.market_condition_history = []
        
        # 성과 추적
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        self.peak_capital = backtest_config.initial_capital
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def analyze_market_condition(self, market_data: pd.DataFrame) -> MarketCondition:
        """시장 국면 분석"""
        
        if market_data.empty:
            return MarketCondition.SIDEWAYS
            
        # 기술적 지표 계산
        close_prices = market_data['close']
        
        # 이동평균
        ma_short = close_prices.rolling(window=10).mean()
        ma_long = close_prices.rolling(window=30).mean()
        
        # RSI 계산
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 변동성 계산
        returns = close_prices.pct_change()
        volatility = returns.rolling(window=20).std()
        
        # 거래량 분석
        volume = market_data['volume']
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_ma
        
        # 최신 데이터
        current_price = close_prices.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_volatility = volatility.iloc[-1]
        current_volume_ratio = volume_ratio.iloc[-1]
        
        # 시장 국면 판단
        if pd.isna(current_ma_short) or pd.isna(current_ma_long):
            return MarketCondition.SIDEWAYS
            
        # 상승장 조건
        if (current_price > current_ma_short > current_ma_long and 
            current_rsi < 70 and current_volume_ratio > 1.2):
            return MarketCondition.BULL_MARKET
            
        # 하락장 조건
        elif (current_price < current_ma_short < current_ma_long and 
              current_rsi > 30 and current_volume_ratio > 1.2):
            return MarketCondition.BEAR_MARKET
            
        # 고변동성 조건
        elif current_volatility > backtest_config.market_analysis['volatility_threshold']:
            return MarketCondition.HIGH_VOLATILITY
            
        # 저변동성 조건
        elif current_volatility < backtest_config.market_analysis['volatility_threshold'] * 0.5:
            return MarketCondition.LOW_VOLATILITY
            
        else:
            return MarketCondition.SIDEWAYS
            
    def should_transition_phase(self, 
                               current_capital: float,
                               market_condition: MarketCondition,
                               recent_trades: List[Dict]) -> Tuple[bool, PhaseType]:
        """Phase 전환 여부 결정"""
        
        # 현재 Phase 설정 가져오기
        if self.current_phase == PhaseType.PHASE1_AGGRESSIVE:
            current_settings = backtest_config.phase1_aggressive
            transition_conditions = backtest_config.phase_transition['aggressive_to_defensive']
        else:
            current_settings = backtest_config.phase2_defensive
            transition_conditions = backtest_config.phase_transition['defensive_to_aggressive']
            
        # 낙폭 계산
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        else:
            self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
            
        # 최근 거래 결과 분석
        if recent_trades:
            last_trades = recent_trades[-10:]  # 최근 10개 거래
            wins = sum(1 for trade in last_trades if trade.get('pnl', 0) > 0)
            losses = len(last_trades) - wins
            
            # 연속 승/패 계산
            consecutive_wins = 0
            consecutive_losses = 0
            
            for trade in reversed(last_trades):
                if trade.get('pnl', 0) > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    
            self.consecutive_wins = consecutive_wins
            self.consecutive_losses = consecutive_losses
            
        # Phase 전환 조건 체크
        if self.current_phase == PhaseType.PHASE1_AGGRESSIVE:
            # 공격 → 방어 전환 조건
            should_transition = (
                self.consecutive_losses >= transition_conditions['consecutive_losses'] or
                self.current_drawdown >= transition_conditions['drawdown_threshold'] or
                market_condition == MarketCondition.BEAR_MARKET or
                market_condition == MarketCondition.HIGH_VOLATILITY
            )
            
            if should_transition:
                return True, PhaseType.PHASE2_DEFENSIVE
                
        else:
            # 방어 → 공격 전환 조건
            should_transition = (
                self.consecutive_wins >= transition_conditions['consecutive_wins'] and
                self.current_drawdown < transition_conditions['profit_threshold'] and
                market_condition == MarketCondition.BULL_MARKET and
                market_condition != MarketCondition.HIGH_VOLATILITY
            )
            
            if should_transition:
                return True, PhaseType.PHASE1_AGGRESSIVE
                
        return False, self.current_phase
        
    def get_current_settings(self) -> Dict:
        """현재 Phase 설정 반환"""
        
        if self.current_phase == PhaseType.PHASE1_AGGRESSIVE:
            settings = backtest_config.phase1_aggressive.copy()
            settings['phase_name'] = 'PHASE1_AGGRESSIVE'
            settings['description'] = '공격 모드 - 높은 레버리지, 적극적 거래'
        else:
            settings = backtest_config.phase2_defensive.copy()
            settings['phase_name'] = 'PHASE2_DEFENSIVE'
            settings['description'] = '방어 모드 - 낮은 레버리지, 보수적 거래'
            
        return settings
        
    def transition_phase(self, new_phase: PhaseType, reason: str = ""):
        """Phase 전환"""
        
        old_phase = self.current_phase
        self.current_phase = new_phase
        
        # 전환 기록
        transition_record = {
            'timestamp': datetime.now(),
            'old_phase': old_phase.value,
            'new_phase': new_phase.value,
            'reason': reason,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': self.current_drawdown,
            'market_condition': self.current_market_condition.value
        }
        
        self.phase_history.append(transition_record)
        
        self.logger.info(f"Phase 전환: {old_phase.value} → {new_phase.value} (이유: {reason})")
        
    def update_market_condition(self, market_condition: MarketCondition):
        """시장 국면 업데이트"""
        
        old_condition = self.current_market_condition
        self.current_market_condition = market_condition
        
        if old_condition != market_condition:
            condition_record = {
                'timestamp': datetime.now(),
                'old_condition': old_condition.value,
                'new_condition': market_condition.value
            }
            self.market_condition_history.append(condition_record)
            
            self.logger.info(f"시장 국면 변화: {old_condition.value} → {market_condition.value}")
            
    def get_dynamic_leverage(self, base_leverage: float, market_condition: MarketCondition) -> float:
        """동적 레버리지 계산"""
        
        # 기본 레버리지
        leverage = base_leverage
        
        # 시장 국면에 따른 조정
        if market_condition == MarketCondition.BULL_MARKET:
            leverage *= 1.2  # 상승장에서 20% 증가
        elif market_condition == MarketCondition.BEAR_MARKET:
            leverage *= 0.7  # 하락장에서 30% 감소
        elif market_condition == MarketCondition.HIGH_VOLATILITY:
            leverage *= 0.6  # 고변동성에서 40% 감소
        elif market_condition == MarketCondition.LOW_VOLATILITY:
            leverage *= 1.1  # 저변동성에서 10% 증가
            
        # 낙폭에 따른 추가 조정
        if self.current_drawdown > 0.1:  # 10% 이상 낙폭
            leverage *= 0.8  # 20% 추가 감소
        elif self.current_drawdown > 0.05:  # 5% 이상 낙폭
            leverage *= 0.9  # 10% 추가 감소
            
        # 연속 손실에 따른 조정
        if self.consecutive_losses >= 3:
            leverage *= 0.7  # 30% 감소
        elif self.consecutive_losses >= 2:
            leverage *= 0.85  # 15% 감소
            
        # 최소/최대 레버리지 제한
        leverage = max(1.0, min(leverage, 5.0))
        
        return leverage
        
    def get_dynamic_position_size(self, base_position_size: float, market_condition: MarketCondition) -> float:
        """동적 포지션 크기 계산"""
        
        # 기본 포지션 크기
        position_size = base_position_size
        
        # 시장 국면에 따른 조정
        if market_condition == MarketCondition.BULL_MARKET:
            position_size *= 1.3  # 상승장에서 30% 증가
        elif market_condition == MarketCondition.BEAR_MARKET:
            position_size *= 0.6  # 하락장에서 40% 감소
        elif market_condition == MarketCondition.HIGH_VOLATILITY:
            position_size *= 0.5  # 고변동성에서 50% 감소
        elif market_condition == MarketCondition.LOW_VOLATILITY:
            position_size *= 1.2  # 저변동성에서 20% 증가
            
        # 낙폭에 따른 조정
        if self.current_drawdown > 0.1:
            position_size *= 0.6  # 40% 감소
        elif self.current_drawdown > 0.05:
            position_size *= 0.8  # 20% 감소
            
        # 최대 포지션 크기 제한
        position_size = min(position_size, backtest_config.max_position_size)
        
        return position_size
        
    def get_phase_status(self) -> Dict:
        """Phase 상태 정보 반환"""
        
        return {
            'current_phase': self.current_phase.value,
            'current_market_condition': self.current_market_condition.value,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': self.current_drawdown,
            'peak_capital': self.peak_capital,
            'phase_transitions': len(self.phase_history),
            'market_condition_changes': len(self.market_condition_history),
            'current_settings': self.get_current_settings()
        }
        
    def get_phase_history(self) -> List[Dict]:
        """Phase 전환 기록 반환"""
        return self.phase_history
        
    def get_market_condition_history(self) -> List[Dict]:
        """시장 국면 변화 기록 반환"""
        return self.market_condition_history 