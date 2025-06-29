"""
동적 레버리지 관리자
시장 국면별 및 Phase별 레버리지 동적 조정
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum

# 설정 파일 임포트
import sys
sys.path.append('../../../')
from config.backtest_config import backtest_config

class MarketCondition(Enum):
    """시장 국면"""
    BULL_MARKET = "BULL_MARKET"      # 상승장
    BEAR_MARKET = "BEAR_MARKET"      # 하락장
    SIDEWAYS = "SIDEWAYS"            # 횡보장
    HIGH_VOLATILITY = "HIGH_VOLATILITY"  # 고변동성
    LOW_VOLATILITY = "LOW_VOLATILITY"    # 저변동성

class PhaseType(Enum):
    """Phase 타입"""
    PHASE1_AGGRESSIVE = "PHASE1_AGGRESSIVE"  # 공격 모드
    PHASE2_DEFENSIVE = "PHASE2_DEFENSIVE"    # 방어 모드

class DynamicLeverageManager:
    """동적 레버리지 관리자"""
    
    def __init__(self):
        self.current_leverage = 1.0
        self.leverage_history = []
        self.adjustment_reasons = []
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 레버리지 설정
        self.leverage_config = {
            'phase1': {
                'base_leverage': 3.0,
                'max_leverage': 7.0,
                'min_leverage': 1.5,
                'market_adjustments': {
                    MarketCondition.BULL_MARKET: 1.3,      # 30% 증가
                    MarketCondition.BEAR_MARKET: 0.6,      # 40% 감소
                    MarketCondition.HIGH_VOLATILITY: 0.5,  # 50% 감소
                    MarketCondition.LOW_VOLATILITY: 1.2,   # 20% 증가
                    MarketCondition.SIDEWAYS: 1.0          # 변화 없음
                }
            },
            'phase2': {
                'base_leverage': 1.5,
                'max_leverage': 5.0,
                'min_leverage': 1.0,
                'market_adjustments': {
                    MarketCondition.BULL_MARKET: 1.4,      # 40% 증가
                    MarketCondition.BEAR_MARKET: 0.7,      # 30% 감소
                    MarketCondition.HIGH_VOLATILITY: 0.6,  # 40% 감소
                    MarketCondition.LOW_VOLATILITY: 1.3,   # 30% 증가
                    MarketCondition.SIDEWAYS: 1.0          # 변화 없음
                }
            }
        }
        
    def calculate_dynamic_leverage(self,
                                  phase: PhaseType,
                                  market_condition: MarketCondition,
                                  current_capital: float,
                                  peak_capital: float,
                                  consecutive_wins: int,
                                  consecutive_losses: int,
                                  volatility: float,
                                  rsi: float) -> float:
        """동적 레버리지 계산"""
        
        # 기본 설정 가져오기
        phase_config = self.leverage_config['phase1'] if phase == PhaseType.PHASE1_AGGRESSIVE else self.leverage_config['phase2']
        
        # 기본 레버리지
        leverage = phase_config['base_leverage']
        
        # 1. 시장 국면에 따른 조정
        market_adjustment = phase_config['market_adjustments'].get(market_condition, 1.0)
        leverage *= market_adjustment
        
        # 2. 낙폭에 따른 조정
        if peak_capital > 0:
            current_drawdown = (peak_capital - current_capital) / peak_capital
            
            if current_drawdown > 0.15:  # 15% 이상 낙폭
                leverage *= 0.6  # 40% 감소
            elif current_drawdown > 0.10:  # 10% 이상 낙폭
                leverage *= 0.7  # 30% 감소
            elif current_drawdown > 0.05:  # 5% 이상 낙폭
                leverage *= 0.85  # 15% 감소
            elif current_drawdown < -0.05:  # 5% 이상 수익
                leverage *= 1.1  # 10% 증가
                
        # 3. 연속 거래 결과에 따른 조정
        if consecutive_losses >= 4:
            leverage *= 0.5  # 50% 감소
        elif consecutive_losses >= 3:
            leverage *= 0.7  # 30% 감소
        elif consecutive_losses >= 2:
            leverage *= 0.85  # 15% 감소
        elif consecutive_wins >= 5:
            leverage *= 1.2  # 20% 증가
        elif consecutive_wins >= 3:
            leverage *= 1.1  # 10% 증가
            
        # 4. 변동성에 따른 조정
        if volatility > 0.10:  # 10% 이상 변동성
            leverage *= 0.7  # 30% 감소
        elif volatility > 0.08:  # 8% 이상 변동성
            leverage *= 0.8  # 20% 감소
        elif volatility < 0.03:  # 3% 이하 변동성
            leverage *= 1.15  # 15% 증가
            
        # 5. RSI에 따른 조정
        if rsi > 80:  # 과매수
            leverage *= 0.8  # 20% 감소
        elif rsi > 70:  # 상대적 과매수
            leverage *= 0.9  # 10% 감소
        elif rsi < 20:  # 과매도
            leverage *= 1.2  # 20% 증가
        elif rsi < 30:  # 상대적 과매도
            leverage *= 1.1  # 10% 증가
            
        # 6. Phase별 특별 조정
        if phase == PhaseType.PHASE1_AGGRESSIVE:
            # 공격 모드에서의 추가 조정
            if market_condition == MarketCondition.BULL_MARKET and volatility < 0.05:
                leverage *= 1.1  # 상승장 + 저변동성 시 10% 추가 증가
        else:
            # 방어 모드에서의 추가 조정
            if market_condition == MarketCondition.BEAR_MARKET:
                leverage *= 0.9  # 하락장에서 추가 10% 감소
                
        # 7. 최소/최대 레버리지 제한
        leverage = max(phase_config['min_leverage'], min(leverage, phase_config['max_leverage']))
        
        # 8. 소수점 둘째 자리로 반올림
        leverage = round(leverage, 2)
        
        return leverage
        
    def get_leverage_adjustment_reason(self,
                                      phase: PhaseType,
                                      market_condition: MarketCondition,
                                      current_drawdown: float,
                                      consecutive_wins: int,
                                      consecutive_losses: int,
                                      volatility: float,
                                      rsi: float) -> str:
        """레버리지 조정 이유 반환"""
        
        reasons = []
        
        # 시장 국면
        if market_condition == MarketCondition.BULL_MARKET:
            reasons.append("상승장")
        elif market_condition == MarketCondition.BEAR_MARKET:
            reasons.append("하락장")
        elif market_condition == MarketCondition.HIGH_VOLATILITY:
            reasons.append("고변동성")
        elif market_condition == MarketCondition.LOW_VOLATILITY:
            reasons.append("저변동성")
            
        # 낙폭
        if current_drawdown > 0.15:
            reasons.append("높은 낙폭(15%+)")
        elif current_drawdown > 0.10:
            reasons.append("중간 낙폭(10%+)")
        elif current_drawdown > 0.05:
            reasons.append("낮은 낙폭(5%+)")
            
        # 연속 거래 결과
        if consecutive_losses >= 4:
            reasons.append("연속 손실 4회+")
        elif consecutive_losses >= 3:
            reasons.append("연속 손실 3회")
        elif consecutive_wins >= 5:
            reasons.append("연속 승리 5회+")
        elif consecutive_wins >= 3:
            reasons.append("연속 승리 3회")
            
        # 변동성
        if volatility > 0.10:
            reasons.append("높은 변동성(10%+)")
        elif volatility < 0.03:
            reasons.append("낮은 변동성(3%-)")
            
        # RSI
        if rsi > 80:
            reasons.append("과매수(RSI 80+)")
        elif rsi < 20:
            reasons.append("과매도(RSI 20-)")
            
        return ", ".join(reasons) if reasons else "기본 설정"
        
    def update_leverage(self,
                       phase: PhaseType,
                       market_condition: MarketCondition,
                       current_capital: float,
                       peak_capital: float,
                       consecutive_wins: int,
                       consecutive_losses: int,
                       volatility: float,
                       rsi: float) -> float:
        """레버리지 업데이트"""
        
        old_leverage = self.current_leverage
        
        # 새로운 레버리지 계산
        new_leverage = self.calculate_dynamic_leverage(
            phase=phase,
            market_condition=market_condition,
            current_capital=current_capital,
            peak_capital=peak_capital,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            volatility=volatility,
            rsi=rsi
        )
        
        # 레버리지 변경 시 기록
        if abs(new_leverage - old_leverage) > 0.1:  # 0.1 이상 변경 시
            reason = self.get_leverage_adjustment_reason(
                phase, market_condition, 
                (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0,
                consecutive_wins, consecutive_losses, volatility, rsi
            )
            
            adjustment_record = {
                'timestamp': datetime.now(),
                'old_leverage': old_leverage,
                'new_leverage': new_leverage,
                'phase': phase.value,
                'market_condition': market_condition.value,
                'reason': reason,
                'current_capital': current_capital,
                'volatility': volatility,
                'rsi': rsi
            }
            
            self.leverage_history.append(adjustment_record)
            self.adjustment_reasons.append(reason)
            
            self.logger.info(f"레버리지 조정: {old_leverage} → {new_leverage} ({reason})")
            
        self.current_leverage = new_leverage
        return new_leverage
        
    def get_leverage_summary(self) -> Dict:
        """레버리지 요약 정보 반환"""
        
        if not self.leverage_history:
            return {
                'current_leverage': self.current_leverage,
                'total_adjustments': 0,
                'average_leverage': self.current_leverage,
                'min_leverage': self.current_leverage,
                'max_leverage': self.current_leverage
            }
            
        leverages = [record['new_leverage'] for record in self.leverage_history]
        
        return {
            'current_leverage': self.current_leverage,
            'total_adjustments': len(self.leverage_history),
            'average_leverage': np.mean(leverages),
            'min_leverage': min(leverages),
            'max_leverage': max(leverages),
            'recent_adjustments': self.leverage_history[-5:] if len(self.leverage_history) >= 5 else self.leverage_history
        }
        
    def get_phase_leverage_guidelines(self) -> Dict:
        """Phase별 레버리지 가이드라인 반환"""
        
        return {
            'phase1_aggressive': {
                'base_leverage': 3.0,
                'max_leverage': 7.0,
                'min_leverage': 1.5,
                'market_conditions': {
                    'bull_market': '3.9x (기본 3.0 x 1.3)',
                    'bear_market': '1.8x (기본 3.0 x 0.6)',
                    'high_volatility': '1.5x (기본 3.0 x 0.5)',
                    'low_volatility': '3.6x (기본 3.0 x 1.2)',
                    'sideways': '3.0x (기본값)'
                }
            },
            'phase2_defensive': {
                'base_leverage': 1.5,
                'max_leverage': 5.0,
                'min_leverage': 1.0,
                'market_conditions': {
                    'bull_market': '2.1x (기본 1.5 x 1.4)',
                    'bear_market': '1.05x (기본 1.5 x 0.7)',
                    'high_volatility': '0.9x (기본 1.5 x 0.6)',
                    'low_volatility': '1.95x (기본 1.5 x 1.3)',
                    'sideways': '1.5x (기본값)'
                }
            }
        }
        
    def get_leverage_history(self) -> List[Dict]:
        """레버리지 변경 기록 반환"""
        return self.leverage_history
        
    def reset_leverage(self):
        """레버리지 초기화"""
        self.current_leverage = 1.0
        self.leverage_history = []
        self.adjustment_reasons = []
        self.logger.info("레버리지 초기화 완료") 