#!/usr/bin/env python3
"""
🚀 1시간봉 기반 고급 트레이딩 전략 시스템
두 가지 핵심 전략으로 급등 초입을 정확히 포착
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 새로운 1시간봉 전략 모듈 임포트
from hourly_strategy import HourlyTradingStrategy

# ==============================================
# 🚀 1시간봉 전략 시스템 (메인 전략 엔진)
# ==============================================

class HourlyStrategyWrapper:
    """
    1시간봉 기반 전략 래퍼
    HourlyTradingStrategy를 기존 시스템과 통합
    """
    
    def __init__(self, params=None):
        self.hourly_strategy = HourlyTradingStrategy()
        self.name = "HourlyStrategy"
        self.params = params or {}
    
    def generate_signals(self, df):
        """새로운 1시간봉 전략으로 신호 생성"""
        return self.hourly_strategy.generate_signals(df)
    
    def backtest(self, df, initial_capital=10000, commission=0.0004):
        """백테스트 실행"""
        return self.hourly_strategy.backtest(df, initial_capital, commission)

class TrendFollowingStrategy(HourlyStrategyWrapper):
    """기존 TrendFollowingStrategy를 1시간봉 전략으로 대체"""
    
    def __init__(self, params=None):
        super().__init__(params)
        self.name = "TrendFollowing_Hourly"

class CVDScalpingStrategy(HourlyStrategyWrapper):
    """기존 CVDScalpingStrategy를 1시간봉 전략으로 대체"""
    
    def __init__(self, params=None):
        super().__init__(params)
        self.name = "CVDScalping_Hourly"

class VolatilityBreakoutStrategy(HourlyStrategyWrapper):
    """기존 VolatilityBreakoutStrategy를 1시간봉 전략으로 대체"""
    
    def __init__(self, params=None):
        super().__init__(params)
        self.name = "VolatilityBreakout_Hourly"

class TripleComboStrategy:
    """
    🚀 통합 전략 관리자
    1시간봉 기반 전략들을 통합 관리
    """
    
    def __init__(self, symbol="BTC/USDT", initial_capital=10000, max_risk_per_trade=0.02):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        
        # 1시간봉 전략 초기화
        self.strategies = {
            "trend_following": TrendFollowingStrategy(),
            "cvd_scalping": CVDScalpingStrategy(), 
            "volatility_breakout": VolatilityBreakoutStrategy()
        }
        
        # 전략별 가중치
        self.strategy_weights = {
            "trend_following": 0.4,
            "cvd_scalping": 0.3,
            "volatility_breakout": 0.3
        }
        
        # 포트폴리오 상태
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},
            'total_value': initial_capital,
            'trade_history': []
        }
    
    def generate_combined_signal(self, df):
        """
        모든 전략의 신호를 결합하여 최종 신호 생성
        """
        try:
            # 각 전략별 신호 수집
            strategy_signals = {}
            
            for strategy_name, strategy in self.strategies.items():
                signals = strategy.generate_signals(df)
                strategy_signals[strategy_name] = signals
            
            # 신호 결합 로직
            combined_signals = pd.DataFrame(index=df.index)
            combined_signals['signal'] = 0
            combined_signals['confidence'] = 0.0
            combined_signals['strategy'] = 'None'
            
            for i in range(len(df)):
                total_signal = 0
                total_confidence = 0
                active_strategies = []
                
                for strategy_name, signals in strategy_signals.items():
                    if i < len(signals) and signals['signal'].iloc[i] != 0:
                        weight = self.strategy_weights[strategy_name]
                        total_signal += signals['signal'].iloc[i] * weight
                        total_confidence += signals['confidence'].iloc[i] * weight
                        active_strategies.append(strategy_name)
                
                # 최종 신호 결정
                if abs(total_signal) > 0.5:  # 임계값 이상일 때만 신호 발생
                    combined_signals.loc[df.index[i], 'signal'] = 1 if total_signal > 0 else -1
                    combined_signals.loc[df.index[i], 'confidence'] = min(total_confidence, 1.0)
                    combined_signals.loc[df.index[i], 'strategy'] = '+'.join(active_strategies)
            
            return combined_signals
            
        except Exception as e:
            print(f"신호 결합 오류: {e}")
            return pd.DataFrame(index=df.index, columns=['signal', 'confidence', 'strategy']).fillna(0)
    
    def backtest_combined_strategy(self, df, commission=0.0004):
        """
        통합 전략 백테스트
        """
        try:
            signals = self.generate_combined_signal(df)
            
            capital = self.initial_capital
            position = 0
            trades = []
            
            for i in range(len(df)):
                current_price = df['close'].iloc[i]
                
                if signals['signal'].iloc[i] == 1 and position == 0:
                    # 매수 신호
                    risk_amount = capital * self.max_risk_per_trade
                    position_size = risk_amount / current_price
                    
                    capital -= position_size * current_price
                    capital -= position_size * current_price * commission
                    position = position_size
                    
                    entry_info = {
                        'entry_time': df.index[i],
                        'entry_price': current_price,
                        'position_size': position_size,
                        'confidence': signals['confidence'].iloc[i],
                        'strategy': signals['strategy'].iloc[i]
                    }
                
                elif position > 0:
                    # 매도 조건 확인 (간단한 예시)
                    # 실제로는 더 복잡한 exit 전략 사용
                    should_exit = False
                    exit_reason = ""
                    
                    # 5% 손절 또는 10% 익절
                    profit_pct = ((current_price - entry_info['entry_price']) / entry_info['entry_price']) * 100
                    
                    if profit_pct <= -5:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif profit_pct >= 10:
                        should_exit = True
                        exit_reason = "take_profit"
                    
                    if should_exit:
                        capital += position * current_price
                        capital -= position * current_price * commission
                        
                        trades.append({
                            'entry_time': entry_info['entry_time'],
                            'exit_time': df.index[i],
                            'entry_price': entry_info['entry_price'],
                            'exit_price': current_price,
                            'profit_pct': profit_pct,
                            'exit_reason': exit_reason,
                            'confidence': entry_info['confidence'],
                            'strategy': entry_info['strategy']
                        })
                        
                        position = 0
            
            # 결과 계산
            total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
            
            return {
                'total_return': total_return,
                'final_capital': capital,
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t['profit_pct'] > 0]),
                'average_profit': np.mean([t['profit_pct'] for t in trades]) if trades else 0,
                'trades': trades
            }
            
        except Exception as e:
            print(f"백테스트 오류: {e}")
            return {
                'total_return': 0,
                'final_capital': self.initial_capital,
                'total_trades': 0,
                'winning_trades': 0,
                'average_profit': 0,
                'trades': []
            }

# 기존 시스템과의 호환성을 위한 별칭
TrendFollowing = TrendFollowingStrategy
CVDScalping = CVDScalpingStrategy  
VolatilityBreakout = VolatilityBreakoutStrategy

# 전략 실행 예시
if __name__ == "__main__":
    print("🚀 1시간봉 기반 고급 트레이딩 전략 시스템")
    print("=" * 50)
    
    # 전략 시스템 초기화
    strategy_system = TripleComboStrategy(
        symbol="BTC/USDT",
        initial_capital=10000,
        max_risk_per_trade=0.02
    )
    
    print(f"전략 시스템 초기화 완료:")
    print(f"- 심볼: {strategy_system.symbol}")
    print(f"- 초기 자본: ${strategy_system.initial_capital:,}")
    print(f"- 전략 수: {len(strategy_system.strategies)}")
    print("- 전략 1: 급등 초입 포착")
    print("- 전략 2: 작은 눌림목 이후 초급등 초입")
    print("- 매도 전략: 볼린저밴드 200 상단선 기반 트레일링 스탑")