#!/usr/bin/env python3
"""
🚀 1시간봉 기반 고급 트레이딩 전략
두 가지 핵심 전략으로 급등 초입 포착
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HourlyTradingStrategy:
    """
    1시간봉 기반 트레이딩 전략 시스템
    """
    
    def __init__(self):
        self.name = "HourlyStrategy"
        self.timeframe = "1h"
        
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        middle_band = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_ma(self, df, period):
        """이동평균선 계산"""
        return df['close'].rolling(window=period).mean()
    
    def check_golden_cross(self, fast_ma, slow_ma, lookback_periods):
        """골든크로스 확인"""
        if len(fast_ma) < lookback_periods:
            return pd.Series([False] * len(fast_ma))
        
        # 현재 fast_ma가 slow_ma 위에 있고
        current_above = fast_ma > slow_ma
        
        # lookback_periods 이전에는 fast_ma가 slow_ma 아래에 있었는지 확인
        shifted_below = fast_ma.shift(lookback_periods) < slow_ma.shift(lookback_periods)
        
        # 두 조건을 모두 만족하면 골든크로스
        golden_cross = current_above & shifted_below
        
        # lookback_periods 내에 골든크로스가 발생했는지 확인
        golden_cross_within_lookback = golden_cross.rolling(window=lookback_periods).sum() > 0
        
        return golden_cross_within_lookback
    
    def strategy1_early_surge(self, df):
        """
        전략 1: 급등 초입
        1시간봉 기준으로 급등 초입 신호 포착
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['confidence'] = 0.0
        
        # 필요한 지표 계산
        bb20_upper, bb20_middle, bb20_lower = self.calculate_bollinger_bands(df, 20, 2)
        bb80_upper, bb80_middle, bb80_lower = self.calculate_bollinger_bands(df, 80, 2)
        bb200_upper, bb200_middle, bb200_lower = self.calculate_bollinger_bands(df, 200, 2)
        ma200 = self.calculate_ma(df, 200)
        ma20 = self.calculate_ma(df, 20)
        
        # 각 조건 확인
        for i in range(200, len(df)):
            # 4-1. 시가 < 볼밴20 상단 + 볼밴80 상단
            condition1 = df['open'].iloc[i] < (bb20_upper.iloc[i] + bb80_upper.iloc[i])
            
            # 4-2. 고가 > 볼밴80 상단 + 볼밴20 상단
            condition2 = df['high'].iloc[i] > (bb80_upper.iloc[i] + bb20_upper.iloc[i])
            
            # 4-3. 저가 대비 고가 폭이 4% 이상
            price_range = ((df['high'].iloc[i] - df['low'].iloc[i]) / df['low'].iloc[i]) * 100
            condition3 = price_range >= 4.0
            
            # 4-4. 시가 > 200이평선
            condition4 = df['open'].iloc[i] > ma200.iloc[i]
            
            # 4-5. 20이평선과 볼밴200 상단선 이격이 2% 이내
            if pd.notna(ma20.iloc[i]) and pd.notna(bb200_upper.iloc[i]) and bb200_upper.iloc[i] != 0:
                separation = abs((ma20.iloc[i] - bb200_upper.iloc[i]) / bb200_upper.iloc[i]) * 100
                condition5 = separation <= 2.0
            else:
                condition5 = False
            
            # 모든 조건 만족시 매수 신호
            if condition1 and condition2 and condition3 and condition4 and condition5:
                signals.loc[df.index[i], 'signal'] = 1
                # 조건 충족 개수에 따른 신뢰도
                confidence = sum([condition1, condition2, condition3, condition4, condition5]) / 5.0
                signals.loc[df.index[i], 'confidence'] = confidence
                
        return signals
    
    def strategy2_pullback_surge(self, df):
        """
        전략 2: 작은 눌림목 이후 초급등 초입
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['confidence'] = 0.0
        
        # 필요한 지표 계산
        bb20_upper, bb20_middle, bb20_lower = self.calculate_bollinger_bands(df, 20, 2)
        bb80_upper, bb80_middle, bb80_lower = self.calculate_bollinger_bands(df, 80, 2)
        bb200_upper, bb200_middle, bb200_lower = self.calculate_bollinger_bands(df, 200, 2)
        
        ma5 = self.calculate_ma(df, 5)
        ma20 = self.calculate_ma(df, 20)
        ma80 = self.calculate_ma(df, 80)
        ma200 = self.calculate_ma(df, 200)
        
        # 골든크로스 확인
        # 5-1. 100봉 이내에 80+200이평선이 볼밴20 하단선 골든크로스
        ma_sum = ma80 + ma200
        golden_cross_1 = self.check_golden_cross(ma_sum, bb20_lower, 100)
        
        # 5-2. 50봉 이내에 80이평선이 200이평선 골든크로스
        golden_cross_2 = self.check_golden_cross(ma80, ma200, 50)
        
        # 5-6. 5봉 이내에 5이평선이 20이평선 골든크로스
        golden_cross_3 = self.check_golden_cross(ma5, ma20, 5)
        
        # 각 조건 확인
        for i in range(200, len(df)):
            # 5-3. 20이평선과 볼밴200 상단선 이격이 2% 이내
            if pd.notna(ma20.iloc[i]) and pd.notna(bb200_upper.iloc[i]) and bb200_upper.iloc[i] != 0:
                separation = abs((ma20.iloc[i] - bb200_upper.iloc[i]) / bb200_upper.iloc[i]) * 100
                condition3 = separation <= 2.0
            else:
                condition3 = False
            
            # 5-4. 시가 < 볼밴20 상단선 + 볼밴80 상단선
            condition4 = df['open'].iloc[i] < (bb20_upper.iloc[i] + bb80_upper.iloc[i])
            
            # 5-5. 저가 대비 고가 4% 이상
            price_range = ((df['high'].iloc[i] - df['low'].iloc[i]) / df['low'].iloc[i]) * 100
            condition5 = price_range >= 4.0
            
            # 모든 조건 만족시 매수 신호
            if (golden_cross_1.iloc[i] and golden_cross_2.iloc[i] and 
                condition3 and condition4 and condition5 and golden_cross_3.iloc[i]):
                signals.loc[df.index[i], 'signal'] = 1
                # 조건 충족에 따른 신뢰도
                conditions_met = sum([golden_cross_1.iloc[i], golden_cross_2.iloc[i], 
                                    condition3, condition4, condition5, golden_cross_3.iloc[i]])
                signals.loc[df.index[i], 'confidence'] = conditions_met / 6.0
                
        return signals
    
    def calculate_exit_strategy(self, df, entry_price, entry_index):
        """
        매도 전략: 볼린저밴드 200 상단선 기반 트레일링 스탑
        """
        bb200_upper, _, _ = self.calculate_bollinger_bands(df, 200, 2)
        
        max_profit = 0
        exit_price = entry_price
        exit_reason = "holding"
        
        for i in range(entry_index + 1, len(df)):
            current_price = df['close'].iloc[i]
            current_profit = ((current_price - entry_price) / entry_price) * 100
            
            # 최대 수익률 업데이트
            if current_profit > max_profit:
                max_profit = current_profit
            
            # 볼린저밴드 200 상단선 위에서 거래중
            if current_price > bb200_upper.iloc[i]:
                # 수익률이 최대 수익률의 70% 이하로 떨어지면 매도
                if current_profit <= max_profit * 0.7:
                    exit_price = current_price
                    exit_reason = "trailing_stop"
                    break
            else:
                # 볼린저밴드 200 상단선 아래로 내려오면 즉시 매도
                exit_price = current_price
                exit_reason = "bb200_breakdown"
                break
                
        return {
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'max_profit': max_profit,
            'final_profit': ((exit_price - entry_price) / entry_price) * 100
        }
    
    def generate_signals(self, df):
        """
        두 전략을 결합하여 최종 신호 생성
        """
        # 전략 1: 급등 초입
        signals1 = self.strategy1_early_surge(df)
        
        # 전략 2: 눌림목 이후 급등
        signals2 = self.strategy2_pullback_surge(df)
        
        # 두 전략 중 하나라도 신호가 있으면 매수
        combined_signals = pd.DataFrame(index=df.index)
        combined_signals['signal'] = (signals1['signal'] == 1) | (signals2['signal'] == 1)
        combined_signals['signal'] = combined_signals['signal'].astype(int)
        
        # 신뢰도는 더 높은 것을 선택
        combined_signals['confidence'] = pd.concat([signals1['confidence'], signals2['confidence']], axis=1).max(axis=1)
        
        # 전략 정보 추가
        combined_signals['strategy'] = ''
        combined_signals.loc[signals1['signal'] == 1, 'strategy'] = 'Strategy1_EarlySurge'
        combined_signals.loc[signals2['signal'] == 1, 'strategy'] = 'Strategy2_PullbackSurge'
        combined_signals.loc[(signals1['signal'] == 1) & (signals2['signal'] == 1), 'strategy'] = 'Both_Strategies'
        
        return combined_signals
    
    def backtest(self, df, initial_capital=10000, commission=0.0004):
        """
        백테스트 실행
        """
        signals = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(len(df)):
            if signals['signal'].iloc[i] == 1 and position == 0:
                # 매수
                entry_price = df['close'].iloc[i]
                position_size = capital * 0.95  # 자금의 95% 사용
                position = position_size / entry_price
                capital -= position_size
                capital -= position_size * commission  # 수수료
                
                # 매도 지점 계산
                exit_info = self.calculate_exit_strategy(df, entry_price, i)
                
                trades.append({
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_info['exit_price'],
                    'exit_reason': exit_info['exit_reason'],
                    'profit_pct': exit_info['final_profit'],
                    'max_profit_pct': exit_info['max_profit'],
                    'strategy': signals['strategy'].iloc[i],
                    'confidence': signals['confidence'].iloc[i]
                })
                
                # 포지션 청산
                capital += position * exit_info['exit_price']
                capital -= position * exit_info['exit_price'] * commission  # 수수료
                position = 0
        
        # 최종 수익률 계산
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        return {
            'total_return': total_return,
            'trades': trades,
            'final_capital': capital,
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t['profit_pct'] > 0]),
            'average_profit': np.mean([t['profit_pct'] for t in trades]) if trades else 0
        }