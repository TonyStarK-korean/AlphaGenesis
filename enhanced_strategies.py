#!/usr/bin/env python3
"""
🚀 강화된 1시간봉 트레이딩 전략 시스템
기존 전략 + 알파 지표들을 조합한 4가지 전략
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
warnings.filterwarnings('ignore')

from hourly_strategy import HourlyTradingStrategy

class AlphaIndicators:
    """
    알파 지표들을 계산하는 클래스
    """
    
    @staticmethod
    def volume_explosion_detector(df, lookback=20, threshold=2.0):
        """거래량 폭발 감지"""
        volume_ma = df['volume'].rolling(window=lookback).mean()
        current_volume = df['volume']
        explosion = current_volume > (volume_ma * threshold)
        return explosion
    
    @staticmethod
    def market_structure_shift(df, lookback=10):
        """시장 구조 변화 감지 (Higher Highs, Higher Lows)"""
        highs = df['high'].rolling(3, center=True).max()
        lows = df['low'].rolling(3, center=True).min()
        
        higher_highs = highs > highs.shift(lookback)
        higher_lows = lows > lows.shift(lookback)
        
        return higher_highs & higher_lows
    
    @staticmethod
    def fibonacci_pullback_strength(df, swing_period=20):
        """피보나치 되돌림 기반 눌림목 강도"""
        # 최근 스윙 고점/저점 찾기
        swing_high = df['high'].rolling(swing_period, center=True).max()
        swing_low = df['low'].rolling(swing_period, center=True).min()
        
        # 현재 가격의 피보나치 레벨 계산
        price_range = swing_high - swing_low
        current_level = (df['close'] - swing_low) / price_range
        
        # 38.2%, 50%, 61.8% 구간에서의 반등 신호
        fib_zones = (
            ((current_level >= 0.35) & (current_level <= 0.42)) |  # 38.2%
            ((current_level >= 0.47) & (current_level <= 0.53)) |  # 50%
            ((current_level >= 0.58) & (current_level <= 0.65))    # 61.8%
        )
        
        return fib_zones
    
    @staticmethod
    def bullish_divergence_detector(df, rsi_period=14, lookback=10):
        """RSI 강세 다이버전스 감지"""
        # RSI 계산 (numpy/pandas 버전)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 최근 가격 저점들
        price_lows = df['low'].rolling(3, center=True).min()
        price_lower_low = price_lows < price_lows.shift(lookback)
        
        # 해당 시점의 RSI 값들
        rsi_at_lows = rsi.where(price_lows == df['low'])
        rsi_higher_low = rsi_at_lows > rsi_at_lows.shift(lookback)
        
        return price_lower_low & rsi_higher_low
    
    @staticmethod
    def liquidity_analysis(df):
        """유동성 분석 (거래량과 가격 변동성 기반)"""
        # 가격 변동성 대비 거래량 비율
        price_change = abs(df['close'].pct_change())
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        
        # 유동성이 충분한 구간 (높은 거래량, 낮은 변동성)
        high_liquidity = (volume_ratio > 1.2) & (price_change < 0.05)
        
        return high_liquidity
    
    @staticmethod
    def volatility_regime_filter(df, period=20):
        """변동성 체제 필터"""
        returns = df['close'].pct_change()
        volatility = returns.rolling(period).std() * np.sqrt(24)  # 1시간봉 기준 일 변동성
        
        # 중간 변동성 구간 (0.5% ~ 4%)
        optimal_volatility = (volatility >= 0.005) & (volatility <= 0.04)
        
        return optimal_volatility
    
    @staticmethod
    def smart_money_flow(df):
        """스마트 머니 플로우 지표"""
        # On-Balance Volume 계산 (numpy/pandas 버전)
        price_change = df['close'].diff()
        obv = (df['volume'] * np.sign(price_change)).fillna(0).cumsum()
        obv_slope = obv.pct_change(5)  # 5봉 변화율
        
        # 가격과 OBV 동조성
        price_change = df['close'].pct_change(5)
        money_flow_alignment = (
            ((price_change > 0) & (obv_slope > 0)) |  # 상승 시 OBV도 상승
            ((price_change < 0) & (obv_slope < 0))    # 하락 시 OBV도 하락
        )
        
        return money_flow_alignment & (obv_slope > 0.02)  # 강한 유입
    
    @staticmethod
    def pattern_strength_score(df):
        """패턴 강도 종합 점수"""
        # 여러 지표를 조합한 종합 점수
        score = pd.Series(0, index=df.index)
        
        # 각 지표별 가중치
        weights = {
            'volume_explosion': 0.25,
            'market_structure': 0.20,
            'fib_pullback': 0.15,
            'divergence': 0.15,
            'liquidity': 0.10,
            'volatility': 0.10,
            'money_flow': 0.05
        }
        
        # 각 지표 계산 및 점수 합산
        if AlphaIndicators.volume_explosion_detector(df).any():
            score += AlphaIndicators.volume_explosion_detector(df) * weights['volume_explosion']
            
        if AlphaIndicators.market_structure_shift(df).any():
            score += AlphaIndicators.market_structure_shift(df) * weights['market_structure']
            
        if AlphaIndicators.fibonacci_pullback_strength(df).any():
            score += AlphaIndicators.fibonacci_pullback_strength(df) * weights['fib_pullback']
            
        if AlphaIndicators.bullish_divergence_detector(df).any():
            score += AlphaIndicators.bullish_divergence_detector(df) * weights['divergence']
            
        if AlphaIndicators.liquidity_analysis(df).any():
            score += AlphaIndicators.liquidity_analysis(df) * weights['liquidity']
            
        if AlphaIndicators.volatility_regime_filter(df).any():
            score += AlphaIndicators.volatility_regime_filter(df) * weights['volatility']
            
        if AlphaIndicators.smart_money_flow(df).any():
            score += AlphaIndicators.smart_money_flow(df) * weights['money_flow']
        
        return score

class EnhancedStrategy1(HourlyTradingStrategy):
    """
    전략 1-1: 급등 초입 + 알파 지표 강화
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Strategy1_1_EarlySurge_Alpha"
        self.alpha_indicators = AlphaIndicators()
    
    def generate_signals(self, df):
        """기존 전략 1 + 알파 지표들을 조합한 신호 생성"""
        # 기존 전략 1 신호
        base_signals = self.strategy1_early_surge(df)
        
        # 알파 지표들 계산
        volume_explosion = self.alpha_indicators.volume_explosion_detector(df)
        market_structure = self.alpha_indicators.market_structure_shift(df)
        liquidity_ok = self.alpha_indicators.liquidity_analysis(df)
        volatility_ok = self.alpha_indicators.volatility_regime_filter(df)
        money_flow = self.alpha_indicators.smart_money_flow(df)
        
        # 강화된 신호 생성
        enhanced_signals = base_signals.copy()
        
        for i in range(len(df)):
            if base_signals['signal'].iloc[i] == 1:  # 기존 신호가 있을 때
                alpha_score = 0
                
                # 알파 지표 점수 계산
                if volume_explosion.iloc[i]: alpha_score += 0.3
                if market_structure.iloc[i]: alpha_score += 0.2
                if liquidity_ok.iloc[i]: alpha_score += 0.2
                if volatility_ok.iloc[i]: alpha_score += 0.15
                if money_flow.iloc[i]: alpha_score += 0.15
                
                # 기존 신뢰도 + 알파 점수
                original_confidence = base_signals['confidence'].iloc[i]
                enhanced_confidence = min(original_confidence + alpha_score, 1.0)
                
                # 알파 점수가 0.4 이상일 때만 신호 유지
                if alpha_score >= 0.4:
                    enhanced_signals.loc[df.index[i], 'confidence'] = enhanced_confidence
                    enhanced_signals.loc[df.index[i], 'strategy'] = 'Strategy1_1_Alpha'
                else:
                    # 알파 점수가 낮으면 신호 제거
                    enhanced_signals.loc[df.index[i], 'signal'] = 0
                    enhanced_signals.loc[df.index[i], 'confidence'] = 0
        
        return enhanced_signals

class EnhancedStrategy2(HourlyTradingStrategy):
    """
    전략 2-1: 소폭 눌림목 후 초급등 + 알파 지표 강화
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Strategy2_1_Pullback_Alpha"
        self.alpha_indicators = AlphaIndicators()
    
    def generate_signals(self, df):
        """기존 전략 2 + 알파 지표들을 조합한 신호 생성"""
        # 기존 전략 2 신호
        base_signals = self.strategy2_pullback_surge(df)
        
        # 알파 지표들 계산
        fib_pullback = self.alpha_indicators.fibonacci_pullback_strength(df)
        divergence = self.alpha_indicators.bullish_divergence_detector(df)
        liquidity_ok = self.alpha_indicators.liquidity_analysis(df)
        volatility_ok = self.alpha_indicators.volatility_regime_filter(df)
        money_flow = self.alpha_indicators.smart_money_flow(df)
        
        # 강화된 신호 생성
        enhanced_signals = base_signals.copy()
        
        for i in range(len(df)):
            if base_signals['signal'].iloc[i] == 1:  # 기존 신호가 있을 때
                alpha_score = 0
                
                # 알파 지표 점수 계산 (눌림목 특화)
                if fib_pullback.iloc[i]: alpha_score += 0.35  # 피보나치 되돌림 가중치 높음
                if divergence.iloc[i]: alpha_score += 0.25   # 다이버전스 중요
                if liquidity_ok.iloc[i]: alpha_score += 0.2
                if volatility_ok.iloc[i]: alpha_score += 0.1
                if money_flow.iloc[i]: alpha_score += 0.1
                
                # 기존 신뢰도 + 알파 점수
                original_confidence = base_signals['confidence'].iloc[i]
                enhanced_confidence = min(original_confidence + alpha_score, 1.0)
                
                # 알파 점수가 0.5 이상일 때만 신호 유지 (더 엄격)
                if alpha_score >= 0.5:
                    enhanced_signals.loc[df.index[i], 'confidence'] = enhanced_confidence
                    enhanced_signals.loc[df.index[i], 'strategy'] = 'Strategy2_1_Alpha'
                else:
                    # 알파 점수가 낮으면 신호 제거
                    enhanced_signals.loc[df.index[i], 'signal'] = 0
                    enhanced_signals.loc[df.index[i], 'confidence'] = 0
        
        return enhanced_signals

class ComprehensiveStrategySystem:
    """
    4가지 전략을 통합 관리하는 시스템
    """
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        
        # 4가지 전략 초기화
        self.strategies = {
            "strategy1_basic": HourlyTradingStrategy(),      # 기존 급등 초입
            "strategy1_alpha": EnhancedStrategy1(),          # 급등 초입 + 알파
            "strategy2_basic": HourlyTradingStrategy(),      # 기존 눌림목 후 급등
            "strategy2_alpha": EnhancedStrategy2()           # 눌림목 후 급등 + 알파
        }
        
        self.strategy_weights = {
            "strategy1_basic": 0.2,
            "strategy1_alpha": 0.3,
            "strategy2_basic": 0.2,
            "strategy2_alpha": 0.3
        }
    
    def compare_strategies(self, df, commission=0.0004):
        """4가지 전략 성능 비교"""
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy_name == "strategy1_basic":
                    signals = strategy.strategy1_early_surge(df)
                elif strategy_name == "strategy2_basic":
                    signals = strategy.strategy2_pullback_surge(df)
                else:
                    signals = strategy.generate_signals(df)
                
                # 백테스트 실행
                backtest_result = self._run_backtest(df, signals, commission)
                results[strategy_name] = backtest_result
                
            except Exception as e:
                print(f"{strategy_name} 오류: {e}")
                results[strategy_name] = self._empty_result()
        
        return results
    
    def _run_backtest(self, df, signals, commission):
        """개별 전략 백테스트"""
        capital = self.initial_capital
        position = 0
        trades = []
        
        for i in range(len(df)):
            if signals['signal'].iloc[i] == 1 and position == 0:
                # 매수
                entry_price = df['close'].iloc[i]
                position_size = capital * 0.95 / entry_price
                position = position_size
                capital -= position_size * entry_price * (1 + commission)
                
                # 간단한 매도 로직 (5% 손절, 10% 익절)
                for j in range(i + 1, min(i + 24, len(df))):  # 최대 24시간 보유
                    current_price = df['close'].iloc[j]
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    if profit_pct <= -5 or profit_pct >= 10 or j == min(i + 23, len(df) - 1):
                        capital += position * current_price * (1 - commission)
                        trades.append({
                            'profit_pct': profit_pct,
                            'duration': j - i
                        })
                        position = 0
                        break
        
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t['profit_pct'] > 0]),
            'average_profit': np.mean([t['profit_pct'] for t in trades]) if trades else 0,
            'win_rate': len([t for t in trades if t['profit_pct'] > 0]) / len(trades) * 100 if trades else 0
        }
    
    def _empty_result(self):
        """빈 결과 반환"""
        return {
            'total_return': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'average_profit': 0,
            'win_rate': 0
        }

# 전략 시스템 인스턴스 생성
enhanced_strategy_system = ComprehensiveStrategySystem()

if __name__ == "__main__":
    print("🚀 강화된 4가지 전략 시스템")
    print("=" * 50)
    print("1. Strategy 1 (Basic): 급등 초입")
    print("2. Strategy 1-1 (Alpha): 급등 초입 + 알파 지표")
    print("3. Strategy 2 (Basic): 눌림목 후 급등")
    print("4. Strategy 2-1 (Alpha): 눌림목 후 급등 + 알파 지표")