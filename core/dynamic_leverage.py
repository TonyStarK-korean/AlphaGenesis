"""
동적 레버리지 관리 시스템
시장 상황과 전략에 따라 최적 레버리지를 자동 계산
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DynamicLeverageManager:
    """동적 레버리지 관리자"""
    
    def __init__(self):
        self.max_leverage = 10.0  # 최대 레버리지
        self.min_leverage = 1.0   # 최소 레버리지
        self.base_leverage = 2.0  # 기본 레버리지
        
        # 시장 국면별 레버리지 계수
        self.market_regime_multipliers = {
            'bull_strong': 1.5,    # 강한 상승장
            'bull_weak': 1.2,      # 약한 상승장
            'sideways': 1.0,       # 횡보장
            'bear_weak': 0.8,      # 약한 하락장
            'bear_strong': 0.6     # 강한 하락장
        }
        
        # 전략별 레버리지 계수
        self.strategy_multipliers = {
            'triple_combo': 1.0,
            'rsi_strategy': 0.8,
            'macd_strategy': 1.1,
            'bollinger_strategy': 1.2,
            'momentum_strategy': 1.4,
            'mean_reversion': 0.9,
            'ml_ensemble': 1.3,
            'grid_trading': 0.7,
            'arbitrage': 0.5
        }
        
        # 변동성 기반 레버리지 조정
        self.volatility_thresholds = {
            'very_low': (0, 5),      # 매우 낮음: 5% 미만
            'low': (5, 10),          # 낮음: 5-10%
            'medium': (10, 20),      # 보통: 10-20%
            'high': (20, 35),        # 높음: 20-35%
            'very_high': (35, 100)   # 매우 높음: 35% 이상
        }
        
        self.volatility_multipliers = {
            'very_low': 2.0,    # 변동성 낮으면 레버리지 높임
            'low': 1.5,
            'medium': 1.0,
            'high': 0.7,
            'very_high': 0.4    # 변동성 높으면 레버리지 낮춤
        }
        
    def calculate_optimal_leverage(
        self, 
        market_data: pd.DataFrame,
        strategy: str,
        current_position: float = 0.0,
        portfolio_value: float = 100000.0,
        risk_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        최적 레버리지 계산
        
        Args:
            market_data: 시장 데이터
            strategy: 전략명
            current_position: 현재 포지션 크기
            portfolio_value: 포트폴리오 가치
            risk_metrics: 리스크 지표들
            
        Returns:
            레버리지 정보 딕셔너리
        """
        try:
            # 데이터 유효성 검사
            if market_data.empty or len(market_data) < 2:
                return {
                    'optimal_leverage': 1.0,
                    'market_regime': 'sideways',
                    'volatility': 0.2,
                    'trend_strength': 0.0,
                    'risk_level': 'medium'
                }
            
            # 1. 시장 국면 분석
            market_regime = self._analyze_market_regime_safe(market_data)
            
            # 2. 변동성 계산
            volatility = self._calculate_volatility_safe(market_data)
            
            # 3. 트렌드 강도 계산
            trend_strength = self._calculate_trend_strength_safe(market_data)
            
            # 4. 기본 레버리지 계산
            base_leverage = self.base_leverage
            
            # 5. 시장 국면별 조정
            market_multiplier = self.market_regime_multipliers.get(market_regime, 1.0)
            
            # 6. 전략별 조정
            strategy_multiplier = self.strategy_multipliers.get(strategy, 1.0)
            
            # 7. 변동성별 조정
            volatility_level = self._get_volatility_level(volatility)
            volatility_multiplier = self.volatility_multipliers.get(volatility_level, 1.0)
            
            # 8. 트렌드 강도별 조정
            trend_multiplier = self._get_trend_multiplier(trend_strength)
            
            # 9. 포지션 크기 기반 조정
            position_multiplier = self._get_position_multiplier(current_position, portfolio_value)
            
            # 10. 리스크 기반 조정
            risk_multiplier = self._get_risk_multiplier(risk_metrics or {})
            
            # 11. 최종 레버리지 계산
            optimal_leverage = (
                base_leverage * 
                market_multiplier * 
                strategy_multiplier * 
                volatility_multiplier * 
                trend_multiplier * 
                position_multiplier * 
                risk_multiplier
            )
            
            # 12. 최소/최대 레버리지 제한
            optimal_leverage = max(self.min_leverage, min(self.max_leverage, optimal_leverage))
            
            # 13. 결과 반환
            return {
                'optimal_leverage': round(optimal_leverage, 2),
                'market_regime': market_regime,
                'volatility': round(volatility, 2),
                'volatility_level': volatility_level,
                'trend_strength': round(trend_strength, 2),
                'components': {
                    'base': base_leverage,
                    'market': market_multiplier,
                    'strategy': strategy_multiplier,
                    'volatility': volatility_multiplier,
                    'trend': trend_multiplier,
                    'position': position_multiplier,
                    'risk': risk_multiplier
                },
                'recommendation': self._get_leverage_recommendation(optimal_leverage)
            }
            
        except Exception as e:
            logger.error(f"레버리지 계산 실패: {e}")
            return {
                'optimal_leverage': self.base_leverage,
                'error': str(e)
            }
    
    def _analyze_market_regime(self, data: pd.DataFrame) -> str:
        """시장 국면 분석"""
        try:
            # 이동평균선 계산
            data['MA20'] = data['close'].rolling(20).mean()
            data['MA50'] = data['close'].rolling(50).mean()
            data['MA200'] = data['close'].rolling(200).mean()
            
            # 현재 가격과 이동평균선 비교
            current_price = data['close'].iloc[-1]
            ma20 = data['MA20'].iloc[-1]
            ma50 = data['MA50'].iloc[-1]
            ma200 = data['MA200'].iloc[-1]
            
            # 이동평균선 정렬 확인
            ma_alignment = (ma20 > ma50 > ma200)
            
            # 가격 모멘텀 계산
            price_momentum = (current_price - data['close'].iloc[-20]) / data['close'].iloc[-20]
            
            # 시장 국면 판단
            if ma_alignment and price_momentum > 0.05:
                return 'bull_strong'
            elif current_price > ma20 and price_momentum > 0.02:
                return 'bull_weak'
            elif current_price < ma20 and price_momentum < -0.05:
                return 'bear_strong'
            elif current_price < ma20 and price_momentum < -0.02:
                return 'bear_weak'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"시장 국면 분석 실패: {e}")
            return 'sideways'
    
    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """변동성 계산 (ATR 기반)"""
        try:
            # ATR 계산
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(period).mean()
            
            # 변동성을 백분율로 변환
            volatility = (atr.iloc[-1] / data['close'].iloc[-1]) * 100
            
            return volatility
            
        except Exception as e:
            logger.error(f"변동성 계산 실패: {e}")
            return 15.0  # 기본값
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """트렌드 강도 계산"""
        try:
            # ADX 계산 (간단한 버전)
            period = 14
            
            # 방향성 지수 계산
            plus_dm = data['high'].diff()
            minus_dm = data['low'].diff() * -1
            
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
            
            # True Range 계산
            tr = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    np.abs(data['high'] - data['close'].shift()),
                    np.abs(data['low'] - data['close'].shift())
                )
            )
            
            # 평활화
            plus_di = (plus_dm.rolling(period).mean() / tr.rolling(period).mean()) * 100
            minus_di = (minus_dm.rolling(period).mean() / tr.rolling(period).mean()) * 100
            
            # ADX 계산
            dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(period).mean()
            
            return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 25.0
            
        except Exception as e:
            logger.error(f"트렌드 강도 계산 실패: {e}")
            return 25.0  # 기본값
    
    def _get_volatility_level(self, volatility: float) -> str:
        """변동성 레벨 결정"""
        for level, (min_vol, max_vol) in self.volatility_thresholds.items():
            if min_vol <= volatility < max_vol:
                return level
        return 'medium'
    
    def _get_trend_multiplier(self, trend_strength: float) -> float:
        """트렌드 강도에 따른 레버리지 배수"""
        if trend_strength > 40:
            return 1.3  # 강한 트렌드
        elif trend_strength > 25:
            return 1.1  # 중간 트렌드
        elif trend_strength > 15:
            return 1.0  # 약한 트렌드
        else:
            return 0.8  # 트렌드 없음
    
    def _get_position_multiplier(self, position: float, portfolio_value: float) -> float:
        """포지션 크기에 따른 레버리지 배수"""
        position_ratio = abs(position) / portfolio_value
        
        if position_ratio > 0.8:
            return 0.5  # 큰 포지션이면 레버리지 낮춤
        elif position_ratio > 0.5:
            return 0.7
        elif position_ratio > 0.3:
            return 0.9
        else:
            return 1.0
    
    def _get_risk_multiplier(self, risk_metrics: Dict) -> float:
        """리스크 지표에 따른 레버리지 배수"""
        multiplier = 1.0
        
        # 최대 낙폭 기반 조정
        max_drawdown = risk_metrics.get('max_drawdown', 0)
        if max_drawdown > 20:
            multiplier *= 0.6
        elif max_drawdown > 15:
            multiplier *= 0.8
        elif max_drawdown > 10:
            multiplier *= 0.9
        
        # 샤프 비율 기반 조정
        sharpe_ratio = risk_metrics.get('sharpe_ratio', 1.0)
        if sharpe_ratio > 2.0:
            multiplier *= 1.2
        elif sharpe_ratio > 1.5:
            multiplier *= 1.1
        elif sharpe_ratio < 0.5:
            multiplier *= 0.7
        
        # 승률 기반 조정
        win_rate = risk_metrics.get('win_rate', 50)
        if win_rate > 70:
            multiplier *= 1.1
        elif win_rate < 40:
            multiplier *= 0.8
        
        return multiplier
    
    def _get_leverage_recommendation(self, leverage: float) -> str:
        """레버리지 추천 메시지"""
        if leverage >= 5.0:
            return "🔥 공격적 레버리지 - 높은 수익 잠재력, 주의 필요"
        elif leverage >= 3.0:
            return "⚡ 적극적 레버리지 - 균형잡힌 위험-수익"
        elif leverage >= 2.0:
            return "✅ 안정적 레버리지 - 적절한 위험 수준"
        else:
            return "🛡️ 보수적 레버리지 - 안전 우선"
    
    def _analyze_market_regime_safe(self, data: pd.DataFrame) -> str:
        """시장 국면 분석 - 안전한 버전"""
        try:
            if data.empty or len(data) < 2:
                return 'sideways'
            
            # 간단한 트렌드 분석
            close_prices = data['close']
            if len(close_prices) < 2:
                return 'sideways'
            
            # 최근 가격 변화율
            recent_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100
            
            if recent_change > 5:
                return 'bull_strong'
            elif recent_change > 2:
                return 'bull_weak'
            elif recent_change > -2:
                return 'sideways'
            elif recent_change > -5:
                return 'bear_weak'
            else:
                return 'bear_strong'
                
        except Exception as e:
            logger.error(f"시장 국면 분석 실패: {e}")
            return 'sideways'
    
    def _calculate_volatility_safe(self, data: pd.DataFrame) -> float:
        """변동성 계산 - 안전한 버전"""
        try:
            if data.empty or len(data) < 2:
                return 0.2
            
            # 간단한 변동성 계산
            close_prices = data['close']
            if len(close_prices) < 2:
                return 0.2
            
            # 가격 변화율의 표준편차
            returns = close_prices.pct_change().dropna()
            if len(returns) > 0:
                return float(returns.std()) * 100  # 백분율로 변환
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"변동성 계산 실패: {e}")
            return 0.2
    
    def _calculate_trend_strength_safe(self, data: pd.DataFrame) -> float:
        """트렌드 강도 계산 - 안전한 버전"""
        try:
            if data.empty or len(data) < 2:
                return 0.0
            
            # 간단한 트렌드 강도 계산
            close_prices = data['close']
            if len(close_prices) < 2:
                return 0.0
            
            # 선형 회귀를 이용한 트렌드 강도
            x = np.arange(len(close_prices))
            y = close_prices.values
            
            # 상관계수를 이용한 트렌드 강도
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
            
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"트렌드 강도 계산 실패: {e}")
            return 0.0
    
    def _assess_risk_level_safe(self, leverage: float, volatility: float, trend_strength: float) -> str:
        """리스크 레벨 평가 - 안전한 버전"""
        try:
            # 리스크 점수 계산 (간단한 버전)
            risk_score = (leverage - 1.0) * 10 + volatility * 100 + (1 - trend_strength) * 20
            
            if risk_score < 30:
                return 'low'
            elif risk_score < 60:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            logger.error(f"리스크 레벨 평가 실패: {e}")
            return 'medium'

class SmartPositionManager:
    """지능형 포지션 관리자"""
    
    def __init__(self):
        self.max_add_count = 5  # 최대 추가매수 횟수
        self.base_position_ratio = 0.02  # 기본 포지션 비율 (2%)
        
    def calculate_position_size(
        self,
        signal_strength: float,
        portfolio_value: float,
        leverage: float,
        market_condition: str,
        is_additional_buy: bool = False,
        current_position_count: int = 0
    ) -> Dict:
        """포지션 크기 계산"""
        
        # 기본 포지션 크기
        base_size = portfolio_value * self.base_position_ratio
        
        # 신호 강도에 따른 조정
        signal_multiplier = min(2.0, max(0.5, signal_strength / 100))
        
        # 시장 상황에 따른 조정
        market_multipliers = {
            'bull_strong': 1.5,
            'bull_weak': 1.2,
            'sideways': 1.0,
            'bear_weak': 0.8,
            'bear_strong': 0.6
        }
        market_multiplier = market_multipliers.get(market_condition, 1.0)
        
        # 추가매수인 경우 조정
        if is_additional_buy:
            add_multiplier = max(0.5, 1.0 - (current_position_count * 0.2))
        else:
            add_multiplier = 1.0
        
        # 최종 포지션 크기
        position_size = base_size * signal_multiplier * market_multiplier * add_multiplier * leverage
        
        return {
            'position_size': round(position_size, 2),
            'base_size': base_size,
            'signal_multiplier': signal_multiplier,
            'market_multiplier': market_multiplier,
            'add_multiplier': add_multiplier,
            'leverage': leverage
        }
    
    def should_add_position(
        self,
        entry_price: float,
        current_price: float,
        position_type: str,  # 'LONG' or 'SHORT'
        current_pnl_percent: float,
        add_count: int,
        market_condition: str
    ) -> Dict:
        """추가매수 여부 판단"""
        
        # 최대 추가매수 횟수 체크
        if add_count >= self.max_add_count:
            return {'should_add': False, 'reason': '최대 추가매수 횟수 초과'}
        
        # 시장 상황에 따른 추가매수 기준
        add_thresholds = {
            'bull_strong': [-0.5, -1.0, -1.5],
            'bull_weak': [-1.0, -2.0, -3.0],
            'sideways': [-1.5, -2.5, -3.5],
            'bear_weak': [-2.0, -3.0, -4.0],
            'bear_strong': [-3.0, -4.0, -5.0]
        }
        
        thresholds = add_thresholds.get(market_condition, [-1.5, -2.5, -3.5])
        
        # 현재 추가매수 차수에 따른 기준
        if add_count < len(thresholds):
            threshold = thresholds[add_count]
        else:
            return {'should_add': False, 'reason': '추가매수 기준 초과'}
        
        # 손실 상황에서만 추가매수
        if position_type == 'LONG':
            price_change = (current_price - entry_price) / entry_price * 100
        else:
            price_change = (entry_price - current_price) / entry_price * 100
        
        if price_change <= threshold:
            return {
                'should_add': True,
                'reason': f'{add_count + 1}차 추가매수 조건 충족 (손실: {price_change:.2f}%)',
                'add_ratio': max(0.5, 1.0 - (add_count * 0.25))  # 차수가 늘어날수록 비중 감소
            }
        
        return {'should_add': False, 'reason': '추가매수 기준 미달'}
    
    def calculate_sell_schedule(
        self,
        total_position: float,
        entry_price: float,
        current_price: float,
        profit_targets: List[float] = None
    ) -> List[Dict]:
        """분할매도 스케줄 계산"""
        
        if profit_targets is None:
            profit_targets = [2.0, 4.0, 6.0, 8.0]  # 기본 수익 목표
        
        sell_schedule = []
        remaining_position = total_position
        
        # 분할매도 비율 (첫 번째부터 마지막까지)
        sell_ratios = [0.25, 0.30, 0.25, 0.20]  # 25%, 30%, 25%, 20%
        
        for i, (target, ratio) in enumerate(zip(profit_targets, sell_ratios)):
            sell_amount = total_position * ratio
            
            sell_schedule.append({
                'stage': i + 1,
                'profit_target': target,
                'sell_amount': sell_amount,
                'sell_ratio': ratio,
                'remaining_after_sell': remaining_position - sell_amount
            })
            
            remaining_position -= sell_amount
            
            if remaining_position <= 0:
                break
        
        return sell_schedule
    
    def _analyze_market_regime_safe(self, data: pd.DataFrame) -> str:
        """시장 국면 분석 - 안전한 버전"""
        try:
            if data.empty or len(data) < 2:
                return 'sideways'
            
            # 간단한 트렌드 분석
            close_prices = data['close']
            if len(close_prices) < 2:
                return 'sideways'
            
            # 최근 가격 변화율
            recent_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100
            
            if recent_change > 5:
                return 'bull_strong'
            elif recent_change > 2:
                return 'bull_weak'
            elif recent_change > -2:
                return 'sideways'
            elif recent_change > -5:
                return 'bear_weak'
            else:
                return 'bear_strong'
                
        except Exception as e:
            logger.error(f"시장 국면 분석 실패: {e}")
            return 'sideways'
    
    def _calculate_volatility_safe(self, data: pd.DataFrame) -> float:
        """변동성 계산 - 안전한 버전"""
        try:
            if data.empty or len(data) < 2:
                return 0.2
            
            # 간단한 변동성 계산
            close_prices = data['close']
            if len(close_prices) < 2:
                return 0.2
            
            # 가격 변화율의 표준편차
            returns = close_prices.pct_change().dropna()
            if len(returns) > 0:
                return float(returns.std()) * 100  # 백분율로 변환
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"변동성 계산 실패: {e}")
            return 0.2
    
    def _calculate_trend_strength_safe(self, data: pd.DataFrame) -> float:
        """트렌드 강도 계산 - 안전한 버전"""
        try:
            if data.empty or len(data) < 2:
                return 0.0
            
            # 간단한 트렌드 강도 계산
            close_prices = data['close']
            if len(close_prices) < 2:
                return 0.0
            
            # 선형 회귀를 이용한 트렌드 강도
            x = np.arange(len(close_prices))
            y = close_prices.values
            
            # 상관계수를 이용한 트렌드 강도
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
            
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"트렌드 강도 계산 실패: {e}")
            return 0.0
    
    def _assess_risk_level_safe(self, leverage: float, volatility: float, trend_strength: float) -> str:
        """리스크 레벨 평가 - 안전한 버전"""
        try:
            # 리스크 점수 계산 (간단한 버전)
            risk_score = (leverage - 1.0) * 10 + volatility * 100 + (1 - trend_strength) * 20
            
            if risk_score < 30:
                return 'low'
            elif risk_score < 60:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            logger.error(f"리스크 레벨 평가 실패: {e}")
            return 'medium'