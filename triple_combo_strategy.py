#!/usr/bin/env python3
"""
🚀 트리플 콤보 전략 시스템
3가지 핵심 전략의 완벽한 조합으로 모든 시장 상황에 대응
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 🚀 트리플 콤보 전략 클래스들 (메인 전략 엔진)
# ==============================================

class TrendFollowingStrategy:
    """
    📈 전략 1: 추세 순응형 R/R 극대화 전략
    - 목표: 상승/하락장에서 큰 추세를 따라가며 손실은 짧게, 수익은 길게
    - 예상 승률: 55-65%
    - 예상 손익비: 1:2.5 이상
    """
    
    def __init__(self, params=None):
        self.name = "TrendFollowing_RR"
        self.params = params or {
            'ma_short': 20,
            'ma_long': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'min_trend_strength': 0.6,
            'obv_confirmation_period': 10
        }
        
    def detect_trend(self, df):
        """추세 방향 감지"""
        try:
            ma_short = df['close'].rolling(self.params['ma_short']).mean()
            ma_long = df['close'].rolling(self.params['ma_long']).mean()
            
            # 추세 방향 (1: 상승, -1: 하락, 0: 횡보)
            trend = np.where(ma_short > ma_long, 1, 
                           np.where(ma_short < ma_long, -1, 0))
            
            # 추세 강도 (이동평균선 간 거리로 측정)
            trend_strength = abs(ma_short - ma_long) / ma_long
            
            return pd.Series(trend, index=df.index), pd.Series(trend_strength, index=df.index)
            
        except Exception as e:
            print(f"추세 감지 오류: {e}")
            return pd.Series(0, index=df.index), pd.Series(0, index=df.index)
    
    def generate_signal(self, row, ml_pred, market_condition):
        """추세 순응형 신호 생성"""
        try:
            signal = {
                'signal': 0,
                'confidence': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'leverage_suggestion': 1.0,
                'strategy': self.name
            }
            
            # 기본 데이터 추출
            close = row['close']
            rsi = row.get('rsi_14', 50)
            atr = row.get('atr_14', close * 0.02)
            
            # 추세 정보 (이미 계산되어 있다고 가정)
            trend = row.get('trend_direction', 0)
            trend_strength = row.get('trend_strength', 0)
            
            # === 1. 추세 필터 ===
            if trend == 0 or trend_strength < self.params['min_trend_strength']:
                return signal  # 추세가 없으면 신호 없음
            
            # === 2. 진입 조건 확인 ===
            entry_conditions = []
            
            # 상승 추세에서의 진입 조건
            if trend == 1:
                # 조건 1: RSI 과매도 구간에서 반등 (눌림목 매수)
                if 20 <= rsi <= 45:
                    entry_conditions.append(('rsi_pullback', 0.3))
                
                # 조건 2: ML 예측 상승 신호
                if ml_pred > 0.01:
                    entry_conditions.append(('ml_bullish', 0.25))
                
                # 조건 3: 거래량 확인 (OBV 상승)
                obv_trend = row.get('obv_trend', 0)
                if obv_trend > 0:
                    entry_conditions.append(('volume_confirm', 0.2))
                
                # 조건 4: 지지선 근처 (볼린저 밴드 하단 근처)
                bb_position = row.get('bb_position', 0.5)
                if bb_position < 0.3:
                    entry_conditions.append(('support_level', 0.25))
                
            # 하락 추세에서의 진입 조건
            elif trend == -1:
                # 조건 1: RSI 과매수 구간에서 반락 (되돌림 매도)
                if 55 <= rsi <= 80:
                    entry_conditions.append(('rsi_pullback', 0.3))
                
                # 조건 2: ML 예측 하락 신호
                if ml_pred < -0.01:
                    entry_conditions.append(('ml_bearish', 0.25))
                
                # 조건 3: 거래량 확인 (OBV 하락)
                obv_trend = row.get('obv_trend', 0)
                if obv_trend < 0:
                    entry_conditions.append(('volume_confirm', 0.2))
                
                # 조건 4: 저항선 근처 (볼린저 밴드 상단 근처)
                bb_position = row.get('bb_position', 0.5)
                if bb_position > 0.7:
                    entry_conditions.append(('resistance_level', 0.25))
            
            # === 3. 신호 생성 ===
            if len(entry_conditions) >= 2:  # 최소 2개 조건 만족
                total_confidence = sum([weight for _, weight in entry_conditions])
                
                if total_confidence >= 0.5:
                    signal['signal'] = trend
                    signal['confidence'] = min(total_confidence, 1.0)
                    
                    # 손익비 설정 (R/R 극대화)
                    stop_loss_distance = atr * self.params['stop_loss_atr']
                    take_profit_distance = atr * self.params['take_profit_atr']
                    
                    # 추세 강도에 따른 손익비 조정
                    strength_multiplier = 1.0 + (trend_strength * 2.0)
                    take_profit_distance *= strength_multiplier
                    
                    if trend == 1:  # 롱
                        signal['stop_loss'] = close - stop_loss_distance
                        signal['take_profit'] = close + take_profit_distance
                    else:  # 숏
                        signal['stop_loss'] = close + stop_loss_distance
                        signal['take_profit'] = close - take_profit_distance
                    
                    # 레버리지 제안 (신뢰도 + 추세 강도 기반)
                    base_leverage = 2.0 + (signal['confidence'] * 2.0)
                    signal['leverage_suggestion'] = min(base_leverage * strength_multiplier, 5.0)
            
            return signal
            
        except Exception as e:
            print(f"추세 신호 생성 오류: {e}")
            return {'signal': 0, 'confidence': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage_suggestion': 1.0, 'strategy': self.name}


class CVDScalpingStrategy:
    """
    🔄 전략 2: CVD 기반 스캘핑 전략
    - 목표: 횡보장에서 매수/매도 압력 분석으로 짧은 수익 반복
    - 예상 승률: 70-80%
    - 예상 손익비: 1:1.2
    """
    
    def __init__(self, params=None):
        self.name = "CVD_Scalping"
        self.params = params or {
            'cvd_threshold': 0.3,
            'rsi_period': 14,
            'rsi_scalp_buy': 45,
            'rsi_scalp_sell': 55,
            'atr_period': 14,
            'scalp_target_atr': 0.8,
            'scalp_stop_atr': 0.6,
            'volume_spike_threshold': 1.5,
            'max_hold_periods': 5
        }
        
    def detect_sideways_market(self, df):
        """횡보장 감지"""
        try:
            # ADX로 추세 강도 측정
            adx = df.get('adx_14', pd.Series(25, index=df.index))
            
            # 볼린저 밴드 폭으로 변동성 측정
            bb_width = df.get('bb_width', pd.Series(0.05, index=df.index))
            
            # 횡보 조건: ADX < 25 and 낮은 변동성
            is_sideways = (adx < 25) & (bb_width < 0.04)
            
            return is_sideways
            
        except Exception as e:
            print(f"횡보장 감지 오류: {e}")
            return pd.Series(False, index=df.index)
    
    def generate_signal(self, row, ml_pred, market_condition):
        """CVD 기반 스캘핑 신호 생성"""
        try:
            signal = {
                'signal': 0,
                'confidence': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'leverage_suggestion': 1.0,
                'strategy': self.name
            }
            
            # 기본 데이터 추출
            close = row['close']
            rsi = row.get('rsi_14', 50)
            atr = row.get('atr_14', close * 0.02)
            
            # CVD 관련 데이터
            cvd = row.get('cvd', 0)
            cvd_sma = row.get('cvd_sma', 0)
            volume_ratio = row.get('volume_ratio', 1.0)
            
            # === 1. 횡보장 필터 ===
            is_sideways = row.get('is_sideways', False)
            if not is_sideways:
                return signal  # 횡보장이 아니면 스캘핑 안함
            
            # === 2. CVD 분석 ===
            cvd_momentum = cvd - cvd_sma
            cvd_strength = abs(cvd_momentum) / (abs(cvd_sma) + 1e-8)
            
            # === 3. 진입 조건 확인 ===
            entry_conditions = []
            
            # 매수 신호 조건들
            if cvd_momentum > self.params['cvd_threshold'] and cvd_strength > 0.2:
                # 조건 1: 강한 매수 압력
                entry_conditions.append(('cvd_bullish', 0.4))
                
                # 조건 2: RSI 과매도 구간
                if rsi < self.params['rsi_scalp_buy']:
                    entry_conditions.append(('rsi_oversold', 0.3))
                
                # 조건 3: 거래량 급증
                if volume_ratio > self.params['volume_spike_threshold']:
                    entry_conditions.append(('volume_spike', 0.2))
                
                # 조건 4: ML 예측 지지
                if ml_pred > 0:
                    entry_conditions.append(('ml_support', 0.1))
                
                potential_signal = 1
                
            # 매도 신호 조건들
            elif cvd_momentum < -self.params['cvd_threshold'] and cvd_strength > 0.2:
                # 조건 1: 강한 매도 압력
                entry_conditions.append(('cvd_bearish', 0.4))
                
                # 조건 2: RSI 과매수 구간
                if rsi > self.params['rsi_scalp_sell']:
                    entry_conditions.append(('rsi_overbought', 0.3))
                
                # 조건 3: 거래량 급증
                if volume_ratio > self.params['volume_spike_threshold']:
                    entry_conditions.append(('volume_spike', 0.2))
                
                # 조건 4: ML 예측 지지
                if ml_pred < 0:
                    entry_conditions.append(('ml_support', 0.1))
                
                potential_signal = -1
            else:
                return signal
            
            # === 4. 신호 생성 ===
            if len(entry_conditions) >= 2:  # 최소 2개 조건 만족
                total_confidence = sum([weight for _, weight in entry_conditions])
                
                if total_confidence >= 0.6:  # 스캘핑은 높은 확신 필요
                    signal['signal'] = potential_signal
                    signal['confidence'] = min(total_confidence, 1.0)
                    
                    # 타이트한 손익비 설정 (스캘핑 특성)
                    stop_distance = atr * self.params['scalp_stop_atr']
                    target_distance = atr * self.params['scalp_target_atr']
                    
                    if potential_signal == 1:  # 롱
                        signal['stop_loss'] = close - stop_distance
                        signal['take_profit'] = close + target_distance
                    else:  # 숏
                        signal['stop_loss'] = close + stop_distance
                        signal['take_profit'] = close - target_distance
                    
                    # 높은 레버리지 (높은 승률 + 타이트한 손절)
                    signal['leverage_suggestion'] = min(3.0 + signal['confidence'], 5.0)
            
            return signal
            
        except Exception as e:
            print(f"CVD 스캘핑 신호 생성 오류: {e}")
            return {'signal': 0, 'confidence': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage_suggestion': 1.0, 'strategy': self.name}


class VolatilityBreakoutStrategy:
    """
    💥 전략 3: 변동성 돌파 전략
    - 목표: 급등/급락장 초입에서 변동성 폭발을 포착
    - 예상 승률: 45-55%
    - 예상 손익비: 1:3.0 이상
    """
    
    def __init__(self, params=None):
        self.name = "Volatility_Breakout"
        self.params = params or {
            'bb_period': 20,
            'bb_std': 2.0,
            'squeeze_threshold': 0.02,  # 볼린저 밴드 폭 임계값
            'squeeze_duration': 10,      # 최소 수축 기간
            'breakout_strength': 0.5,    # 돌파 강도
            'atr_period': 14,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 4.0,
            'volume_confirmation': 1.5
        }
        
    def detect_squeeze(self, df):
        """변동성 수축 (Squeeze) 감지"""
        try:
            # 볼린저 밴드 폭 계산
            bb_width = df.get('bb_width', pd.Series(0.05, index=df.index))
            
            # 변동성 수축 조건
            is_squeeze = bb_width < self.params['squeeze_threshold']
            
            # 수축 지속 기간 계산
            squeeze_duration = is_squeeze.rolling(window=self.params['squeeze_duration']).sum()
            
            # 충분한 기간 동안 수축된 상태
            valid_squeeze = squeeze_duration >= self.params['squeeze_duration']
            
            return valid_squeeze
            
        except Exception as e:
            print(f"변동성 수축 감지 오류: {e}")
            return pd.Series(False, index=df.index)
    
    def generate_signal(self, row, ml_pred, market_condition):
        """변동성 돌파 신호 생성"""
        try:
            signal = {
                'signal': 0,
                'confidence': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'leverage_suggestion': 1.0,
                'strategy': self.name
            }
            
            # 기본 데이터 추출
            close = row['close']
            high = row['high']
            low = row['low']
            atr = row.get('atr_14', close * 0.02)
            volume_ratio = row.get('volume_ratio', 1.0)
            
            # 볼린저 밴드 정보
            bb_upper = row.get('bb_upper', close * 1.02)
            bb_lower = row.get('bb_lower', close * 0.98)
            bb_width = row.get('bb_width', 0.05)
            
            # === 1. 변동성 수축 조건 확인 ===
            was_squeezed = row.get('was_squeezed', False)
            if not was_squeezed:
                return signal  # 사전 수축이 없었으면 돌파 신호 없음
            
            # === 2. 돌파 강도 계산 ===
            upper_breakout_strength = max(0, (close - bb_upper) / bb_upper)
            lower_breakout_strength = max(0, (bb_lower - close) / bb_lower)
            
            # === 3. 진입 조건 확인 ===
            entry_conditions = []
            potential_signal = 0
            
            # 상향 돌파 조건들
            if upper_breakout_strength > self.params['breakout_strength']:
                entry_conditions.append(('upper_breakout', 0.4))
                potential_signal = 1
                
                # 추가 확인 조건들
                if volume_ratio > self.params['volume_confirmation']:
                    entry_conditions.append(('volume_confirm', 0.3))
                
                if ml_pred > 0.02:
                    entry_conditions.append(('ml_bullish', 0.2))
                
                # 캔들 패턴 확인 (강한 상승 캔들)
                if (close - row['open']) / row['open'] > 0.01:
                    entry_conditions.append(('strong_candle', 0.1))
            
            # 하향 돌파 조건들
            elif lower_breakout_strength > self.params['breakout_strength']:
                entry_conditions.append(('lower_breakout', 0.4))
                potential_signal = -1
                
                # 추가 확인 조건들
                if volume_ratio > self.params['volume_confirmation']:
                    entry_conditions.append(('volume_confirm', 0.3))
                
                if ml_pred < -0.02:
                    entry_conditions.append(('ml_bearish', 0.2))
                
                # 캔들 패턴 확인 (강한 하락 캔들)
                if (row['open'] - close) / row['open'] > 0.01:
                    entry_conditions.append(('strong_candle', 0.1))
            
                # === 급락장 성공률 향상을 위한 필터링 조건 추가 ===
                # 1. 하락 추세 확인 (단기 이평선이 장기 이평선 아래에 있는지)
                if row.get('ma_20', close) > row.get('ma_50', close):
                    # 상승 추세 중의 일시적 하락 돌파는 무시
                    return signal
                # 2. 음봉 캔들 확인 (돌파 캔들이 음봉이어야 함)
                if close > row['open']:
                    # 양봉 돌파는 신뢰도가 낮으므로 무시
                    return signal
            
            # === 4. 신호 생성 ===
            if len(entry_conditions) >= 2:  # 최소 2개 조건 만족
                total_confidence = sum([weight for _, weight in entry_conditions])
                
                if total_confidence >= 0.6:
                    signal['signal'] = potential_signal
                    signal['confidence'] = min(total_confidence, 1.0)
                    
                    # 넓은 손익비 설정 (홈런 전략)
                    stop_distance = atr * self.params['stop_loss_atr']
                    target_distance = atr * self.params['take_profit_atr']
                    
                    # 돌파 강도에 따른 손익비 조정
                    breakout_strength = max(upper_breakout_strength, lower_breakout_strength)
                    strength_multiplier = 1.0 + (breakout_strength * 3.0)
                    target_distance *= strength_multiplier
                    
                    if potential_signal == 1:  # 롱
                        signal['stop_loss'] = close - stop_distance
                        signal['take_profit'] = close + target_distance
                    else:  # 숏
                        signal['stop_loss'] = close + stop_distance
                        signal['take_profit'] = close - target_distance
                    
                    # 보수적 레버리지 (낮은 승률 보상)
                    signal['leverage_suggestion'] = min(2.0 + signal['confidence'], 4.0)
            
            return signal
            
        except Exception as e:
            print(f"변동성 돌파 신호 생성 오류: {e}")
            return {'signal': 0, 'confidence': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage_suggestion': 1.0, 'strategy': self.name}


class TripleComboStrategy:
    """
    🚀 트리플 콤보 전략 매니저
    - 3가지 전략을 시장 상황에 따라 자동 선택
    - 각 전략의 신호를 종합하여 최적의 매매 결정
    """
    
    def __init__(self, params=None):
        self.name = "Triple_Combo"
        self.strategies = {
            'trend': TrendFollowingStrategy(),
            'scalping': CVDScalpingStrategy(), 
            'breakout': VolatilityBreakoutStrategy()
        }
        self.params = params or {
            'trend_priority': 0.5,      # 추세 전략 우선순위
            'scalping_priority': 0.3,   # 스캘핑 전략 우선순위
            'breakout_priority': 0.2,   # 돌파 전략 우선순위
            'min_confidence': 0.6,      # 최소 신뢰도
            'max_concurrent_signals': 2  # 동시 신호 최대 개수
        }
        self.last_strategy = "unknown"
        
    def analyze_market_phase(self, row, df_recent):
        """시장 국면 분석"""
        try:
            # 추세 강도
            trend_strength = row.get('trend_strength', 0)
            
            # 변동성 수준
            volatility = row.get('volatility_20', 0.05)
            
            # ADX (추세 강도)
            adx = row.get('adx_14', 25)
            
            # 시장 국면 판단
            if adx > 30 and trend_strength > 0.3:
                return 'trending'  # 추세장
            elif volatility < 0.03 and adx < 20:
                return 'sideways'  # 횡보장
            elif volatility > 0.08:
                return 'volatile'  # 변동성 장
            else:
                return 'mixed'     # 복합적
                
        except Exception as e:
            print(f"시장 국면 분석 오류: {e}")
            return 'mixed'
    
    def generate_signal(self, row, ml_pred, market_condition, df_recent=None):
        """통합 신호 생성"""
        try:
            # 각 전략별 신호 생성
            signals = {}
            
            # 시장 국면 분석
            market_phase = self.analyze_market_phase(row, df_recent)
            
            # 추세 전략 신호
            trend_signal = self.strategies['trend'].generate_signal(row, ml_pred, market_condition)
            if trend_signal['signal'] != 0:
                signals['trend'] = trend_signal
            
            # 스캘핑 전략 신호
            scalping_signal = self.strategies['scalping'].generate_signal(row, ml_pred, market_condition)
            if scalping_signal['signal'] != 0:
                signals['scalping'] = scalping_signal
            
            # 돌파 전략 신호
            breakout_signal = self.strategies['breakout'].generate_signal(row, ml_pred, market_condition)
            if breakout_signal['signal'] != 0:
                signals['breakout'] = breakout_signal
            
            # 신호 없음
            if len(signals) == 0:
                return {'signal': 0, 'confidence': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage_suggestion': 1.0, 'strategy': self.name}
            
            # 시장 국면에 따른 전략 우선순위 조정
            priorities = self.params.copy()
            if market_phase == 'trending':
                priorities['trend_priority'] = 0.6
                priorities['scalping_priority'] = 0.2
                priorities['breakout_priority'] = 0.2
            elif market_phase == 'sideways':
                priorities['trend_priority'] = 0.2
                priorities['scalping_priority'] = 0.6
                priorities['breakout_priority'] = 0.2
            elif market_phase == 'volatile':
                priorities['trend_priority'] = 0.3
                priorities['scalping_priority'] = 0.2
                priorities['breakout_priority'] = 0.5
            
            # 최고 신뢰도 신호 선택
            best_signal = None
            best_score = 0
            
            for strategy_name, signal in signals.items():
                priority = priorities.get(f'{strategy_name}_priority', 0.33)
                score = signal['confidence'] * priority
                
                if score > best_score and signal['confidence'] >= self.params['min_confidence']:
                    best_score = score
                    best_signal = signal.copy()
                    best_signal['strategy'] = f"{self.name}_{strategy_name}"
                    best_signal['market_phase'] = market_phase
                    self.last_strategy = strategy_name
            
            if best_signal is None:
                return {'signal': 0, 'confidence': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage_suggestion': 1.0, 'strategy': self.name}
            
            return best_signal
            
        except Exception as e:
            print(f"트리플 콤보 신호 생성 오류: {e}")
            return {'signal': 0, 'confidence': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage_suggestion': 1.0, 'strategy': self.name}


# ==============================================
# 🎯 트리플 콤보 백테스트 엔진
# ==============================================

def check_position_exit(row, position, entry_price, stop_loss, take_profit):
    """포지션 청산 조건 확인"""
    current_price = row['close']
    
    # 손절매 확인
    if position == 1 and current_price <= stop_loss:
        return True, "stop_loss"
    elif position == -1 and current_price >= stop_loss:
        return True, "stop_loss"
    
    # 익절매 확인
    if position == 1 and current_price >= take_profit:
        return True, "take_profit"
    elif position == -1 and current_price <= take_profit:
        return True, "take_profit"
    
    return False, None


def calculate_pnl(position, entry_price, exit_price, position_size, leverage):
    """손익 계산"""
    if position == 1:  # 롱
        price_change = (exit_price - entry_price) / entry_price
    else:  # 숏
        price_change = (entry_price - exit_price) / entry_price
    
    return position_size * price_change * leverage


def print_detailed_trade_log(trade_record):
    """상세 거래 로그 출력"""
    print(f"\n{'='*60}")
    print(f"📋 거래 상세 로그")
    print(f"{'='*60}")
    print(f"⏰ 진입 시간: {trade_record['entry_time']}")
    print(f"⏰ 청산 시간: {trade_record['exit_time']}")
    print(f"🎯 전략: {trade_record['strategy']}")
    print(f"📍 포지션: {'롱(매수)' if trade_record['position'] == 1 else '숏(매도)'}")
    print(f"💰 진입가: {trade_record['entry_price']:.4f}")
    print(f"💰 청산가: {trade_record['exit_price']:.4f}")
    print(f"📊 포지션 크기: {trade_record['size']:,.0f}")
    print(f"⚖️  레버리지: {trade_record['leverage']:.1f}x")
    print(f"📈 손익(수수료 전): {trade_record['pnl']:,.0f}원")
    print(f"💸 순손익(수수료 후): {trade_record['net_pnl']:,.0f}원")
    print(f"🏁 청산 사유: {trade_record['reason']}")
    
    # 수익률 계산
    return_pct = (trade_record['net_pnl'] / trade_record['size']) * 100
    print(f"📊 수익률: {return_pct:.2f}%")
    
    # 성과 판정
    if trade_record['net_pnl'] > 0:
        print(f"✅ 결과: 이익 거래")
    else:
        print(f"❌ 결과: 손실 거래")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    print("🚀 트리플 콤보 전략 시스템 로드 완료!")
    print("   📈 추세 순응형 R/R 극대화 전략")
    print("   🔄 CVD 기반 스캘핑 전략") 
    print("   💥 변동성 돌파 전략")
    print("   🎯 통합 전략 매니저") 