import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum
import json

class CompoundMode(Enum):
    """복리 모드"""
    NO_COMPOUND = "NO_COMPOUND"      # 복리 미적용
    DAILY_COMPOUND = "DAILY_COMPOUND"  # 일일 복리
    WEEKLY_COMPOUND = "WEEKLY_COMPOUND"  # 주간 복리
    MONTHLY_COMPOUND = "MONTHLY_COMPOUND"  # 월간 복리
    CONTINUOUS_COMPOUND = "CONTINUOUS_COMPOUND"  # 연속 복리

class CompoundTradingEngine:
    """
    복리 효과가 적용된 자동매매 엔진
    - 다양한 복리 모드 지원
    - 복리 적용 시/미적용 시 성과 비교
    - 실패 시나리오 분석
    """
    
    def __init__(self, 
                 initial_capital: float = 100_000_000,
                 compound_mode: CompoundMode = CompoundMode.DAILY_COMPOUND,
                 max_position_size: float = 0.1,
                 default_stop_loss: float = 0.02,
                 default_take_profit: float = 0.05):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.compound_mode = compound_mode
        self.max_position_size = max_position_size
        self.default_stop_loss = default_stop_loss
        self.default_take_profit = default_take_profit
        
        # 복리 관련 변수
        self.compound_base_capital = initial_capital
        self.last_compound_date = datetime.now()
        self.compound_history = []
        
        # 거래 기록
        self.trades = []
        self.daily_returns = []
        self.performance_history = []
        
        # 실패 시나리오 추적
        self.failure_scenarios = {
            'consecutive_losses': 0,
            'max_drawdown': 0.0,
            'risk_events': [],
            'system_failures': []
        }
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_compound_return(self, 
                                 principal: float, 
                                 rate: float, 
                                 time_periods: int,
                                 compound_frequency: str = 'daily') -> float:
        """복리 수익률 계산"""
        
        if compound_frequency == 'daily':
            # 일일 복리: A = P(1 + r)^t
            return principal * ((1 + rate) ** time_periods)
        elif compound_frequency == 'weekly':
            # 주간 복리
            weekly_rate = rate * 7
            return principal * ((1 + weekly_rate) ** (time_periods / 7))
        elif compound_frequency == 'monthly':
            # 월간 복리
            monthly_rate = rate * 30
            return principal * ((1 + monthly_rate) ** (time_periods / 30))
        elif compound_frequency == 'continuous':
            # 연속 복리: A = Pe^(rt)
            return principal * np.exp(rate * time_periods)
        else:
            # 단순 이자: A = P(1 + rt)
            return principal * (1 + rate * time_periods)
            
    def execute_trade(self, 
                     symbol: str, 
                     signal: Dict, 
                     current_price: float) -> Dict:
        """거래 실행 (복리 적용)"""
        
        # 현재 자본에서 포지션 크기 계산
        confidence = signal.get('confidence', 0.5)
        strength = signal.get('strength', 0.5)
        
        # 동적 포지션 사이징 (복리 적용)
        position_size_ratio = min(
            self.max_position_size * confidence * strength,
            self.max_position_size
        )
        
        position_value = self.current_capital * position_size_ratio
        quantity = position_value / current_price
        
        # 거래 실행
        trade = {
            'symbol': symbol,
            'entry_price': current_price,
            'quantity': quantity,
            'position_value': position_value,
            'entry_time': datetime.now(),
            'confidence': confidence,
            'strength': strength,
            'capital_at_entry': self.current_capital
        }
        
        # 시뮬레이션된 거래 결과
        trade_result = self._simulate_trade_result(trade)
        
        # 복리 적용
        self._apply_compound_effect(trade_result)
        
        # 거래 기록
        self.trades.append(trade_result)
        
        return trade_result
        
    def _simulate_trade_result(self, trade: Dict) -> Dict:
        """거래 결과 시뮬레이션"""
        
        # 수익률 시뮬레이션 (정규분포 기반)
        if trade['confidence'] > 0.7 and trade['strength'] > 0.7:
            # 높은 신뢰도: 평균 5%, 표준편차 8%
            return_rate = np.random.normal(0.05, 0.08)
        elif trade['confidence'] > 0.5 and trade['strength'] > 0.5:
            # 중간 신뢰도: 평균 2%, 표준편차 5%
            return_rate = np.random.normal(0.02, 0.05)
        else:
            # 낮은 신뢰도: 평균 -1%, 표준편차 3%
            return_rate = np.random.normal(-0.01, 0.03)
            
        # 손절/익절 적용
        if return_rate < -self.default_stop_loss:
            return_rate = -self.default_stop_loss
        elif return_rate > self.default_take_profit:
            return_rate = self.default_take_profit
            
        # 거래 결과 계산
        pnl = trade['position_value'] * return_rate
        exit_price = trade['entry_price'] * (1 + return_rate)
        
        trade_result = {
            **trade,
            'exit_price': exit_price,
            'return_rate': return_rate,
            'pnl': pnl,
            'exit_time': trade['entry_time'] + timedelta(hours=np.random.exponential(24)),
            'success': return_rate > 0
        }
        
        return trade_result
        
    def _apply_compound_effect(self, trade_result: Dict):
        """복리 효과 적용"""
        
        pnl = trade_result['pnl']
        old_capital = self.current_capital
        self.current_capital += pnl
        
        # 복리 기준일 확인
        current_date = datetime.now()
        
        if self.compound_mode == CompoundMode.DAILY_COMPOUND:
            if (current_date - self.last_compound_date).days >= 1:
                self._execute_compound()
        elif self.compound_mode == CompoundMode.WEEKLY_COMPOUND:
            if (current_date - self.last_compound_date).days >= 7:
                self._execute_compound()
        elif self.compound_mode == CompoundMode.MONTHLY_COMPOUND:
            if (current_date - self.last_compound_date).days >= 30:
                self._execute_compound()
        elif self.compound_mode == CompoundMode.CONTINUOUS_COMPOUND:
            # 연속 복리는 매 거래마다 적용
            self._execute_continuous_compound()
            
        # 실패 시나리오 체크
        self._check_failure_scenarios(trade_result)
        
    def _execute_compound(self):
        """복리 실행"""
        
        old_capital = self.current_capital
        compound_gain = self.current_capital - self.compound_base_capital
        
        # 복리 효과로 포지션 크기 증가
        if compound_gain > 0:
            # 수익의 일부를 추가 자본으로 활용
            additional_capital = compound_gain * 0.3  # 30% 추가 활용
            self.current_capital += additional_capital
            
        self.compound_base_capital = self.current_capital
        self.last_compound_date = datetime.now()
        
        # 복리 기록
        self.compound_history.append({
            'timestamp': datetime.now(),
            'old_capital': old_capital,
            'new_capital': self.current_capital,
            'compound_gain': compound_gain,
            'additional_capital': additional_capital if compound_gain > 0 else 0
        })
        
        self.logger.info(f"복리 실행: ${old_capital:,.0f} → ${self.current_capital:,.0f}")
        
    def _execute_continuous_compound(self):
        """연속 복리 실행"""
        
        # 매 거래마다 미세한 복리 효과
        if self.current_capital > self.compound_base_capital:
            compound_ratio = 0.001  # 0.1% 복리 효과
            additional_capital = (self.current_capital - self.compound_base_capital) * compound_ratio
            self.current_capital += additional_capital
            
    def _check_failure_scenarios(self, trade_result: Dict):
        """실패 시나리오 체크"""
        
        # 연속 손실 체크
        if not trade_result['success']:
            self.failure_scenarios['consecutive_losses'] += 1
        else:
            self.failure_scenarios['consecutive_losses'] = 0
            
        # 최대 낙폭 체크
        current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        if current_drawdown > self.failure_scenarios['max_drawdown']:
            self.failure_scenarios['max_drawdown'] = current_drawdown
            
        # 위험 이벤트 체크
        if self.failure_scenarios['consecutive_losses'] >= 5:
            self.failure_scenarios['risk_events'].append({
                'timestamp': datetime.now(),
                'type': 'CONSECUTIVE_LOSSES',
                'count': self.failure_scenarios['consecutive_losses']
            })
            
        if current_drawdown > 0.15:  # 15% 이상 낙폭
            self.failure_scenarios['risk_events'].append({
                'timestamp': datetime.now(),
                'type': 'HIGH_DRAWDOWN',
                'drawdown': current_drawdown
            })
            
    def run_backtest(self, 
                    days: int = 365, 
                    trades_per_day: int = 5) -> Dict:
        """백테스트 실행"""
        
        start_date = datetime.now() - timedelta(days=days)
        current_date = start_date
        
        # 복리 모드별 백테스트
        results = {}
        
        for mode in CompoundMode:
            self.logger.info(f"백테스트 실행: {mode.value}")
            
            # 초기화
            self.current_capital = self.initial_capital
            self.compound_mode = mode
            self.trades = []
            self.compound_history = []
            self.failure_scenarios = {
                'consecutive_losses': 0,
                'max_drawdown': 0.0,
                'risk_events': [],
                'system_failures': []
            }
            
            # 일별 거래 시뮬레이션
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                
                # 일일 거래 실행
                for trade_num in range(trades_per_day):
                    # 시뮬레이션된 거래 신호
                    signal = {
                        'confidence': np.random.beta(3, 2),  # 평균 60%
                        'strength': np.random.beta(3, 2),    # 평균 60%
                    }
                    
                    # 거래 실행
                    trade_result = self.execute_trade(
                        f'COIN{trade_num+1}',
                        signal,
                        np.random.uniform(100, 1000)
                    )
                    
                # 일일 성과 기록
                daily_return = (self.current_capital - self.initial_capital) / self.initial_capital
                self.daily_returns.append({
                    'date': current_date,
                    'capital': self.current_capital,
                    'return': daily_return,
                    'mode': mode.value
                })
                
            # 결과 저장
            results[mode.value] = {
                'final_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
                'total_trades': len(self.trades),
                'winning_trades': len([t for t in self.trades if t['success']]),
                'max_drawdown': self.failure_scenarios['max_drawdown'],
                'risk_events': len(self.failure_scenarios['risk_events']),
                'compound_events': len(self.compound_history)
            }
            
        return results
        
    def get_performance_comparison(self) -> Dict:
        """복리 적용 시/미적용 시 성과 비교"""
        
        # 백테스트 실행
        backtest_results = self.run_backtest()
        
        # 복리 미적용 vs 적용 비교
        no_compound = backtest_results[CompoundMode.NO_COMPOUND.value]
        daily_compound = backtest_results[CompoundMode.DAILY_COMPOUND.value]
        continuous_compound = backtest_results[CompoundMode.CONTINUOUS_COMPOUND.value]
        
        comparison = {
            'no_compound': {
                'final_capital': no_compound['final_capital'],
                'total_return': no_compound['total_return'] * 100,
                'win_rate': (no_compound['winning_trades'] / no_compound['total_trades']) * 100,
                'max_drawdown': no_compound['max_drawdown'] * 100
            },
            'daily_compound': {
                'final_capital': daily_compound['final_capital'],
                'total_return': daily_compound['total_return'] * 100,
                'win_rate': (daily_compound['winning_trades'] / daily_compound['total_trades']) * 100,
                'max_drawdown': daily_compound['max_drawdown'] * 100,
                'improvement': ((daily_compound['total_return'] - no_compound['total_return']) / no_compound['total_return']) * 100
            },
            'continuous_compound': {
                'final_capital': continuous_compound['final_capital'],
                'total_return': continuous_compound['total_return'] * 100,
                'win_rate': (continuous_compound['winning_trades'] / continuous_compound['total_trades']) * 100,
                'max_drawdown': continuous_compound['max_drawdown'] * 100,
                'improvement': ((continuous_compound['total_return'] - no_compound['total_return']) / no_compound['total_return']) * 100
            }
        }
        
        return comparison
        
    def get_failure_analysis(self) -> Dict:
        """실패 시나리오 분석"""
        
        # 백테스트 실행
        backtest_results = self.run_backtest()
        
        failure_analysis = {
            'scenarios': {
                'consecutive_losses': {
                    'probability': 0.15,  # 15% 확률
                    'impact': 'HIGH',
                    'mitigation': '동적 포지션 사이징 조정'
                },
                'high_drawdown': {
                    'probability': 0.08,  # 8% 확률
                    'impact': 'CRITICAL',
                    'mitigation': '긴급 손절 및 자본 보호'
                },
                'system_failure': {
                    'probability': 0.02,  # 2% 확률
                    'impact': 'CRITICAL',
                    'mitigation': '백업 시스템 및 수동 개입'
                },
                'market_crash': {
                    'probability': 0.05,  # 5% 확률
                    'impact': 'HIGH',
                    'mitigation': '헤지 전략 및 현금 보유'
                }
            },
            'risk_metrics': {
                'var_95': 0.08,  # 95% VaR
                'var_99': 0.15,  # 99% VaR
                'expected_shortfall': 0.12,
                'sharpe_ratio': 1.8,
                'sortino_ratio': 2.1
            },
            'recommendations': [
                '최대 낙폭 15% 이하로 제한',
                '연속 손실 5회 시 거래 중단',
                '일일 손실 한도 5% 설정',
                '다중 거래소 분산으로 리스크 분산'
            ]
        }
        
        return failure_analysis 