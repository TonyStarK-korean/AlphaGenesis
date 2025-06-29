import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import threading
import time
from enum import Enum

class ExchangePhase(Enum):
    """거래소 Phase"""
    PHASE1_ACCELERATION = "PHASE1_ACCELERATION"  # 가속화 모드
    PHASE2_DEFENSIVE = "PHASE2_DEFENSIVE"        # 방어 모드

class MultiExchangeAccelerator:
    """
    다중 거래소 분산 가속화 전략
    - Phase1 가속화: 고수익 알트코인 전략
    - Phase2 전환: 안정적 대형코인 전략
    - 자동 분산: 거래소 자동 분산 및 재분배
    """
    
    def __init__(self, 
                 initial_capital: float = 100_000_000,
                 max_exchanges: int = 8,
                 acceleration_threshold: float = 0.8):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_exchanges = max_exchanges
        self.acceleration_threshold = acceleration_threshold
        
        # 거래소 관리
        self.exchanges = {}
        self.exchange_phases = {}
        self.exchange_capitals = {}
        
        # 초기 거래소 설정
        self._initialize_exchanges()
        
        # 성과 추적
        self.performance_history = []
        self.distribution_history = []
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 모니터링 스레드
        self.monitoring_thread = None
        self.is_monitoring = False
        
    def _initialize_exchanges(self):
        """거래소 초기화"""
        
        # 첫 번째 거래소 설정
        self.exchanges['exchange1'] = {
            'name': 'exchange1',
            'capital': self.initial_capital,
            'phase': ExchangePhase.PHASE1_ACCELERATION,
            'created_at': datetime.now(),
            'performance': 0.0,
            'trades': 0,
            'win_rate': 0.0
        }
        
        self.exchange_phases['exchange1'] = ExchangePhase.PHASE1_ACCELERATION
        self.exchange_capitals['exchange1'] = self.initial_capital
        
    def start_acceleration_strategy(self):
        """가속화 전략 시작"""
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._acceleration_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("🚀 다중 거래소 가속화 전략 시작")
        
    def stop_acceleration_strategy(self):
        """가속화 전략 중지"""
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        self.logger.info("🛑 다중 거래소 가속화 전략 중지")
        
    def _acceleration_loop(self):
        """가속화 루프"""
        
        while self.is_monitoring:
            try:
                # 각 거래소 성과 평가
                self._evaluate_exchange_performance()
                
                # Phase 전환 필요성 확인
                self._check_phase_transitions()
                
                # 거래소 분산 필요성 확인
                self._check_distribution_needs()
                
                # 성과 업데이트
                self._update_performance_metrics()
                
                # 10분 대기
                time.sleep(600)
                
            except Exception as e:
                self.logger.error(f"가속화 루프 오류: {str(e)}")
                time.sleep(60)
                
    def _evaluate_exchange_performance(self):
        """거래소 성과 평가"""
        
        for exchange_name, exchange_data in self.exchanges.items():
            # 시뮬레이션된 성과 계산
            if exchange_data['phase'] == ExchangePhase.PHASE1_ACCELERATION:
                # Phase 1: 높은 변동성, 높은 수익률
                daily_return = np.random.normal(0.05, 0.15)  # 평균 5%, 표준편차 15%
                win_rate = np.random.beta(6, 4)  # 평균 60% 승률
            else:
                # Phase 2: 낮은 변동성, 안정적 수익률
                daily_return = np.random.normal(0.02, 0.08)  # 평균 2%, 표준편차 8%
                win_rate = np.random.beta(7, 3)  # 평균 70% 승률
                
            # 성과 업데이트
            exchange_data['performance'] += daily_return
            exchange_data['win_rate'] = win_rate
            exchange_data['trades'] += np.random.poisson(5)  # 하루 평균 5거래
            
            # 자본 업데이트
            new_capital = exchange_data['capital'] * (1 + daily_return)
            exchange_data['capital'] = new_capital
            self.exchange_capitals[exchange_name] = new_capital
            
    def _check_phase_transitions(self):
        """Phase 전환 확인"""
        
        for exchange_name, exchange_data in self.exchanges.items():
            current_phase = exchange_data['phase']
            
            if current_phase == ExchangePhase.PHASE1_ACCELERATION:
                # Phase 1에서 Phase 2로 전환 조건
                if self._should_transition_to_phase2(exchange_data):
                    self._transition_to_phase2(exchange_name)
                    
    def _should_transition_to_phase2(self, exchange_data: Dict) -> bool:
        """Phase 2 전환 조건 확인"""
        
        # 성과가 임계값을 넘으면 Phase 2로 전환
        if exchange_data['performance'] > self.acceleration_threshold:
            return True
            
        # 자본이 특정 크기를 넘으면 Phase 2로 전환
        if exchange_data['capital'] > self.initial_capital * 2:  # 2배 이상
            return True
            
        # 승률이 낮으면 Phase 2로 전환 (리스크 관리)
        if exchange_data['win_rate'] < 0.4:  # 40% 미만
            return True
            
        return False
        
    def _transition_to_phase2(self, exchange_name: str):
        """Phase 2로 전환"""
        
        exchange_data = self.exchanges[exchange_name]
        old_phase = exchange_data['phase']
        
        exchange_data['phase'] = ExchangePhase.PHASE2_DEFENSIVE
        self.exchange_phases[exchange_name] = ExchangePhase.PHASE2_DEFENSIVE
        
        self.logger.info(f"🔄 {exchange_name}: Phase 1 → Phase 2 전환")
        self.logger.info(f"   성과: {exchange_data['performance']:.2%}")
        self.logger.info(f"   자본: ${exchange_data['capital']:,.0f}")
        
        # Phase 전환 기록
        self.distribution_history.append({
            'timestamp': datetime.now(),
            'exchange': exchange_name,
            'old_phase': old_phase.value,
            'new_phase': ExchangePhase.PHASE2_DEFENSIVE.value,
            'performance': exchange_data['performance'],
            'capital': exchange_data['capital']
        })
        
    def _check_distribution_needs(self):
        """분산 필요성 확인"""
        
        # Phase 1 거래소들 확인
        phase1_exchanges = [
            name for name, data in self.exchanges.items()
            if data['phase'] == ExchangePhase.PHASE1_ACCELERATION
        ]
        
        # Phase 2 거래소들 확인
        phase2_exchanges = [
            name for name, data in self.exchanges.items()
            if data['phase'] == ExchangePhase.PHASE2_DEFENSIVE
        ]
        
        # 새로운 거래소 추가 조건
        if len(self.exchanges) < self.max_exchanges:
            # Phase 1 거래소가 성과를 내면 새로운 거래소 추가
            for exchange_name in phase1_exchanges:
                exchange_data = self.exchanges[exchange_name]
                
                if (exchange_data['performance'] > 0.3 and  # 30% 이상 성과
                    exchange_data['capital'] > self.initial_capital * 1.5):  # 1.5배 이상 자본
                    
                    self._add_new_exchange(exchange_name)
                    break
                    
    def _add_new_exchange(self, source_exchange: str):
        """새로운 거래소 추가"""
        
        if len(self.exchanges) >= self.max_exchanges:
            return
            
        # 새로운 거래소 이름 생성
        new_exchange_name = f'exchange{len(self.exchanges) + 1}'
        
        # 소스 거래소에서 자본 분할
        source_capital = self.exchanges[source_exchange]['capital']
        split_capital = source_capital * 0.5  # 50% 분할
        
        # 소스 거래소 자본 감소
        self.exchanges[source_exchange]['capital'] = source_capital - split_capital
        self.exchange_capitals[source_exchange] = source_capital - split_capital
        
        # 새로운 거래소 생성
        self.exchanges[new_exchange_name] = {
            'name': new_exchange_name,
            'capital': split_capital,
            'phase': ExchangePhase.PHASE1_ACCELERATION,
            'created_at': datetime.now(),
            'performance': 0.0,
            'trades': 0,
            'win_rate': 0.0
        }
        
        self.exchange_phases[new_exchange_name] = ExchangePhase.PHASE1_ACCELERATION
        self.exchange_capitals[new_exchange_name] = split_capital
        
        self.logger.info(f"📊 새로운 거래소 추가: {new_exchange_name}")
        self.logger.info(f"   소스: {source_exchange} (${split_capital:,.0f} 분할)")
        self.logger.info(f"   총 거래소 수: {len(self.exchanges)}")
        
        # 분산 기록
        self.distribution_history.append({
            'timestamp': datetime.now(),
            'action': 'ADD_EXCHANGE',
            'new_exchange': new_exchange_name,
            'source_exchange': source_exchange,
            'split_capital': split_capital,
            'total_exchanges': len(self.exchanges)
        })
        
    def _update_performance_metrics(self):
        """성과 지표 업데이트"""
        
        total_capital = sum(self.exchange_capitals.values())
        total_performance = (total_capital / self.initial_capital - 1) * 100
        
        # Phase별 성과 계산
        phase1_capital = sum([
            capital for name, capital in self.exchange_capitals.items()
            if self.exchange_phases[name] == ExchangePhase.PHASE1_ACCELERATION
        ])
        
        phase2_capital = total_capital - phase1_capital
        
        performance_metrics = {
            'timestamp': datetime.now(),
            'total_capital': total_capital,
            'total_performance': total_performance,
            'phase1_capital': phase1_capital,
            'phase2_capital': phase2_capital,
            'phase1_ratio': phase1_capital / total_capital if total_capital > 0 else 0,
            'phase2_ratio': phase2_capital / total_capital if total_capital > 0 else 0,
            'total_exchanges': len(self.exchanges),
            'phase1_exchanges': len([p for p in self.exchange_phases.values() if p == ExchangePhase.PHASE1_ACCELERATION]),
            'phase2_exchanges': len([p for p in self.exchange_phases.values() if p == ExchangePhase.PHASE2_DEFENSIVE])
        }
        
        self.performance_history.append(performance_metrics)
        
        # 현재 자본 업데이트
        self.current_capital = total_capital
        
    def get_strategy_status(self) -> Dict:
        """전략 상태 반환"""
        
        total_capital = sum(self.exchange_capitals.values())
        
        return {
            'total_capital': total_capital,
            'total_performance': (total_capital / self.initial_capital - 1) * 100,
            'exchanges': self.exchanges,
            'exchange_phases': {k: v.value for k, v in self.exchange_phases.items()},
            'exchange_capitals': self.exchange_capitals,
            'phase1_count': len([p for p in self.exchange_phases.values() if p == ExchangePhase.PHASE1_ACCELERATION]),
            'phase2_count': len([p for p in self.exchange_phases.values() if p == ExchangePhase.PHASE2_DEFENSIVE]),
            'max_exchanges': self.max_exchanges
        }
        
    def get_performance_summary(self) -> Dict:
        """성과 요약 반환"""
        
        if not self.performance_history:
            return {'error': '성과 데이터가 없습니다.'}
            
        recent_performance = self.performance_history[-30:]  # 최근 30일
        
        if not recent_performance:
            return {'error': '최근 성과 데이터가 없습니다.'}
            
        # 성과 통계
        performances = [p['total_performance'] for p in recent_performance]
        
        return {
            'current_performance': performances[-1],
            'avg_performance': np.mean(performances),
            'max_performance': np.max(performances),
            'min_performance': np.min(performances),
            'performance_volatility': np.std(performances),
            'total_exchanges': len(self.exchanges),
            'phase1_exchanges': len([p for p in self.exchange_phases.values() if p == ExchangePhase.PHASE1_ACCELERATION]),
            'phase2_exchanges': len([p for p in self.exchange_phases.values() if p == ExchangePhase.PHASE2_DEFENSIVE]),
            'recent_distributions': self.distribution_history[-10:]  # 최근 10개 분산 기록
        }
        
    def simulate_strategy(self, days: int = 365) -> Dict:
        """전략 시뮬레이션"""
        
        # 초기 상태 백업
        original_exchanges = self.exchanges.copy()
        original_capitals = self.exchange_capitals.copy()
        original_phases = self.exchange_phases.copy()
        
        # 시뮬레이션 실행
        simulation_results = []
        
        for day in range(days):
            # 일일 성과 계산
            self._evaluate_exchange_performance()
            
            # Phase 전환 확인
            self._check_phase_transitions()
            
            # 분산 확인
            self._check_distribution_needs()
            
            # 성과 업데이트
            self._update_performance_metrics()
            
            # 결과 기록
            if day % 30 == 0:  # 30일마다 기록
                status = self.get_strategy_status()
                simulation_results.append({
                    'day': day,
                    'total_capital': status['total_capital'],
                    'total_performance': status['total_performance'],
                    'exchanges': len(self.exchanges),
                    'phase1_count': status['phase1_count'],
                    'phase2_count': status['phase2_count']
                })
                
        # 원래 상태 복원
        self.exchanges = original_exchanges
        self.exchange_capitals = original_capitals
        self.exchange_phases = original_phases
        
        return {
            'simulation_days': days,
            'final_capital': simulation_results[-1]['total_capital'] if simulation_results else self.initial_capital,
            'final_performance': simulation_results[-1]['total_performance'] if simulation_results else 0,
            'max_exchanges_reached': max([r['exchanges'] for r in simulation_results]) if simulation_results else 1,
            'daily_results': simulation_results
        } 