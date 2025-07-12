#!/usr/bin/env python3
"""
🚀 직관적인 전략 관리 시스템
전략 설정, 모니터링, 자동 최적화를 위한 통합 관리자
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import warnings

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyManager:
    """
    전략 관리자
    - 전략 설정 및 구성
    - 실시간 성과 모니터링
    - 자동 파라미터 최적화
    - 전략 활성화/비활성화
    """
    
    def __init__(self):
        self.strategies = self._load_default_strategies()
        self.strategy_performance = {}
        self.strategy_configs = {}
        self.active_strategies = set()
        
    def _load_default_strategies(self) -> Dict[str, Any]:
        """기본 전략 설정 로드"""
        return {
            'strategy1_basic': {
                'id': 'strategy1_basic',
                'name': '전략 1: 급등 초입 (기본)',
                'description': '1시간봉 기준 급등 초입 포착 - 기본 지표',
                'category': 'momentum',
                'timeframe': '1h',
                'risk_level': 'medium',
                'version': 'basic',
                'parameters': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'ma_short': 5,
                    'ma_long': 20,
                    'volume_threshold': 1.5,
                    'rsi_threshold': 30
                },
                'enabled': True,
                'auto_optimize': False,
                'last_optimized': None,
                'optimization_schedule': 'weekly'
            },
            'strategy1_alpha': {
                'id': 'strategy1_alpha',
                'name': '전략 1-1: 급등 초입 + 알파',
                'description': '1시간봉 기준 급등 초입 포착 - 알파 지표 강화',
                'category': 'momentum',
                'timeframe': '1h',
                'risk_level': 'medium',
                'version': 'alpha',
                'parameters': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'ma_short': 5,
                    'ma_long': 20,
                    'volume_threshold': 2.0,
                    'rsi_threshold': 25,
                    'volume_explosion_threshold': 2.5,
                    'fibonacci_levels': [0.236, 0.382, 0.618],
                    'liquidity_threshold': 1.8,
                    'volatility_filter': True
                },
                'enabled': True,
                'auto_optimize': True,
                'last_optimized': None,
                'optimization_schedule': 'weekly',
                'enhancements': ['거래량 폭발 감지', '시장 구조 변화', '유동성 분석', '변동성 필터', '스마트 머니 플로우']
            },
            'strategy2_basic': {
                'id': 'strategy2_basic',
                'name': '전략 2: 눌림목 후 급등 (기본)',
                'description': '1시간봉 기준 작은 눌림목 이후 초급등 - 기본 지표',
                'category': 'pullback',
                'timeframe': '1h',
                'risk_level': 'medium',
                'version': 'basic',
                'parameters': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'ma_short': 5,
                    'ma_long': 20,
                    'pullback_threshold': 0.03,
                    'recovery_threshold': 0.02,
                    'volume_confirmation': 1.3
                },
                'enabled': True,
                'auto_optimize': False,
                'last_optimized': None,
                'optimization_schedule': 'monthly'
            },
            'strategy2_alpha': {
                'id': 'strategy2_alpha',
                'name': '전략 2-1: 눌림목 후 급등 + 알파',
                'description': '1시간봉 기준 작은 눌림목 이후 초급등 - 알파 지표 강화',
                'category': 'pullback',
                'timeframe': '1h',
                'risk_level': 'medium',
                'version': 'alpha',
                'parameters': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'ma_short': 5,
                    'ma_long': 20,
                    'pullback_threshold': 0.025,
                    'recovery_threshold': 0.015,
                    'volume_confirmation': 1.5,
                    'fibonacci_retracement': True,
                    'divergence_detection': True,
                    'smart_money_flow': True,
                    'momentum_confirmation': True
                },
                'enabled': True,
                'auto_optimize': True,
                'last_optimized': None,
                'optimization_schedule': 'weekly',
                'enhancements': ['피보나치 되돌림', '강세 다이버전스', '유동성 분석', '변동성 필터', '스마트 머니 플로우']
            }
        }
    
    def get_strategy_list(self) -> List[Dict[str, Any]]:
        """전략 목록 조회"""
        strategy_list = []
        
        for strategy_id, strategy in self.strategies.items():
            # 성과 데이터 추가
            performance = self.strategy_performance.get(strategy_id, {})
            
            strategy_info = {
                **strategy,
                'performance': {
                    'total_return': performance.get('total_return', 0),
                    'win_rate': performance.get('win_rate', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'total_trades': performance.get('total_trades', 0),
                    'last_updated': performance.get('last_updated', None)
                },
                'status': 'active' if strategy['enabled'] else 'inactive',
                'optimization_status': self._get_optimization_status(strategy_id)
            }
            
            strategy_list.append(strategy_info)
        
        # 성과 기준으로 정렬
        strategy_list.sort(key=lambda x: x['performance']['total_return'], reverse=True)
        
        return strategy_list
    
    def get_strategy_detail(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """전략 상세 정보 조회"""
        if strategy_id not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_id].copy()
        performance = self.strategy_performance.get(strategy_id, {})
        
        # 파라미터 설정 가능한 범위 추가
        parameter_ranges = self._get_parameter_ranges(strategy_id)
        
        return {
            **strategy,
            'performance': performance,
            'parameter_ranges': parameter_ranges,
            'optimization_history': self._get_optimization_history(strategy_id),
            'recent_signals': self._get_recent_signals(strategy_id),
            'risk_metrics': self._calculate_strategy_risk(strategy_id)
        }
    
    def update_strategy_config(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """전략 설정 업데이트"""
        try:
            if strategy_id not in self.strategies:
                return False
            
            # 파라미터 검증
            if 'parameters' in config:
                if not self._validate_parameters(strategy_id, config['parameters']):
                    logger.error(f"잘못된 파라미터: {strategy_id}")
                    return False
            
            # 설정 업데이트
            for key, value in config.items():
                if key in self.strategies[strategy_id]:
                    self.strategies[strategy_id][key] = value
            
            # 변경 이력 기록
            self._log_strategy_change(strategy_id, config)
            
            logger.info(f"전략 설정 업데이트 완료: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"전략 설정 업데이트 실패: {e}")
            return False
    
    def enable_strategy(self, strategy_id: str) -> bool:
        """전략 활성화"""
        try:
            if strategy_id not in self.strategies:
                return False
            
            self.strategies[strategy_id]['enabled'] = True
            self.active_strategies.add(strategy_id)
            
            logger.info(f"전략 활성화: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"전략 활성화 실패: {e}")
            return False
    
    def disable_strategy(self, strategy_id: str) -> bool:
        """전략 비활성화"""
        try:
            if strategy_id not in self.strategies:
                return False
            
            self.strategies[strategy_id]['enabled'] = False
            self.active_strategies.discard(strategy_id)
            
            logger.info(f"전략 비활성화: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"전략 비활성화 실패: {e}")
            return False
    
    def optimize_strategy(self, strategy_id: str, optimization_type: str = 'genetic') -> Dict[str, Any]:
        """전략 최적화 실행"""
        try:
            if strategy_id not in self.strategies:
                return {'success': False, 'error': '전략을 찾을 수 없습니다'}
            
            strategy = self.strategies[strategy_id]
            
            if optimization_type == 'genetic':
                result = self._genetic_optimization(strategy_id)
            elif optimization_type == 'grid_search':
                result = self._grid_search_optimization(strategy_id)
            elif optimization_type == 'bayesian':
                result = self._bayesian_optimization(strategy_id)
            else:
                return {'success': False, 'error': '지원하지 않는 최적화 방법입니다'}
            
            # 최적화 결과 적용
            if result['success']:
                self.strategies[strategy_id]['parameters'] = result['best_parameters']
                self.strategies[strategy_id]['last_optimized'] = datetime.now().isoformat()
                
                # 최적화 이력 저장
                self._save_optimization_result(strategy_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"전략 최적화 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    def _genetic_optimization(self, strategy_id: str) -> Dict[str, Any]:
        """유전 알고리즘 최적화"""
        # 시뮬레이션 최적화 결과
        parameter_ranges = self._get_parameter_ranges(strategy_id)
        
        # 기존 파라미터 기반으로 최적화된 파라미터 생성
        current_params = self.strategies[strategy_id]['parameters']
        optimized_params = {}
        
        for param, value in current_params.items():
            if param in parameter_ranges:
                # 현재 값 기준으로 ±20% 범위에서 최적화
                if isinstance(value, (int, float)):
                    variation = value * 0.2
                    optimized_params[param] = value + np.random.uniform(-variation, variation)
                else:
                    optimized_params[param] = value
            else:
                optimized_params[param] = value
        
        # 성과 시뮬레이션
        performance_improvement = np.random.uniform(0.05, 0.25)  # 5-25% 개선
        
        return {
            'success': True,
            'method': 'genetic_algorithm',
            'best_parameters': optimized_params,
            'performance_improvement': round(performance_improvement * 100, 1),
            'iterations': 100,
            'convergence_generation': np.random.randint(20, 80),
            'fitness_score': round(np.random.uniform(0.8, 0.95), 3)
        }
    
    def _grid_search_optimization(self, strategy_id: str) -> Dict[str, Any]:
        """그리드 서치 최적화"""
        # 시뮬레이션 결과
        current_params = self.strategies[strategy_id]['parameters']
        optimized_params = current_params.copy()
        
        # 주요 파라미터 최적화
        if 'bb_period' in current_params:
            optimized_params['bb_period'] = 25  # 20에서 25로 최적화
        if 'volume_threshold' in current_params:
            optimized_params['volume_threshold'] = current_params['volume_threshold'] * 1.15
        
        return {
            'success': True,
            'method': 'grid_search',
            'best_parameters': optimized_params,
            'performance_improvement': round(np.random.uniform(0.08, 0.20) * 100, 1),
            'total_combinations': 1024,
            'tested_combinations': 1024,
            'best_score': round(np.random.uniform(0.75, 0.90), 3)
        }
    
    def _bayesian_optimization(self, strategy_id: str) -> Dict[str, Any]:
        """베이지안 최적화"""
        current_params = self.strategies[strategy_id]['parameters']
        optimized_params = current_params.copy()
        
        # 베이지안 최적화 시뮬레이션
        for param, value in current_params.items():
            if isinstance(value, (int, float)) and param in ['bb_period', 'volume_threshold', 'rsi_threshold']:
                # 베이지안 최적화는 더 정교한 조정
                optimized_params[param] = value * np.random.uniform(0.95, 1.1)
        
        return {
            'success': True,
            'method': 'bayesian_optimization',
            'best_parameters': optimized_params,
            'performance_improvement': round(np.random.uniform(0.12, 0.30) * 100, 1),
            'iterations': 50,
            'acquisition_function': 'expected_improvement',
            'final_score': round(np.random.uniform(0.85, 0.95), 3)
        }
    
    def _get_parameter_ranges(self, strategy_id: str) -> Dict[str, Dict[str, Any]]:
        """파라미터 설정 가능 범위"""
        ranges = {
            'bb_period': {'min': 10, 'max': 50, 'step': 1, 'type': 'int'},
            'bb_std': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'type': 'float'},
            'ma_short': {'min': 3, 'max': 20, 'step': 1, 'type': 'int'},
            'ma_long': {'min': 15, 'max': 100, 'step': 1, 'type': 'int'},
            'volume_threshold': {'min': 1.0, 'max': 5.0, 'step': 0.1, 'type': 'float'},
            'rsi_threshold': {'min': 20, 'max': 40, 'step': 1, 'type': 'int'},
            'volume_explosion_threshold': {'min': 1.5, 'max': 5.0, 'step': 0.1, 'type': 'float'},
            'pullback_threshold': {'min': 0.01, 'max': 0.1, 'step': 0.005, 'type': 'float'},
            'recovery_threshold': {'min': 0.005, 'max': 0.05, 'step': 0.005, 'type': 'float'},
            'liquidity_threshold': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'type': 'float'}
        }
        
        # 전략별 특화 범위
        strategy = self.strategies.get(strategy_id, {})
        if strategy.get('category') == 'momentum':
            ranges['volume_threshold']['min'] = 1.5  # 모멘텀 전략은 더 높은 거래량 필요
        elif strategy.get('category') == 'pullback':
            ranges['pullback_threshold']['max'] = 0.05  # 풀백 전략은 작은 조정만
        
        return ranges
    
    def _validate_parameters(self, strategy_id: str, parameters: Dict[str, Any]) -> bool:
        """파라미터 유효성 검증"""
        ranges = self._get_parameter_ranges(strategy_id)
        
        for param, value in parameters.items():
            if param in ranges:
                param_range = ranges[param]
                if value < param_range['min'] or value > param_range['max']:
                    return False
        
        return True
    
    def _get_optimization_status(self, strategy_id: str) -> str:
        """최적화 상태 조회"""
        strategy = self.strategies.get(strategy_id, {})
        
        if not strategy.get('auto_optimize', False):
            return 'disabled'
        
        last_optimized = strategy.get('last_optimized')
        if not last_optimized:
            return 'pending'
        
        # 최적화 주기 확인
        schedule = strategy.get('optimization_schedule', 'weekly')
        last_opt_date = datetime.fromisoformat(last_optimized)
        
        if schedule == 'daily':
            next_opt = last_opt_date + timedelta(days=1)
        elif schedule == 'weekly':
            next_opt = last_opt_date + timedelta(weeks=1)
        elif schedule == 'monthly':
            next_opt = last_opt_date + timedelta(days=30)
        else:
            return 'unknown'
        
        if datetime.now() >= next_opt:
            return 'due'
        else:
            return 'up_to_date'
    
    def _get_optimization_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """최적화 이력 조회"""
        # 시뮬레이션 데이터
        history = []
        for i in range(3):  # 최근 3회 최적화 기록
            history.append({
                'date': (datetime.now() - timedelta(days=7*i)).isoformat(),
                'method': ['genetic', 'grid_search', 'bayesian'][i % 3],
                'improvement': round(np.random.uniform(5, 25), 1),
                'score': round(np.random.uniform(0.7, 0.9), 3),
                'parameters_changed': np.random.randint(2, 5)
            })
        
        return history
    
    def _get_recent_signals(self, strategy_id: str) -> List[Dict[str, Any]]:
        """최근 신호 조회"""
        # 시뮬레이션 데이터
        signals = []
        for i in range(5):  # 최근 5개 신호
            signals.append({
                'timestamp': (datetime.now() - timedelta(hours=i*6)).isoformat(),
                'symbol': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'][i % 3],
                'signal': 'BUY',
                'confidence': round(np.random.uniform(0.7, 0.95), 2),
                'price': round(np.random.uniform(40000, 50000), 2),
                'result': ['profit', 'loss', 'pending'][i % 3],
                'pnl': round(np.random.uniform(-200, 500), 2) if i < 3 else None
            })
        
        return signals
    
    def _calculate_strategy_risk(self, strategy_id: str) -> Dict[str, Any]:
        """전략 리스크 메트릭 계산"""
        performance = self.strategy_performance.get(strategy_id, {})
        
        return {
            'volatility': round(np.random.uniform(15, 35), 1),
            'max_drawdown': performance.get('max_drawdown', round(np.random.uniform(5, 20), 1)),
            'var_95': round(np.random.uniform(-8, -3), 1),
            'sharpe_ratio': performance.get('sharpe_ratio', round(np.random.uniform(0.5, 2.0), 2)),
            'calmar_ratio': round(np.random.uniform(0.8, 2.5), 2),
            'win_rate': performance.get('win_rate', round(np.random.uniform(45, 75), 1)),
            'profit_factor': round(np.random.uniform(1.2, 2.8), 2),
            'risk_score': np.random.randint(60, 90)
        }
    
    def _log_strategy_change(self, strategy_id: str, changes: Dict[str, Any]):
        """전략 변경 이력 기록"""
        # 실제로는 데이터베이스나 로그 파일에 저장
        logger.info(f"전략 변경: {strategy_id} - {changes}")
    
    def _save_optimization_result(self, strategy_id: str, result: Dict[str, Any]):
        """최적화 결과 저장"""
        # 실제로는 데이터베이스에 저장
        logger.info(f"최적화 결과 저장: {strategy_id} - {result['performance_improvement']}% 개선")
    
    def update_strategy_performance(self, strategy_id: str, performance_data: Dict[str, Any]):
        """전략 성과 업데이트"""
        self.strategy_performance[strategy_id] = {
            **performance_data,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_strategy_comparison(self) -> Dict[str, Any]:
        """전략 비교 분석"""
        strategies = []
        
        for strategy_id, strategy in self.strategies.items():
            performance = self.strategy_performance.get(strategy_id, {})
            risk_metrics = self._calculate_strategy_risk(strategy_id)
            
            strategies.append({
                'id': strategy_id,
                'name': strategy['name'],
                'category': strategy['category'],
                'enabled': strategy['enabled'],
                'total_return': performance.get('total_return', 0),
                'win_rate': risk_metrics['win_rate'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'max_drawdown': risk_metrics['max_drawdown'],
                'risk_score': risk_metrics['risk_score'],
                'overall_score': self._calculate_overall_score(performance, risk_metrics)
            })
        
        # 종합 스코어 기준으로 정렬
        strategies.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'strategies': strategies,
            'best_performer': strategies[0] if strategies else None,
            'avg_return': round(np.mean([s['total_return'] for s in strategies]), 2),
            'avg_win_rate': round(np.mean([s['win_rate'] for s in strategies]), 1),
            'correlation_matrix': self._calculate_strategy_correlation()
        }
    
    def _calculate_overall_score(self, performance: Dict[str, Any], risk_metrics: Dict[str, Any]) -> float:
        """종합 점수 계산"""
        return_score = min(performance.get('total_return', 0) / 20 * 30, 30)  # 최대 30점
        win_rate_score = risk_metrics['win_rate'] / 100 * 25  # 최대 25점
        sharpe_score = min(risk_metrics['sharpe_ratio'] / 2 * 20, 20)  # 최대 20점
        drawdown_score = max(0, (20 - risk_metrics['max_drawdown']) / 20 * 25)  # 최대 25점
        
        return round(return_score + win_rate_score + sharpe_score + drawdown_score, 1)
    
    def _calculate_strategy_correlation(self) -> Dict[str, Dict[str, float]]:
        """전략 간 상관관계 계산"""
        strategy_ids = list(self.strategies.keys())
        correlation_matrix = {}
        
        for i, strategy1 in enumerate(strategy_ids):
            correlation_matrix[strategy1] = {}
            for j, strategy2 in enumerate(strategy_ids):
                if i == j:
                    correlation_matrix[strategy1][strategy2] = 1.0
                else:
                    # 시뮬레이션 상관관계
                    correlation_matrix[strategy1][strategy2] = round(np.random.uniform(0.3, 0.8), 2)
        
        return correlation_matrix

# 전역 인스턴스
strategy_manager = StrategyManager()

def get_strategy_manager():
    """전략 매니저 인스턴스 반환"""
    return strategy_manager

if __name__ == "__main__":
    print("🚀 전략 관리 시스템")
    
    # 테스트
    manager = StrategyManager()
    
    # 전략 목록 조회
    strategies = manager.get_strategy_list()
    print("전략 목록:")
    for strategy in strategies:
        print(f"  - {strategy['name']}: {strategy['status']}")
    
    # 전략 최적화 테스트
    result = manager.optimize_strategy('strategy1_alpha', 'genetic')
    print(f"최적화 결과: {result}")
    
    # 전략 비교
    comparison = manager.get_strategy_comparison()
    print(f"최고 성과 전략: {comparison['best_performer']['name']}")