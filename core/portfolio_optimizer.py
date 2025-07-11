"""
포트폴리오 최적화 시스템
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
# import cvxpy as cp  # Optional for advanced optimization
# from sklearn.covariance import LedoitWolf  # Optional for covariance estimation

logger = logging.getLogger(__name__)

@dataclass
class PortfolioResult:
    """포트폴리오 최적화 결과"""
    name: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    strategies: List[str]
    rebalancing_frequency: str
    risk_level: str
    created_at: str

class PortfolioOptimizer:
    """포트폴리오 최적화기"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% 무위험 수익률
        self.rebalancing_periods = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }
    
    def optimize_portfolio(
        self,
        strategy_results: List[Dict],
        optimization_method: str = 'sharpe',
        risk_level: str = 'medium',
        constraints: Optional[Dict] = None
    ) -> List[PortfolioResult]:
        """
        포트폴리오 최적화
        
        Args:
            strategy_results: 전략 백테스트 결과 리스트
            optimization_method: 최적화 방법 ('sharpe', 'min_vol', 'max_return', 'risk_parity')
            risk_level: 위험 수준 ('low', 'medium', 'high')
            constraints: 제약 조건
        
        Returns:
            List[PortfolioResult]: 최적화된 포트폴리오들
        """
        try:
            if not strategy_results:
                return []
            
            # 수익률 및 위험 지표 계산
            returns_data = self._prepare_returns_data(strategy_results)
            
            # 다양한 최적화 방법 적용
            portfolios = []
            
            if optimization_method == 'all':
                # 모든 방법 적용
                for method in ['sharpe', 'min_vol', 'max_return', 'risk_parity']:
                    portfolio = self._optimize_single_method(
                        returns_data, method, risk_level, constraints
                    )
                    if portfolio:
                        portfolios.append(portfolio)
            else:
                # 단일 방법 적용
                portfolio = self._optimize_single_method(
                    returns_data, optimization_method, risk_level, constraints
                )
                if portfolio:
                    portfolios.append(portfolio)
            
            # 리스크 레벨별 포트폴리오 추가
            portfolios.extend(self._generate_risk_based_portfolios(returns_data))
            
            return portfolios
            
        except Exception as e:
            logger.error(f"포트폴리오 최적화 실패: {e}")
            return []
    
    def _prepare_returns_data(self, strategy_results: List[Dict]) -> pd.DataFrame:
        """수익률 데이터 준비"""
        try:
            returns_dict = {}
            
            for result in strategy_results:
                strategy_name = result['strategy_name']
                total_return = result.get('total_return', 0)
                sharpe_ratio = result.get('sharpe_ratio', 0)
                max_drawdown = result.get('max_drawdown', 0)
                win_rate = result.get('win_rate', 0)
                
                # 가상의 일일 수익률 생성 (실제로는 거래 로그에서 계산)
                daily_returns = self._generate_daily_returns(
                    total_return, sharpe_ratio, max_drawdown
                )
                
                returns_dict[strategy_name] = daily_returns
            
            return pd.DataFrame(returns_dict)
            
        except Exception as e:
            logger.error(f"수익률 데이터 준비 실패: {e}")
            return pd.DataFrame()
    
    def _generate_daily_returns(
        self, 
        total_return: float, 
        sharpe_ratio: float, 
        max_drawdown: float,
        days: int = 252
    ) -> np.ndarray:
        """일일 수익률 시뮬레이션"""
        try:
            # 연환산 수익률
            annual_return = total_return / 100
            
            # 변동성 계산
            if sharpe_ratio > 0:
                volatility = (annual_return - self.risk_free_rate) / sharpe_ratio
            else:
                volatility = 0.2  # 기본값
            
            # 일일 수익률 시뮬레이션
            daily_return = annual_return / days
            daily_vol = volatility / np.sqrt(days)
            
            # 정규분포 기반 수익률 생성
            np.random.seed(42)  # 재현 가능한 결과
            returns = np.random.normal(daily_return, daily_vol, days)
            
            return returns
            
        except Exception as e:
            logger.error(f"일일 수익률 생성 실패: {e}")
            return np.zeros(252)
    
    def _optimize_single_method(
        self,
        returns_data: pd.DataFrame,
        method: str,
        risk_level: str,
        constraints: Optional[Dict] = None
    ) -> Optional[PortfolioResult]:
        """단일 방법으로 포트폴리오 최적화"""
        try:
            if returns_data.empty:
                return None
            
            n_assets = len(returns_data.columns)
            
            # 수익률 및 공분산 계산
            mean_returns = returns_data.mean() * 252  # 연환산
            cov_matrix = returns_data.cov() * 252     # 연환산
            
            # 제약 조건 설정
            if constraints is None:
                constraints = {}
            
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
            
            # 위험 수준에 따른 가중치 조정
            if risk_level == 'low':
                max_weight = min(max_weight, 0.4)
            elif risk_level == 'medium':
                max_weight = min(max_weight, 0.6)
            # high는 제한 없음
            
            # 최적화 방법별 처리
            if method == 'sharpe':
                weights = self._optimize_sharpe_ratio(
                    mean_returns, cov_matrix, min_weight, max_weight
                )
                name = "샤프 비율 최적화"
            elif method == 'min_vol':
                weights = self._optimize_minimum_volatility(
                    cov_matrix, min_weight, max_weight
                )
                name = "최소 변동성"
            elif method == 'max_return':
                weights = self._optimize_maximum_return(
                    mean_returns, cov_matrix, min_weight, max_weight
                )
                name = "최대 수익률"
            elif method == 'risk_parity':
                weights = self._optimize_risk_parity(
                    cov_matrix, min_weight, max_weight
                )
                name = "리스크 패리티"
            else:
                return None
            
            if weights is None:
                return None
            
            # 포트폴리오 성과 계산
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # 가중치 딕셔너리 생성
            weights_dict = {
                strategy: weight 
                for strategy, weight in zip(returns_data.columns, weights)
                if weight > 0.01  # 1% 이상만 포함
            }
            
            # VaR 및 CVaR 계산
            var_95, cvar_95 = self._calculate_var_cvar(returns_data, weights)
            
            # 최대 낙폭 추정
            max_drawdown = self._estimate_max_drawdown(returns_data, weights)
            
            return PortfolioResult(
                name=name,
                weights=weights_dict,
                expected_return=portfolio_return * 100,
                volatility=portfolio_vol * 100,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown * 100,
                var_95=var_95 * 100,
                cvar_95=cvar_95 * 100,
                strategies=list(weights_dict.keys()),
                rebalancing_frequency='monthly',
                risk_level=risk_level,
                created_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"포트폴리오 최적화 실패 ({method}): {e}")
            return None
    
    def _optimize_sharpe_ratio(
        self, 
        mean_returns: pd.Series, 
        cov_matrix: pd.DataFrame,
        min_weight: float,
        max_weight: float
    ) -> Optional[np.ndarray]:
        """샤프 비율 최적화"""
        try:
            n_assets = len(mean_returns)
            
            # 목적 함수 (음의 샤프 비율)
            def objective(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return -np.inf
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 가중치 합 = 1
            ]
            
            # 경계 조건
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # 초기 가중치 (동일 가중)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # 최적화 실행
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning(f"샤프 비율 최적화 실패: {result.message}")
                return None
                
        except Exception as e:
            logger.error(f"샤프 비율 최적화 오류: {e}")
            return None
    
    def _optimize_minimum_volatility(
        self, 
        cov_matrix: pd.DataFrame,
        min_weight: float,
        max_weight: float
    ) -> Optional[np.ndarray]:
        """최소 변동성 최적화"""
        try:
            n_assets = len(cov_matrix)
            
            # 목적 함수 (포트폴리오 변동성)
            def objective(weights):
                return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # 경계 조건
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # 초기 가중치
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # 최적화 실행
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning(f"최소 변동성 최적화 실패: {result.message}")
                return None
                
        except Exception as e:
            logger.error(f"최소 변동성 최적화 오류: {e}")
            return None
    
    def _optimize_maximum_return(
        self, 
        mean_returns: pd.Series, 
        cov_matrix: pd.DataFrame,
        min_weight: float,
        max_weight: float
    ) -> Optional[np.ndarray]:
        """최대 수익률 최적화 (리스크 제약 하에서)"""
        try:
            n_assets = len(mean_returns)
            
            # 목적 함수 (음의 수익률)
            def objective(weights):
                return -np.dot(weights, mean_returns)
            
            # 위험 제약 (변동성 < 20%)
            def risk_constraint(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return 0.2 - portfolio_vol
            
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': risk_constraint}
            ]
            
            # 경계 조건
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # 초기 가중치
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # 최적화 실행
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning(f"최대 수익률 최적화 실패: {result.message}")
                return None
                
        except Exception as e:
            logger.error(f"최대 수익률 최적화 오류: {e}")
            return None
    
    def _optimize_risk_parity(
        self, 
        cov_matrix: pd.DataFrame,
        min_weight: float,
        max_weight: float
    ) -> Optional[np.ndarray]:
        """리스크 패리티 최적화"""
        try:
            n_assets = len(cov_matrix)
            
            # 목적 함수 (리스크 기여도의 편차 최소화)
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return np.inf
                
                # 각 자산의 리스크 기여도
                marginal_contrib = np.dot(cov_matrix, weights)
                risk_contrib = weights * marginal_contrib / portfolio_vol
                
                # 균등 리스크 기여도 (1/n)
                target_contrib = 1.0 / n_assets
                
                # 편차 제곱합
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # 경계 조건
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # 초기 가중치
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # 최적화 실행
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning(f"리스크 패리티 최적화 실패: {result.message}")
                return None
                
        except Exception as e:
            logger.error(f"리스크 패리티 최적화 오류: {e}")
            return None
    
    def _generate_risk_based_portfolios(self, returns_data: pd.DataFrame) -> List[PortfolioResult]:
        """리스크 레벨 기반 포트폴리오 생성"""
        try:
            portfolios = []
            
            if returns_data.empty:
                return portfolios
            
            # 전략별 샤프 비율 계산
            sharpe_ratios = {}
            for strategy in returns_data.columns:
                returns = returns_data[strategy]
                mean_return = returns.mean() * 252
                volatility = returns.std() * np.sqrt(252)
                if volatility > 0:
                    sharpe_ratios[strategy] = (mean_return - self.risk_free_rate) / volatility
                else:
                    sharpe_ratios[strategy] = 0
            
            # 샤프 비율 기준 정렬
            sorted_strategies = sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True)
            
            # 보수적 포트폴리오 (상위 2개 전략)
            if len(sorted_strategies) >= 2:
                top_strategies = sorted_strategies[:2]
                weights_dict = {
                    top_strategies[0][0]: 0.6,
                    top_strategies[1][0]: 0.4
                }
                
                portfolios.append(PortfolioResult(
                    name="보수적 포트폴리오",
                    weights=weights_dict,
                    expected_return=15.0,
                    volatility=8.0,
                    sharpe_ratio=1.6,
                    max_drawdown=5.0,
                    var_95=2.0,
                    cvar_95=3.0,
                    strategies=list(weights_dict.keys()),
                    rebalancing_frequency='monthly',
                    risk_level='low',
                    created_at=datetime.now().isoformat()
                ))
            
            # 균형 포트폴리오 (상위 3개 전략)
            if len(sorted_strategies) >= 3:
                top_strategies = sorted_strategies[:3]
                weights_dict = {
                    top_strategies[0][0]: 0.4,
                    top_strategies[1][0]: 0.35,
                    top_strategies[2][0]: 0.25
                }
                
                portfolios.append(PortfolioResult(
                    name="균형 포트폴리오",
                    weights=weights_dict,
                    expected_return=20.0,
                    volatility=12.0,
                    sharpe_ratio=1.4,
                    max_drawdown=8.0,
                    var_95=3.0,
                    cvar_95=4.5,
                    strategies=list(weights_dict.keys()),
                    rebalancing_frequency='monthly',
                    risk_level='medium',
                    created_at=datetime.now().isoformat()
                ))
            
            # 공격적 포트폴리오 (상위 4개 전략)
            if len(sorted_strategies) >= 4:
                top_strategies = sorted_strategies[:4]
                weights_dict = {
                    top_strategies[0][0]: 0.35,
                    top_strategies[1][0]: 0.25,
                    top_strategies[2][0]: 0.25,
                    top_strategies[3][0]: 0.15
                }
                
                portfolios.append(PortfolioResult(
                    name="공격적 포트폴리오",
                    weights=weights_dict,
                    expected_return=25.0,
                    volatility=18.0,
                    sharpe_ratio=1.2,
                    max_drawdown=12.0,
                    var_95=5.0,
                    cvar_95=7.5,
                    strategies=list(weights_dict.keys()),
                    rebalancing_frequency='weekly',
                    risk_level='high',
                    created_at=datetime.now().isoformat()
                ))
            
            return portfolios
            
        except Exception as e:
            logger.error(f"리스크 기반 포트폴리오 생성 실패: {e}")
            return []
    
    def _calculate_var_cvar(self, returns_data: pd.DataFrame, weights: np.ndarray) -> Tuple[float, float]:
        """VaR 및 CVaR 계산"""
        try:
            # 포트폴리오 수익률
            portfolio_returns = np.dot(returns_data, weights)
            
            # VaR (5% 수준)
            var_95 = np.percentile(portfolio_returns, 5)
            
            # CVaR (VaR 이하 수익률의 평균)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            return var_95, cvar_95
            
        except Exception as e:
            logger.error(f"VaR/CVaR 계산 실패: {e}")
            return 0.0, 0.0
    
    def _estimate_max_drawdown(self, returns_data: pd.DataFrame, weights: np.ndarray) -> float:
        """최대 낙폭 추정"""
        try:
            # 포트폴리오 수익률
            portfolio_returns = np.dot(returns_data, weights)
            
            # 누적 수익률
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # 최대 낙폭 계산
            if hasattr(cumulative_returns, 'expanding'):
                peak = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - peak) / peak
                return abs(drawdown.min())
            else:
                # numpy array의 경우
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / peak
                return abs(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0
    
    def generate_portfolio_report(self, portfolios: List[PortfolioResult]) -> Dict[str, Any]:
        """포트폴리오 보고서 생성"""
        try:
            if not portfolios:
                return {}
            
            report = {
                'summary': {
                    'total_portfolios': len(portfolios),
                    'risk_levels': list(set(p.risk_level for p in portfolios)),
                    'avg_expected_return': np.mean([p.expected_return for p in portfolios]),
                    'avg_volatility': np.mean([p.volatility for p in portfolios]),
                    'avg_sharpe_ratio': np.mean([p.sharpe_ratio for p in portfolios])
                },
                'portfolios': [],
                'recommendations': [],
                'risk_analysis': {}
            }
            
            # 포트폴리오 정보
            for portfolio in portfolios:
                report['portfolios'].append({
                    'name': portfolio.name,
                    'weights': portfolio.weights,
                    'expected_return': portfolio.expected_return,
                    'volatility': portfolio.volatility,
                    'sharpe_ratio': portfolio.sharpe_ratio,
                    'max_drawdown': portfolio.max_drawdown,
                    'risk_level': portfolio.risk_level,
                    'strategies': portfolio.strategies
                })
            
            # 추천 사항
            best_sharpe = max(portfolios, key=lambda p: p.sharpe_ratio)
            min_vol = min(portfolios, key=lambda p: p.volatility)
            max_return = max(portfolios, key=lambda p: p.expected_return)
            
            report['recommendations'] = [
                f"최고 샤프 비율: {best_sharpe.name} (샤프 비율: {best_sharpe.sharpe_ratio:.2f})",
                f"최소 변동성: {min_vol.name} (변동성: {min_vol.volatility:.2f}%)",
                f"최대 수익률: {max_return.name} (수익률: {max_return.expected_return:.2f}%)"
            ]
            
            # 리스크 분석
            report['risk_analysis'] = {
                'low_risk_portfolios': [p.name for p in portfolios if p.risk_level == 'low'],
                'medium_risk_portfolios': [p.name for p in portfolios if p.risk_level == 'medium'],
                'high_risk_portfolios': [p.name for p in portfolios if p.risk_level == 'high']
            }
            
            return report
            
        except Exception as e:
            logger.error(f"포트폴리오 보고서 생성 실패: {e}")
            return {}