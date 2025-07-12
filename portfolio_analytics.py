#!/usr/bin/env python3
"""
🚀 고급 포트폴리오 분석 모듈
실전급 포트폴리오 분석과 리스크 관리
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPortfolioAnalytics:
    """
    고급 포트폴리오 분석기
    - 실시간 성과 분석
    - 리스크 메트릭 계산
    - 샤프 비율, 칼마 비율 등 고급 지표
    - 포트폴리오 최적화 제안
    """
    
    def __init__(self):
        self.portfolio_history = []
        self.trade_history = []
        self.benchmark_data = None
        
    def add_portfolio_snapshot(self, snapshot: Dict[str, Any]):
        """포트폴리오 스냅샷 추가"""
        snapshot['timestamp'] = datetime.now().isoformat()
        self.portfolio_history.append(snapshot)
        
        # 최근 1000개만 유지 (메모리 관리)
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def add_trade_record(self, trade: Dict[str, Any]):
        """거래 기록 추가"""
        trade['timestamp'] = datetime.now().isoformat()
        self.trade_history.append(trade)
        
        # 최근 5000개만 유지
        if len(self.trade_history) > 5000:
            self.trade_history = self.trade_history[-5000:]
    
    def calculate_performance_metrics(self, period_days: int = 30) -> Dict[str, Any]:
        """성과 지표 계산"""
        try:
            if len(self.portfolio_history) < 2:
                return self._empty_metrics()
            
            # 기간별 데이터 필터링
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_data = [
                snap for snap in self.portfolio_history 
                if datetime.fromisoformat(snap['timestamp']) >= cutoff_date
            ]
            
            if len(recent_data) < 2:
                return self._empty_metrics()
            
            # 포트폴리오 가치 시계열
            portfolio_values = [snap.get('total_value', 0) for snap in recent_data]
            timestamps = [datetime.fromisoformat(snap['timestamp']) for snap in recent_data]
            
            # 수익률 계산
            returns = self._calculate_returns(portfolio_values)
            
            # 기본 성과 지표
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
            annualized_return = self._annualize_return(total_return, len(returns))
            volatility = np.std(returns) * np.sqrt(252) * 100  # 연환산 변동성
            
            # 리스크 조정 수익률
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # 승률 및 손익비
            win_rate, profit_loss_ratio = self._calculate_win_loss_metrics()
            
            # VaR (Value at Risk)
            var_95 = self._calculate_var(returns, confidence=0.95)
            var_99 = self._calculate_var(returns, confidence=0.99)
            
            # 베타 계산 (시장 대비)
            beta = self._calculate_beta(returns)
            
            return {
                'period_days': period_days,
                'total_return': round(total_return, 2),
                'annualized_return': round(annualized_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'calmar_ratio': round(calmar_ratio, 3),
                'max_drawdown': round(max_drawdown, 2),
                'win_rate': round(win_rate, 1),
                'profit_loss_ratio': round(profit_loss_ratio, 2),
                'var_95': round(var_95, 2),
                'var_99': round(var_99, 2),
                'beta': round(beta, 3),
                'current_value': portfolio_values[-1],
                'peak_value': max(portfolio_values),
                'total_trades': len([t for t in self.trade_history 
                                   if datetime.fromisoformat(t['timestamp']) >= cutoff_date]),
                'avg_trade_size': self._calculate_avg_trade_size(cutoff_date),
                'largest_win': self._calculate_largest_win(cutoff_date),
                'largest_loss': self._calculate_largest_loss(cutoff_date)
            }
            
        except Exception as e:
            logger.error(f"성과 지표 계산 실패: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """빈 메트릭 반환"""
        return {
            'period_days': 0,
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_loss_ratio': 0,
            'var_95': 0,
            'var_99': 0,
            'beta': 0,
            'current_value': 0,
            'peak_value': 0,
            'total_trades': 0,
            'avg_trade_size': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
    
    def _calculate_returns(self, values: List[float]) -> np.ndarray:
        """수익률 계산"""
        if len(values) < 2:
            return np.array([])
        
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        
        return np.array(returns)
    
    def _annualize_return(self, total_return: float, periods: int) -> float:
        """연환산 수익률 계산"""
        if periods <= 0:
            return 0
        
        # 일 단위를 년 단위로 환산 (252 거래일 기준)
        periods_per_year = 252 / periods if periods < 252 else periods / 252
        return ((1 + total_return/100) ** periods_per_year - 1) * 100
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate/252  # 일일 무위험 수익률
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """칼마 비율 계산"""
        if len(returns) == 0:
            return 0
        
        annual_return = np.mean(returns) * 252 * 100
        max_dd = self._calculate_max_drawdown_from_returns(returns)
        
        return annual_return / abs(max_dd) if max_dd != 0 else 0
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """최대 낙폭 계산"""
        if len(values) < 2:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """수익률에서 최대 낙폭 계산"""
        if len(returns) == 0:
            return 0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown) * 100
    
    def _calculate_win_loss_metrics(self) -> tuple:
        """승률 및 손익비 계산"""
        if len(self.trade_history) == 0:
            return 0, 0
        
        profitable_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / len(self.trade_history) * 100
        
        if len(losing_trades) == 0:
            profit_loss_ratio = float('inf')
        else:
            avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        
        return win_rate, profit_loss_ratio
    
    def _calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Value at Risk 계산"""
        if len(returns) == 0:
            return 0
        
        return np.percentile(returns, (1 - confidence) * 100) * 100
    
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """베타 계산 (시장 대비)"""
        # 실제로는 시장 지수 데이터가 필요하지만, 여기서는 근사값 사용
        if len(returns) < 10:
            return 1.0
        
        # BTC 수익률을 시장 프록시로 사용하는 근사
        market_volatility = 0.04  # 일일 4% 변동성 가정
        portfolio_volatility = np.std(returns)
        
        # 상관관계 가정 (0.7)
        correlation = 0.7
        
        return correlation * (portfolio_volatility / market_volatility)
    
    def _calculate_avg_trade_size(self, cutoff_date: datetime) -> float:
        """평균 거래 크기 계산"""
        recent_trades = [
            t for t in self.trade_history 
            if datetime.fromisoformat(t['timestamp']) >= cutoff_date
        ]
        
        if len(recent_trades) == 0:
            return 0
        
        sizes = [abs(t.get('size', 0)) for t in recent_trades]
        return np.mean(sizes) if sizes else 0
    
    def _calculate_largest_win(self, cutoff_date: datetime) -> float:
        """최대 수익 거래"""
        recent_trades = [
            t for t in self.trade_history 
            if datetime.fromisoformat(t['timestamp']) >= cutoff_date and t.get('pnl', 0) > 0
        ]
        
        if len(recent_trades) == 0:
            return 0
        
        return max(t['pnl'] for t in recent_trades)
    
    def _calculate_largest_loss(self, cutoff_date: datetime) -> float:
        """최대 손실 거래"""
        recent_trades = [
            t for t in self.trade_history 
            if datetime.fromisoformat(t['timestamp']) >= cutoff_date and t.get('pnl', 0) < 0
        ]
        
        if len(recent_trades) == 0:
            return 0
        
        return min(t['pnl'] for t in recent_trades)
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """종합 리스크 보고서 생성"""
        try:
            # 다양한 기간별 분석
            metrics_7d = self.calculate_performance_metrics(7)
            metrics_30d = self.calculate_performance_metrics(30)
            metrics_90d = self.calculate_performance_metrics(90)
            
            # 리스크 스코어 계산
            risk_score = self._calculate_risk_score(metrics_30d)
            
            # 포트폴리오 건강도 평가
            health_score = self._calculate_health_score(metrics_30d)
            
            # 개선 제안
            recommendations = self._generate_recommendations(metrics_30d, risk_score)
            
            return {
                'risk_score': risk_score,
                'health_score': health_score,
                'risk_level': self._get_risk_level(risk_score),
                'health_level': self._get_health_level(health_score),
                'metrics_comparison': {
                    '7_days': metrics_7d,
                    '30_days': metrics_30d,
                    '90_days': metrics_90d
                },
                'recommendations': recommendations,
                'alerts': self._generate_alerts(metrics_30d),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"리스크 보고서 생성 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_score(self, metrics: Dict[str, Any]) -> int:
        """리스크 스코어 계산 (0-100, 낮을수록 위험)"""
        score = 100
        
        # 변동성 (높을수록 차감)
        if metrics['volatility'] > 50:
            score -= 30
        elif metrics['volatility'] > 30:
            score -= 20
        elif metrics['volatility'] > 20:
            score -= 10
        
        # 최대 낙폭 (높을수록 차감)
        if metrics['max_drawdown'] > 30:
            score -= 25
        elif metrics['max_drawdown'] > 20:
            score -= 15
        elif metrics['max_drawdown'] > 10:
            score -= 10
        
        # 샤프 비율 (낮을수록 차감)
        if metrics['sharpe_ratio'] < 0:
            score -= 20
        elif metrics['sharpe_ratio'] < 1:
            score -= 10
        elif metrics['sharpe_ratio'] > 2:
            score += 10
        
        # VaR (높을수록 차감)
        if metrics['var_95'] < -10:
            score -= 15
        elif metrics['var_95'] < -5:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> int:
        """포트폴리오 건강도 계산 (0-100)"""
        score = 50  # 기본 점수
        
        # 수익률
        if metrics['total_return'] > 20:
            score += 25
        elif metrics['total_return'] > 10:
            score += 15
        elif metrics['total_return'] > 0:
            score += 10
        else:
            score -= 20
        
        # 승률
        if metrics['win_rate'] > 70:
            score += 20
        elif metrics['win_rate'] > 60:
            score += 15
        elif metrics['win_rate'] > 50:
            score += 10
        elif metrics['win_rate'] < 40:
            score -= 15
        
        # 손익비
        if metrics['profit_loss_ratio'] > 2:
            score += 15
        elif metrics['profit_loss_ratio'] > 1.5:
            score += 10
        elif metrics['profit_loss_ratio'] < 1:
            score -= 10
        
        # 거래 활동성
        if metrics['total_trades'] > 50:
            score += 10
        elif metrics['total_trades'] < 10:
            score -= 5
        
        return max(0, min(100, score))
    
    def _get_risk_level(self, score: int) -> str:
        """리스크 레벨 텍스트"""
        if score >= 80:
            return "낮음 😊"
        elif score >= 60:
            return "보통 😐"
        elif score >= 40:
            return "높음 😰"
        else:
            return "매우 높음 🚨"
    
    def _get_health_level(self, score: int) -> str:
        """건강도 레벨 텍스트"""
        if score >= 80:
            return "우수 💪"
        elif score >= 60:
            return "양호 👍"
        elif score >= 40:
            return "보통 😐"
        else:
            return "주의 필요 ⚠️"
    
    def _generate_recommendations(self, metrics: Dict[str, Any], risk_score: int) -> List[str]:
        """개선 제안 생성"""
        recommendations = []
        
        if metrics['max_drawdown'] > 20:
            recommendations.append("🛡️ 리스크 관리 강화: 포지션 크기를 줄이고 손절선을 더 엄격하게 설정하세요")
        
        if metrics['win_rate'] < 50:
            recommendations.append("🎯 전략 개선: 승률이 낮습니다. 진입 조건을 더 엄격하게 설정해보세요")
        
        if metrics['profit_loss_ratio'] < 1:
            recommendations.append("💰 손익비 개선: 평균 수익이 평균 손실보다 작습니다. 익절 전략을 검토하세요")
        
        if metrics['sharpe_ratio'] < 1:
            recommendations.append("📈 리스크 대비 수익률 개선: 변동성에 비해 수익률이 낮습니다")
        
        if risk_score < 60:
            recommendations.append("⚠️ 전체적인 리스크 관리 점검이 필요합니다")
        
        if len(recommendations) == 0:
            recommendations.append("✅ 현재 포트폴리오 상태가 양호합니다. 현재 전략을 유지하세요")
        
        return recommendations
    
    def _generate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """알림 생성"""
        alerts = []
        
        if metrics['max_drawdown'] > 25:
            alerts.append({
                'level': 'danger',
                'message': f"⚠️ 최대 낙폭이 {metrics['max_drawdown']:.1f}%에 도달했습니다",
                'action': "즉시 리스크 관리 검토 필요"
            })
        
        if metrics['var_99'] < -15:
            alerts.append({
                'level': 'warning',
                'message': f"📉 99% VaR이 {metrics['var_99']:.1f}%입니다",
                'action': "포지션 크기 축소 고려"
            })
        
        if metrics['win_rate'] < 30:
            alerts.append({
                'level': 'warning',
                'message': f"🎯 승률이 {metrics['win_rate']:.1f}%로 낮습니다",
                'action': "전략 재검토 필요"
            })
        
        if len(alerts) == 0:
            alerts.append({
                'level': 'info',
                'message': "✅ 현재 특별한 위험 요소가 감지되지 않았습니다",
                'action': "현재 전략 유지"
            })
        
        return alerts

# 전역 인스턴스
portfolio_analytics = AdvancedPortfolioAnalytics()

def get_portfolio_analytics():
    """포트폴리오 분석기 인스턴스 반환"""
    return portfolio_analytics

if __name__ == "__main__":
    print("🚀 고급 포트폴리오 분석 모듈")
    
    # 테스트 데이터
    analytics = AdvancedPortfolioAnalytics()
    
    # 샘플 포트폴리오 스냅샷
    for i in range(30):
        value = 10000 + np.random.normal(500, 200) * i
        analytics.add_portfolio_snapshot({
            'total_value': value,
            'positions': 3,
            'cash': 1000
        })
    
    # 샘플 거래 기록
    for i in range(50):
        pnl = np.random.normal(50, 100)
        analytics.add_trade_record({
            'symbol': 'BTC/USDT',
            'size': 0.1,
            'pnl': pnl,
            'strategy': 'test'
        })
    
    # 분석 실행
    metrics = analytics.calculate_performance_metrics(30)
    print("성과 지표:", json.dumps(metrics, indent=2, ensure_ascii=False))
    
    risk_report = analytics.generate_risk_report()
    print("리스크 보고서:", json.dumps(risk_report, indent=2, ensure_ascii=False))