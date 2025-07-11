"""
전략 통합 비교분석 시스템
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import asyncio

from .backtest_engine import RealBacktestEngine, BacktestResult
from .portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

@dataclass
class StrategyAnalysis:
    """전략 분석 결과"""
    strategy_name: str
    performance_score: float
    risk_score: float
    consistency_score: float
    market_adaptability: float
    recommendation: str
    strengths: List[str]
    weaknesses: List[str]
    optimization_suggestions: List[str]

@dataclass
class MarketRegimeAnalysis:
    """시장 국면 분석"""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    volatility_level: str  # 'low', 'medium', 'high'
    trend_strength: float
    dominant_patterns: List[str]
    recommended_strategies: List[str]

class StrategyAnalyzer:
    """전략 통합 비교분석기"""
    
    def __init__(self):
        self.backtest_engine = RealBacktestEngine()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.analysis_results = []
        
        # 전략 가중치 설정
        self.strategy_weights = {
            'performance': 0.35,  # 성과
            'risk': 0.25,         # 위험도
            'consistency': 0.20,  # 일관성
            'adaptability': 0.20  # 적응력
        }
        
        # 시장 국면별 전략 선호도
        self.regime_preferences = {
            'bull_strong': ['momentum_strategy', 'ml_ensemble', 'triple_combo'],
            'bull_weak': ['rsi_strategy', 'triple_combo', 'macd_strategy'],
            'sideways': ['rsi_strategy', 'macd_strategy', 'ml_ensemble'],
            'bear_weak': ['rsi_strategy', 'triple_combo', 'ml_ensemble'],
            'bear_strong': ['momentum_strategy', 'ml_ensemble', 'rsi_strategy'],
            'volatile': ['ml_ensemble', 'triple_combo', 'momentum_strategy']
        }
    
    async def analyze_all_strategies(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000000,
        log_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        모든 전략 통합 분석
        
        Args:
            start_date: 분석 시작일
            end_date: 분석 종료일
            initial_capital: 초기 자본
            log_callback: 로그 콜백 함수
            
        Returns:
            Dict: 통합 분석 결과
        """
        try:
            if log_callback:
                log_callback("🔍 전략 통합 분석 시작", "system", 0)
            
            # 시장 국면 분석
            market_regime = await self.analyze_market_regime(start_date, end_date, log_callback)
            
            # 모든 전략 백테스트
            strategy_results = await self.backtest_all_strategies(
                start_date, end_date, initial_capital, log_callback
            )
            
            # 전략별 상세 분석
            strategy_analyses = []
            for result in strategy_results:
                analysis = self.analyze_strategy_performance(result, market_regime)
                strategy_analyses.append(analysis)
            
            # 전략 순위 및 추천
            rankings = self.rank_strategies(strategy_analyses)
            recommendations = self.generate_recommendations(strategy_analyses, market_regime)
            
            # 포트폴리오 조합 제안
            portfolio_combinations = self.suggest_portfolio_combinations(strategy_analyses)
            
            # 포트폴리오 최적화 실행
            portfolio_optimization = None
            if strategy_results:
                try:
                    # 전략 결과를 포트폴리오 최적화 형태로 변환
                    portfolio_strategy_results = []
                    for result in strategy_results:
                        portfolio_strategy_results.append({
                            'strategy_name': result.strategy_name,
                            'total_return': result.total_return,
                            'sharpe_ratio': result.sharpe_ratio,
                            'max_drawdown': result.max_drawdown,
                            'win_rate': result.win_rate,
                            'volatility': result.total_return / result.sharpe_ratio if result.sharpe_ratio > 0 else 20.0
                        })
                    
                    # 다양한 최적화 방법으로 포트폴리오 생성
                    optimized_portfolios = self.portfolio_optimizer.optimize_portfolio(
                        strategy_results=portfolio_strategy_results,
                        optimization_method='all',
                        risk_level='medium'
                    )
                    
                    # 포트폴리오 보고서 생성
                    portfolio_report = self.portfolio_optimizer.generate_portfolio_report(optimized_portfolios)
                    
                    portfolio_optimization = {
                        'portfolios': optimized_portfolios,
                        'report': portfolio_report
                    }
                    
                    if log_callback:
                        log_callback(f"📊 포트폴리오 최적화 완료 ({len(optimized_portfolios)}개)", "system", 95)
                        
                except Exception as e:
                    logger.error(f"포트폴리오 최적화 실패: {e}")
                    if log_callback:
                        log_callback(f"⚠️ 포트폴리오 최적화 실패: {str(e)}", "warning", 95)
            
            if log_callback:
                log_callback("✅ 전략 통합 분석 완료", "system", 100)
            
            return {
                'market_regime': market_regime,
                'strategy_results': strategy_results,
                'strategy_analyses': strategy_analyses,
                'rankings': rankings,
                'recommendations': recommendations,
                'portfolio_combinations': portfolio_combinations,
                'portfolio_optimization': portfolio_optimization,
                'analysis_summary': self.generate_analysis_summary(strategy_analyses, market_regime)
            }
            
        except Exception as e:
            logger.error(f"전략 통합 분석 실패: {e}")
            if log_callback:
                log_callback(f"❌ 분석 실패: {str(e)}", "error", 0)
            raise e
    
    async def analyze_market_regime(
        self,
        start_date: datetime,
        end_date: datetime,
        log_callback: Optional[callable] = None
    ) -> MarketRegimeAnalysis:
        """시장 국면 분석"""
        try:
            if log_callback:
                log_callback("📊 시장 국면 분석 중...", "analysis", 10)
            
            # BTC 데이터로 시장 국면 분석
            btc_data = await self.backtest_engine.data_manager.download_historical_data(
                'BTC/USDT', '1h', start_date, end_date
            )
            
            if btc_data.empty:
                raise ValueError("시장 데이터를 가져올 수 없습니다")
            
            # 기술적 지표 추가
            btc_data = self.backtest_engine.data_manager.add_technical_indicators(btc_data)
            
            # 트렌드 분석
            returns = btc_data['close'].pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            
            # 변동성 계산
            volatility = returns.std() * np.sqrt(24 * 365)  # 연환산
            
            # 트렌드 강도 계산
            trend_strength = abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
            
            # 국면 분류
            avg_return = returns.mean()
            if avg_return > 0.001:  # 강한 상승
                regime_type = 'bull_strong'
            elif avg_return > 0:     # 약한 상승
                regime_type = 'bull_weak'
            elif avg_return > -0.001:  # 횡보
                regime_type = 'sideways'
            elif avg_return > -0.002:  # 약한 하락
                regime_type = 'bear_weak'
            else:                    # 강한 하락
                regime_type = 'bear_strong'
            
            # 변동성 수준
            if volatility < 0.3:
                volatility_level = 'low'
            elif volatility < 0.6:
                volatility_level = 'medium'
            else:
                volatility_level = 'high'
            
            # 패턴 분석
            dominant_patterns = self.identify_market_patterns(btc_data)
            
            # 추천 전략
            recommended_strategies = self.regime_preferences.get(regime_type, [])
            
            if log_callback:
                log_callback(f"📈 시장 국면: {regime_type} ({volatility_level} 변동성)", "analysis", 15)
            
            return MarketRegimeAnalysis(
                regime_type=regime_type,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
                dominant_patterns=dominant_patterns,
                recommended_strategies=recommended_strategies
            )
            
        except Exception as e:
            logger.error(f"시장 국면 분석 실패: {e}")
            # 기본값 반환
            return MarketRegimeAnalysis(
                regime_type='sideways',
                volatility_level='medium',
                trend_strength=0.5,
                dominant_patterns=['범위권 거래'],
                recommended_strategies=['rsi_strategy', 'macd_strategy']
            )
    
    def identify_market_patterns(self, data: pd.DataFrame) -> List[str]:
        """시장 패턴 식별"""
        try:
            patterns = []
            
            # RSI 패턴
            rsi_avg = data['RSI'].mean()
            if rsi_avg > 70:
                patterns.append('과매수 상태')
            elif rsi_avg < 30:
                patterns.append('과매도 상태')
            else:
                patterns.append('RSI 중립')
            
            # MACD 패턴
            macd_positive = (data['MACD'] > data['MACD_Signal']).sum()
            macd_total = len(data)
            if macd_positive / macd_total > 0.6:
                patterns.append('MACD 상승 추세')
            elif macd_positive / macd_total < 0.4:
                patterns.append('MACD 하락 추세')
            else:
                patterns.append('MACD 혼재')
            
            # 볼린저 밴드 패턴
            bb_squeeze = ((data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']).mean()
            if bb_squeeze < 0.04:
                patterns.append('볼린저 밴드 수축')
            elif bb_squeeze > 0.08:
                patterns.append('볼린저 밴드 확장')
            else:
                patterns.append('볼린저 밴드 정상')
            
            return patterns
            
        except Exception as e:
            logger.error(f"패턴 식별 실패: {e}")
            return ['분석 불가']
    
    async def backtest_all_strategies(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        log_callback: Optional[callable] = None
    ) -> List[BacktestResult]:
        """모든 전략 백테스트"""
        try:
            if log_callback:
                log_callback("🚀 전체 전략 백테스트 시작", "system", 20)
            
            results = []
            strategies = list(self.backtest_engine.strategies.keys())
            
            for i, strategy_id in enumerate(strategies):
                try:
                    if log_callback:
                        progress = 20 + (i / len(strategies)) * 60
                        log_callback(f"  └─ {self.backtest_engine.strategies[strategy_id]['name']} 테스트 중...", "analysis", progress)
                    
                    # 백테스트 설정
                    config = {
                        'strategy': strategy_id,
                        'symbol': 'BTC/USDT',  # 대표 심볼
                        'symbol_type': 'individual',
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'timeframe': self.backtest_engine.strategies[strategy_id]['timeframe'],
                        'initial_capital': initial_capital,
                        'ml_optimization': True
                    }
                    
                    # 백테스트 실행
                    result = await self.backtest_engine.run_backtest(config, None)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"전략 {strategy_id} 백테스트 실패: {e}")
                    continue
            
            if log_callback:
                log_callback(f"✅ 전체 전략 백테스트 완료 ({len(results)}개)", "system", 80)
            
            return results
            
        except Exception as e:
            logger.error(f"전체 전략 백테스트 실패: {e}")
            return []
    
    def analyze_strategy_performance(
        self,
        result: BacktestResult,
        market_regime: MarketRegimeAnalysis
    ) -> StrategyAnalysis:
        """전략 성과 분석"""
        try:
            # 성과 점수 (0-100)
            performance_score = min(100, max(0, result.total_return + 50))
            
            # 위험 점수 (낮을수록 좋음, 역산)
            risk_score = max(0, 100 - result.max_drawdown * 2)
            
            # 일관성 점수
            consistency_score = min(100, result.sharpe_ratio * 50) if result.sharpe_ratio > 0 else 0
            
            # 시장 적응성 점수
            regime_fit = 1.0
            if result.strategy_name.lower().replace(' ', '_') in market_regime.recommended_strategies:
                regime_fit = 1.2
            
            adaptability_score = min(100, result.win_rate * regime_fit)
            
            # 종합 점수
            total_score = (
                performance_score * self.strategy_weights['performance'] +
                risk_score * self.strategy_weights['risk'] +
                consistency_score * self.strategy_weights['consistency'] +
                adaptability_score * self.strategy_weights['adaptability']
            )
            
            # 추천 생성
            recommendation = self.generate_strategy_recommendation(
                result, total_score, market_regime
            )
            
            # 강점과 약점 분석
            strengths, weaknesses = self.analyze_strengths_weaknesses(result)
            
            # 최적화 제안
            optimization_suggestions = self.generate_optimization_suggestions(result)
            
            return StrategyAnalysis(
                strategy_name=result.strategy_name,
                performance_score=performance_score,
                risk_score=risk_score,
                consistency_score=consistency_score,
                market_adaptability=adaptability_score,
                recommendation=recommendation,
                strengths=strengths,
                weaknesses=weaknesses,
                optimization_suggestions=optimization_suggestions
            )
            
        except Exception as e:
            logger.error(f"전략 성과 분석 실패: {e}")
            return StrategyAnalysis(
                strategy_name=result.strategy_name,
                performance_score=0,
                risk_score=0,
                consistency_score=0,
                market_adaptability=0,
                recommendation="분석 불가",
                strengths=[],
                weaknesses=[],
                optimization_suggestions=[]
            )
    
    def generate_strategy_recommendation(
        self,
        result: BacktestResult,
        score: float,
        market_regime: MarketRegimeAnalysis
    ) -> str:
        """전략 추천 생성"""
        try:
            if score >= 80:
                return "🌟 최적 전략 - 적극 활용 권장"
            elif score >= 70:
                return "✅ 우수 전략 - 활용 권장"
            elif score >= 60:
                return "⚠️ 보통 전략 - 조건부 활용"
            elif score >= 50:
                return "🔄 개선 필요 - 파라미터 최적화 권장"
            elif score >= 40:
                return "⚡ 대폭 개선 필요 - 전략 재검토 권장"
            else:
                return "❌ 폐기 권장 - 다른 전략 사용"
                
        except Exception as e:
            logger.error(f"전략 추천 생성 실패: {e}")
            return "분석 불가"
    
    def analyze_strengths_weaknesses(self, result: BacktestResult) -> Tuple[List[str], List[str]]:
        """강점과 약점 분석"""
        try:
            strengths = []
            weaknesses = []
            
            # 수익률 분석
            if result.total_return > 20:
                strengths.append(f"높은 수익률 ({result.total_return:.1f}%)")
            elif result.total_return < 5:
                weaknesses.append(f"낮은 수익률 ({result.total_return:.1f}%)")
            
            # 위험 분석
            if result.max_drawdown < 10:
                strengths.append(f"낮은 최대 낙폭 ({result.max_drawdown:.1f}%)")
            elif result.max_drawdown > 20:
                weaknesses.append(f"높은 최대 낙폭 ({result.max_drawdown:.1f}%)")
            
            # 승률 분석
            if result.win_rate > 70:
                strengths.append(f"높은 승률 ({result.win_rate:.1f}%)")
            elif result.win_rate < 50:
                weaknesses.append(f"낮은 승률 ({result.win_rate:.1f}%)")
            
            # 샤프 비율 분석
            if result.sharpe_ratio > 1.5:
                strengths.append(f"우수한 샤프 비율 ({result.sharpe_ratio:.2f})")
            elif result.sharpe_ratio < 0.5:
                weaknesses.append(f"낮은 샤프 비율 ({result.sharpe_ratio:.2f})")
            
            # 거래 빈도 분석
            if result.total_trades > 100:
                strengths.append("활발한 거래 빈도")
            elif result.total_trades < 20:
                weaknesses.append("거래 기회 부족")
            
            return strengths, weaknesses
            
        except Exception as e:
            logger.error(f"강점/약점 분석 실패: {e}")
            return [], []
    
    def generate_optimization_suggestions(self, result: BacktestResult) -> List[str]:
        """최적화 제안 생성"""
        try:
            suggestions = []
            
            # 수익률 개선
            if result.total_return < 15:
                suggestions.append("포지션 사이즈 조정을 통한 수익률 개선")
                suggestions.append("진입/청산 조건 강화")
            
            # 위험 관리
            if result.max_drawdown > 15:
                suggestions.append("스톱로스 조건 강화")
                suggestions.append("동적 레버리지 조정")
            
            # 승률 개선
            if result.win_rate < 60:
                suggestions.append("신호 필터링 강화")
                suggestions.append("시장 조건별 적응형 파라미터 적용")
            
            # 거래 빈도 조정
            if result.total_trades < 30:
                suggestions.append("거래 빈도 증가를 위한 조건 완화")
            elif result.total_trades > 200:
                suggestions.append("과도한 거래 방지를 위한 조건 강화")
            
            # ML 최적화
            if not result.ml_optimized:
                suggestions.append("머신러닝 최적화 적용")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"최적화 제안 생성 실패: {e}")
            return []
    
    def rank_strategies(self, analyses: List[StrategyAnalysis]) -> List[Dict[str, Any]]:
        """전략 순위 매기기"""
        try:
            # 종합 점수 계산
            scored_strategies = []
            for analysis in analyses:
                total_score = (
                    analysis.performance_score * self.strategy_weights['performance'] +
                    analysis.risk_score * self.strategy_weights['risk'] +
                    analysis.consistency_score * self.strategy_weights['consistency'] +
                    analysis.market_adaptability * self.strategy_weights['adaptability']
                )
                
                scored_strategies.append({
                    'strategy_name': analysis.strategy_name,
                    'total_score': total_score,
                    'performance_score': analysis.performance_score,
                    'risk_score': analysis.risk_score,
                    'consistency_score': analysis.consistency_score,
                    'adaptability_score': analysis.market_adaptability,
                    'recommendation': analysis.recommendation
                })
            
            # 점수 기준 정렬
            scored_strategies.sort(key=lambda x: x['total_score'], reverse=True)
            
            # 순위 추가
            for i, strategy in enumerate(scored_strategies):
                strategy['rank'] = i + 1
            
            return scored_strategies
            
        except Exception as e:
            logger.error(f"전략 순위 매기기 실패: {e}")
            return []
    
    def generate_recommendations(
        self,
        analyses: List[StrategyAnalysis],
        market_regime: MarketRegimeAnalysis
    ) -> Dict[str, Any]:
        """통합 추천 생성"""
        try:
            recommendations = {
                'market_analysis': {
                    'regime': market_regime.regime_type,
                    'volatility': market_regime.volatility_level,
                    'trend_strength': market_regime.trend_strength,
                    'key_patterns': market_regime.dominant_patterns
                },
                'strategy_recommendations': {
                    'top_strategies': [],
                    'avoid_strategies': [],
                    'optimization_needed': []
                },
                'portfolio_suggestions': [],
                'risk_management': [],
                'market_timing': []
            }
            
            # 전략별 분류
            for analysis in analyses:
                total_score = (
                    analysis.performance_score * self.strategy_weights['performance'] +
                    analysis.risk_score * self.strategy_weights['risk'] +
                    analysis.consistency_score * self.strategy_weights['consistency'] +
                    analysis.market_adaptability * self.strategy_weights['adaptability']
                )
                
                if total_score >= 70:
                    recommendations['strategy_recommendations']['top_strategies'].append({
                        'name': analysis.strategy_name,
                        'score': total_score,
                        'strengths': analysis.strengths
                    })
                elif total_score >= 50:
                    recommendations['strategy_recommendations']['optimization_needed'].append({
                        'name': analysis.strategy_name,
                        'score': total_score,
                        'suggestions': analysis.optimization_suggestions
                    })
                else:
                    recommendations['strategy_recommendations']['avoid_strategies'].append({
                        'name': analysis.strategy_name,
                        'score': total_score,
                        'weaknesses': analysis.weaknesses
                    })
            
            # 포트폴리오 제안
            recommendations['portfolio_suggestions'] = self.suggest_portfolio_combinations(analyses)
            
            # 위험 관리 제안
            recommendations['risk_management'] = [
                f"현재 시장 변동성: {market_regime.volatility_level}",
                "동적 레버리지 관리 필수",
                "분할 진입/청산 전략 활용",
                "시장 국면별 전략 전환 준비"
            ]
            
            # 시장 타이밍 제안
            recommendations['market_timing'] = [
                f"현재 시장 국면: {market_regime.regime_type}",
                f"추천 전략: {', '.join(market_regime.recommended_strategies)}",
                "시장 국면 변화 시 전략 재평가 필요"
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"통합 추천 생성 실패: {e}")
            return {}
    
    def suggest_portfolio_combinations(self, analyses: List[StrategyAnalysis]) -> List[Dict[str, Any]]:
        """포트폴리오 조합 제안"""
        try:
            combinations = []
            
            # 성과 기준 상위 전략들
            top_strategies = sorted(analyses, key=lambda x: x.performance_score, reverse=True)[:3]
            
            # 조합 1: 균형 포트폴리오
            if len(top_strategies) >= 2:
                combinations.append({
                    'name': '균형 포트폴리오',
                    'strategies': [
                        {'name': top_strategies[0].strategy_name, 'weight': 0.4},
                        {'name': top_strategies[1].strategy_name, 'weight': 0.3},
                        {'name': top_strategies[2].strategy_name if len(top_strategies) > 2 else top_strategies[0].strategy_name, 'weight': 0.3}
                    ],
                    'expected_return': sum(s.performance_score for s in top_strategies[:2]) / 2,
                    'risk_level': 'Medium'
                })
            
            # 조합 2: 고수익 포트폴리오
            high_return_strategies = sorted(analyses, key=lambda x: x.performance_score, reverse=True)[:2]
            if len(high_return_strategies) >= 2:
                combinations.append({
                    'name': '고수익 포트폴리오',
                    'strategies': [
                        {'name': high_return_strategies[0].strategy_name, 'weight': 0.6},
                        {'name': high_return_strategies[1].strategy_name, 'weight': 0.4}
                    ],
                    'expected_return': sum(s.performance_score for s in high_return_strategies) / 2,
                    'risk_level': 'High'
                })
            
            # 조합 3: 안전 포트폴리오
            safe_strategies = sorted(analyses, key=lambda x: x.risk_score, reverse=True)[:2]
            if len(safe_strategies) >= 2:
                combinations.append({
                    'name': '안전 포트폴리오',
                    'strategies': [
                        {'name': safe_strategies[0].strategy_name, 'weight': 0.5},
                        {'name': safe_strategies[1].strategy_name, 'weight': 0.5}
                    ],
                    'expected_return': sum(s.performance_score for s in safe_strategies) / 2,
                    'risk_level': 'Low'
                })
            
            return combinations
            
        except Exception as e:
            logger.error(f"포트폴리오 조합 제안 실패: {e}")
            return []
    
    def generate_analysis_summary(
        self,
        analyses: List[StrategyAnalysis],
        market_regime: MarketRegimeAnalysis
    ) -> Dict[str, Any]:
        """분석 요약 생성"""
        try:
            # 평균 성과 계산
            avg_performance = np.mean([a.performance_score for a in analyses])
            avg_risk = np.mean([a.risk_score for a in analyses])
            avg_consistency = np.mean([a.consistency_score for a in analyses])
            
            # 최고/최저 전략
            best_strategy = max(analyses, key=lambda x: x.performance_score)
            worst_strategy = min(analyses, key=lambda x: x.performance_score)
            
            # 요약 생성
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'market_regime': market_regime.regime_type,
                'total_strategies_analyzed': len(analyses),
                'average_performance': avg_performance,
                'average_risk_score': avg_risk,
                'average_consistency': avg_consistency,
                'best_strategy': {
                    'name': best_strategy.strategy_name,
                    'score': best_strategy.performance_score,
                    'recommendation': best_strategy.recommendation
                },
                'worst_strategy': {
                    'name': worst_strategy.strategy_name,
                    'score': worst_strategy.performance_score,
                    'recommendation': worst_strategy.recommendation
                },
                'key_insights': [
                    f"현재 시장 국면: {market_regime.regime_type}",
                    f"최고 성과 전략: {best_strategy.strategy_name}",
                    f"평균 성과 점수: {avg_performance:.1f}",
                    f"시장 변동성: {market_regime.volatility_level}"
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"분석 요약 생성 실패: {e}")
            return {}