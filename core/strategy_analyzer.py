"""
전략 통합 비교분석 시스템 - 수정 버전
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
    """전략 통합 비교분석기 - 수정 버전"""
    
    def __init__(self):
        self.backtest_engine = RealBacktestEngine()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.analysis_results = []
        
        # 전략 가중치 설정
        self.strategy_weights = {
            'performance': 0.35,  # 성과
            'risk': 0.25,         # 위험도
            'consistency': 0.20,  # 일관성
            'adaptability': 0.20  # 적응성
        }
        
        # 시장 국면별 전략 선호도
        self.regime_preferences = {
            'bull_strong': ['momentum_strategy', 'ml_ensemble', 'triple_combo'],
            'bull_weak': ['triple_combo', 'macd_strategy', 'rsi_strategy'],
            'sideways': ['rsi_strategy', 'macd_strategy', 'triple_combo'],
            'bear_weak': ['rsi_strategy', 'triple_combo', 'macd_strategy'],
            'bear_strong': ['rsi_strategy', 'momentum_strategy', 'ml_ensemble']
        }
    
    async def analyze_all_strategies(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        log_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """모든 전략 통합 분석"""
        try:
            if log_callback:
                log_callback("🚀 전략 통합 분석 시작", "system", 5)
            
            # 1. 시장 국면 분석
            market_regime = await self.analyze_market_regime_safe(start_date, end_date, log_callback)
            
            # 2. 전략별 백테스트
            strategy_results = await self.run_all_strategies_backtest(
                start_date, end_date, initial_capital, log_callback
            )
            
            # 3. 전략 성능 분석
            strategy_analyses = []
            for result in strategy_results:
                analysis = self.analyze_strategy_performance_safe(result, market_regime)
                strategy_analyses.append(analysis)
            
            # 4. 전략 순위 및 추천
            rankings = self.rank_strategies_safe(strategy_analyses)
            recommendations = self.generate_recommendations_safe(strategy_analyses, market_regime)
            
            # 5. 포트폴리오 조합 제안
            portfolio_combinations = self.suggest_portfolio_combinations_safe(strategy_analyses)
            
            # 6. 포트폴리오 최적화 실행
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
                'analysis_summary': self.generate_analysis_summary_safe(strategy_analyses, market_regime)
            }
            
        except Exception as e:
            logger.error(f"전략 통합 분석 실패: {e}")
            if log_callback:
                log_callback(f"❌ 분석 실패: {str(e)}", "error", 0)
            raise e
    
    async def analyze_market_regime_safe(
        self,
        start_date: datetime,
        end_date: datetime,
        log_callback: Optional[callable] = None
    ) -> MarketRegimeAnalysis:
        """시장 국면 분석 - 안전한 버전"""
        try:
            if log_callback:
                log_callback("📊 시장 국면 분석 중...", "analysis", 10)
            
            # BTC 데이터로 시장 국면 분석 - 로컬 데이터 사용
            try:
                btc_data = await self.backtest_engine.data_manager.download_historical_data(
                    'BTC/USDT', '1h', start_date, end_date
                )
            except Exception as e:
                if log_callback:
                    log_callback(f"⚠️ 실시간 데이터 로드 실패, 로컬 데이터 사용: {str(e)}", "warning", 12)
                
                # 로컬 데이터 사용
                try:
                    btc_data = self.backtest_engine.data_manager.load_market_data('BTC/USDT', '1h')
                except Exception as local_error:
                    if log_callback:
                        log_callback(f"⚠️ 로컬 데이터도 실패, 기본값 사용: {str(local_error)}", "warning", 15)
                    btc_data = pd.DataFrame()
            
            if btc_data.empty:
                # 데이터가 없으면 기본값 반환
                if log_callback:
                    log_callback("📊 데이터 부족으로 기본 시장 국면 사용", "warning", 15)
                return MarketRegimeAnalysis(
                    regime_type='sideways',
                    volatility_level='medium',
                    trend_strength=0.0,
                    dominant_patterns=['데이터 부족'],
                    recommended_strategies=['triple_combo', 'rsi_strategy']
                )
            
            # 기술적 지표 추가
            btc_data = self.backtest_engine.data_manager.add_technical_indicators(btc_data)
            
            # 트렌드 분석 - 안전한 데이터 처리
            volatility = 0.2  # 기본값
            trend_strength = 0.0
            regime_type = 'sideways'
            
            try:
                if isinstance(btc_data, pd.DataFrame) and len(btc_data) > 1:
                    returns = btc_data['close'].pct_change().dropna()
                    
                    if len(returns) > 0:
                        # 변동성 계산
                        volatility = float(returns.std() * np.sqrt(24 * 365))  # 연환산
                        
                        # 트렌드 강도 계산
                        returns_mean = float(returns.mean())
                        returns_std = float(returns.std())
                        trend_strength = abs(returns_mean) / returns_std if returns_std > 0 else 0.0
                        
                        # 국면 분류
                        if returns_mean > 0.001:  # 강한 상승
                            regime_type = 'bull_strong'
                        elif returns_mean > 0:     # 약한 상승
                            regime_type = 'bull_weak'
                        elif returns_mean > -0.001:  # 횡보
                            regime_type = 'sideways'
                        elif returns_mean > -0.002:  # 약한 하락
                            regime_type = 'bear_weak'
                        else:                    # 강한 하락
                            regime_type = 'bear_strong'
                        
            except Exception as e:
                logger.error(f"시장 국면 분석 실패: {e}")
                volatility = 0.2
                trend_strength = 0.0
                regime_type = 'sideways'
            
            # 변동성 수준
            if volatility < 0.3:
                volatility_level = 'low'
            elif volatility < 0.6:
                volatility_level = 'medium'
            else:
                volatility_level = 'high'
            
            # 패턴 분석
            dominant_patterns = self.identify_market_patterns_safe(btc_data)
            
            # 추천 전략
            recommended_strategies = self.regime_preferences.get(regime_type, ['triple_combo', 'rsi_strategy'])
            
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
    
    def identify_market_patterns_safe(self, data: pd.DataFrame) -> List[str]:
        """시장 패턴 식별 - 안전한 버전"""
        try:
            patterns = []
            
            # 데이터 유효성 검사
            if data.empty or len(data) == 0:
                return ['데이터 부족']
            
            # RSI 패턴
            if 'RSI' in data.columns and not data['RSI'].empty:
                try:
                    rsi_avg = float(data['RSI'].mean()) if not pd.isna(data['RSI'].mean()) else 50
                    if rsi_avg > 70:
                        patterns.append('과매수 상태')
                    elif rsi_avg < 30:
                        patterns.append('과매도 상태')
                    else:
                        patterns.append('RSI 중립')
                except Exception as e:
                    logger.error(f"RSI 패턴 분석 실패: {e}")
                    patterns.append('RSI 분석 실패')
            
            # MACD 패턴
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                try:
                    macd_positive = (data['MACD'] > data['MACD_Signal']).sum()
                    macd_total = len(data)
                    if macd_total > 0:
                        ratio = macd_positive / macd_total
                        if ratio > 0.6:
                            patterns.append('MACD 상승 추세')
                        elif ratio < 0.4:
                            patterns.append('MACD 하락 추세')
                        else:
                            patterns.append('MACD 중립')
                except Exception as e:
                    logger.error(f"MACD 패턴 분석 실패: {e}")
                    patterns.append('MACD 분석 실패')
            
            return patterns if patterns else ['패턴 분석 실패']
            
        except Exception as e:
            logger.error(f"시장 패턴 식별 실패: {e}")
            return ['패턴 분석 실패']
    
    async def run_all_strategies_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        log_callback: Optional[callable] = None
    ) -> List[BacktestResult]:
        """모든 전략 백테스트 - 안전한 버전"""
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
    
    def analyze_strategy_performance_safe(self, result: BacktestResult, market_regime: MarketRegimeAnalysis) -> StrategyAnalysis:
        """전략 성능 분석 - 안전한 버전"""
        try:
            # 기본 점수 계산
            performance_score = max(0, min(100, result.total_return))
            risk_score = max(0, 100 - result.max_drawdown)
            consistency_score = max(0, min(100, result.win_rate))
            adaptability_score = max(0, min(100, result.sharpe_ratio * 20))
            
            # 종합 점수 계산
            total_score = (
                performance_score * self.strategy_weights['performance'] +
                risk_score * self.strategy_weights['risk'] +
                consistency_score * self.strategy_weights['consistency'] +
                adaptability_score * self.strategy_weights['adaptability']
            )
            
            # 추천 등급 결정
            if total_score >= 80:
                recommendation = '강력 추천'
            elif total_score >= 60:
                recommendation = '추천'
            elif total_score >= 40:
                recommendation = '보통'
            else:
                recommendation = '비추천'
            
            # 강점과 약점 분석
            strengths = []
            weaknesses = []
            
            if result.total_return > 10:
                strengths.append('높은 수익률')
            if result.max_drawdown < 10:
                strengths.append('낮은 리스크')
            if result.win_rate > 60:
                strengths.append('높은 승률')
            if result.sharpe_ratio > 1.5:
                strengths.append('우수한 리스크 대비 수익')
            
            if result.total_return < 5:
                weaknesses.append('낮은 수익률')
            if result.max_drawdown > 20:
                weaknesses.append('높은 리스크')
            if result.win_rate < 40:
                weaknesses.append('낮은 승률')
            if result.sharpe_ratio < 0.5:
                weaknesses.append('낮은 리스크 대비 수익')
            
            # 최적화 제안
            optimization_suggestions = []
            if result.max_drawdown > 15:
                optimization_suggestions.append('손실 제한 강화')
            if result.win_rate < 50:
                optimization_suggestions.append('진입 신호 정확도 개선')
            if result.total_trades < 10:
                optimization_suggestions.append('거래 빈도 증가')
            
            return StrategyAnalysis(
                strategy_name=result.strategy_name,
                performance_score=performance_score,
                risk_score=risk_score,
                consistency_score=consistency_score,
                market_adaptability=adaptability_score,
                recommendation=recommendation,
                strengths=strengths or ['분석 결과 없음'],
                weaknesses=weaknesses or ['분석 결과 없음'],
                optimization_suggestions=optimization_suggestions or ['최적화 제안 없음']
            )
            
        except Exception as e:
            logger.error(f"전략 성능 분석 실패: {e}")
            return StrategyAnalysis(
                strategy_name=result.strategy_name if hasattr(result, 'strategy_name') else '알 수 없음',
                performance_score=0,
                risk_score=0,
                consistency_score=0,
                market_adaptability=0,
                recommendation='분석 실패',
                strengths=['분석 실패'],
                weaknesses=['분석 실패'],
                optimization_suggestions=['분석 실패']
            )
    
    def rank_strategies_safe(self, analyses: List[StrategyAnalysis]) -> List[Dict]:
        """전략 순위 매기기 - 안전한 버전"""
        try:
            rankings = []
            for analysis in analyses:
                score = (
                    analysis.performance_score * 0.4 +
                    analysis.risk_score * 0.3 +
                    analysis.consistency_score * 0.2 +
                    analysis.market_adaptability * 0.1
                )
                rankings.append({
                    'strategy': analysis.strategy_name,
                    'score': score,
                    'recommendation': analysis.recommendation
                })
            
            # 점수 순으로 정렬
            rankings.sort(key=lambda x: x['score'], reverse=True)
            return rankings
            
        except Exception as e:
            logger.error(f"전략 순위 매기기 실패: {e}")
            return []
    
    def generate_recommendations_safe(self, analyses: List[StrategyAnalysis], market_regime: MarketRegimeAnalysis) -> List[Dict]:
        """추천 사항 생성 - 안전한 버전"""
        try:
            recommendations = []
            
            # 상위 전략 추천
            if analyses:
                best_strategy = max(analyses, key=lambda x: x.performance_score)
                recommendations.append({
                    'type': '최고 성과 전략',
                    'strategy': best_strategy.strategy_name,
                    'reason': f'수익률 {best_strategy.performance_score:.1f}점으로 최고 성과'
                })
            
            # 시장 국면별 추천
            if market_regime.recommended_strategies:
                recommendations.append({
                    'type': '시장 국면 적합 전략',
                    'strategy': market_regime.recommended_strategies[0],
                    'reason': f'{market_regime.regime_type} 시장에 적합'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"추천 사항 생성 실패: {e}")
            return []
    
    def suggest_portfolio_combinations_safe(self, analyses: List[StrategyAnalysis]) -> List[Dict]:
        """포트폴리오 조합 제안 - 안전한 버전"""
        try:
            combinations = []
            
            if len(analyses) >= 2:
                # 상위 2개 전략 조합
                top_strategies = sorted(analyses, key=lambda x: x.performance_score, reverse=True)[:2]
                combinations.append({
                    'name': '고성과 조합',
                    'strategies': [s.strategy_name for s in top_strategies],
                    'weights': [0.6, 0.4],
                    'reason': '높은 성과 전략들의 조합'
                })
            
            return combinations
            
        except Exception as e:
            logger.error(f"포트폴리오 조합 제안 실패: {e}")
            return []
    
    def generate_analysis_summary_safe(self, analyses: List[StrategyAnalysis], market_regime: MarketRegimeAnalysis) -> Dict:
        """분석 요약 생성 - 안전한 버전"""
        try:
            if not analyses:
                return {'summary': '분석 결과 없음'}
            
            avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
            avg_risk = sum(a.risk_score for a in analyses) / len(analyses)
            
            return {
                'total_strategies': len(analyses),
                'avg_performance': avg_performance,
                'avg_risk': avg_risk,
                'market_regime': market_regime.regime_type,
                'volatility_level': market_regime.volatility_level,
                'summary': f'{len(analyses)}개 전략 분석 완료, 평균 성과 {avg_performance:.1f}점'
            }
            
        except Exception as e:
            logger.error(f"분석 요약 생성 실패: {e}")
            return {'summary': '분석 요약 생성 실패'}