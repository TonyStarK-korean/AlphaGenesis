import pandas as pd
import numpy as np
import sys
import os
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# 프로젝트 루트를 파이썬 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.market_analysis.market_regime_analyzer import MarketRegimeAnalyzer, MarketRegime
from src.ml_prediction.price_prediction_engine import PricePredictionEngine
from src.strategy_manager.advanced_volatility_momentum import AdvancedVolatilityMomentumStrategy
from src.strategy_manager.ai_mean_reversion import AIMeanReversionStrategy
from tests.advanced_backtest_engine import AdvancedBacktestEngine

class UltimateTradingSystem:
    """
    코인선물 전세계 상위 0.01% 장중매매 시스템
    - 시장 국면 분석 (5가지)
    - ML 기반 상승 예측
    - 시장 국면별 전략 선택
    - 동적 레버리지 조정
    - 실시간 백테스팅
    """
    
    def __init__(self):
        self.market_analyzer = MarketRegimeAnalyzer()
        self.prediction_engine = PricePredictionEngine()
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = []
        self.trading_results = []
        
    def run_complete_analysis(self, data: pd.DataFrame):
        """완전한 분석 및 백테스팅 실행"""
        print("🚀 코인선물 전세계 상위 0.01% 장중매매 시스템")
        print("=" * 80)
        
        # 1. ML 모델 훈련
        print("🤖 1단계: ML 모델 훈련")
        self.prediction_engine.train_models(data)
        print()
        
        # 2. 시장 국면 분석
        print("📊 2단계: 시장 국면 분석")
        self._analyze_market_regimes(data)
        print()
        
        # 3. 시장 국면별 백테스팅
        print("🔄 3단계: 시장 국면별 백테스팅")
        self._run_regime_specific_backtests(data)
        print()
        
        # 4. 최종 결과 분석
        print("📈 4단계: 최종 결과 분석")
        self._analyze_final_results()
        
    def _analyze_market_regimes(self, data: pd.DataFrame):
        """시장 국면 분석"""
        print("🔄 시장 국면 분석 중...")
        
        # 시간별 시장 국면 분석 (1시간마다)
        analysis_interval = 60  # 60분마다 분석
        
        for i in range(100, len(data), analysis_interval):
            if i >= len(data):
                break
                
            current_data = data.iloc[:i+1]
            regime = self.market_analyzer.analyze_market_regime(current_data)
            
            self.regime_history.append({
                'timestamp': data.index[i],
                'regime': regime,
                'price': data['Close'].iloc[i]
            })
            
            # 진행률 표시
            if i % (len(data) // 10) == 0:
                progress = (i / len(data)) * 100
                print(f"   📊 시장 국면 분석 진행률: {progress:.1f}%")
        
        # 국면별 통계
        regime_counts = {}
        for record in self.regime_history:
            regime_name = record['regime'].value
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
        
        print("\n📊 시장 국면 분석 결과:")
        total_periods = len(self.regime_history)
        for regime, count in regime_counts.items():
            percentage = (count / total_periods) * 100
            print(f"   - {regime}: {count}회 ({percentage:.1f}%)")
            
        # 가장 빈번한 국면
        most_common_regime = max(regime_counts.items(), key=lambda x: x[1])
        print(f"   🎯 가장 빈번한 국면: {most_common_regime[0]} ({most_common_regime[1]}회)")
        
    def _run_regime_specific_backtests(self, data: pd.DataFrame):
        """시장 국면별 백테스팅"""
        print("🔄 시장 국면별 백테스팅 실행 중...")
        
        # 국면별로 데이터 분할
        regime_data = {}
        current_regime = None
        current_start_idx = 0
        
        for i, record in enumerate(self.regime_history):
            if record['regime'] != current_regime:
                # 이전 국면 데이터 저장
                if current_regime is not None:
                    regime_name = current_regime.value
                    if regime_name not in regime_data:
                        regime_data[regime_name] = []
                    regime_data[regime_name].append(data.iloc[current_start_idx:i*60])
                
                # 새 국면 시작
                current_regime = record['regime']
                current_start_idx = i * 60
        
        # 마지막 국면 처리
        if current_regime is not None:
            regime_name = current_regime.value
            if regime_name not in regime_data:
                regime_data[regime_name] = []
            regime_data[regime_name].append(data.iloc[current_start_idx:])
        
        # 각 국면별 백테스팅
        for regime_name, data_chunks in regime_data.items():
            if not data_chunks:
                continue
                
            print(f"\n📊 {regime_name} 국면 백테스팅:")
            
            # 가장 긴 데이터 청크 선택
            longest_chunk = max(data_chunks, key=len)
            
            if len(longest_chunk) < 1000:
                print(f"   ⚠️  {regime_name} 국면 데이터 부족 (건너뜀)")
                continue
            
            # 국면별 전략 선택
            regime_enum = self._get_regime_enum(regime_name)
            strategy_name = self.market_analyzer.get_regime_strategy(regime_enum)
            leverage = self.market_analyzer.get_optimal_leverage(regime_enum)
            
            print(f"   🎯 선택된 전략: {strategy_name}")
            print(f"   ⚡ 최적 레버리지: {leverage}배")
            
            # 전략 실행
            self._run_strategy_backtest(longest_chunk, strategy_name, regime_name, leverage)
            
    def _get_regime_enum(self, regime_name: str) -> MarketRegime:
        """문자열을 MarketRegime enum으로 변환"""
        regime_map = {
            "급등": MarketRegime.RAPID_RISE,
            "상승": MarketRegime.BULL_MARKET,
            "횡보": MarketRegime.SIDEWAYS,
            "하락": MarketRegime.BEAR_MARKET,
            "급락": MarketRegime.CRASH
        }
        return regime_map.get(regime_name, MarketRegime.SIDEWAYS)
        
    def _run_strategy_backtest(self, data: pd.DataFrame, strategy_name: str, regime_name: str, leverage: float):
        """전략별 백테스팅"""
        initial_capital = 100_000_000  # 1억원
        
        if strategy_name == "momentum_breakout":
            strategy = AdvancedVolatilityMomentumStrategy(k_base=0.3, volume_weight=0.4)
        elif strategy_name == "trend_following":
            strategy = AdvancedVolatilityMomentumStrategy(k_base=0.5, volume_weight=0.3)
        elif strategy_name == "mean_reversion":
            strategy = AIMeanReversionStrategy(window=15, std_dev=1.5)
        elif strategy_name == "short_momentum":
            strategy = AdvancedVolatilityMomentumStrategy(k_base=0.4, volume_weight=0.5)
        elif strategy_name == "btc_short_only":
            strategy = AIMeanReversionStrategy(window=10, std_dev=1.0)
        else:
            strategy = AIMeanReversionStrategy(window=20, std_dev=2.0)
        
        # 레버리지 조정된 백테스팅
        backtest = AdvancedBacktestEngine(
            data=data.copy(),
            strategy=strategy,
            initial_capital=initial_capital,
            commission_rate=0.0005,
            slippage_rate=0.0002,
            max_position_size=0.1 * leverage,  # 레버리지 적용
            stop_loss_pct=0.02 / leverage if leverage > 0 else 0.02,  # 레버리지에 따른 스탑로스 조정
            take_profit_pct=0.05 * leverage if leverage > 0 else 0.05  # 레버리지에 따른 익절 조정
        )
        
        print(f"   🔄 {strategy.name} 백테스팅 실행 중...")
        backtest.run_backtest()
        
        # 결과 저장
        final_value = backtest.results['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_capital - 1) * 100
        
        self.trading_results.append({
            'regime': regime_name,
            'strategy': strategy.name,
            'leverage': leverage,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': backtest.risk_metrics.get('max_drawdown', 0) * 100,
            'sharpe_ratio': backtest.risk_metrics.get('sharpe_ratio', 0),
            'num_trades': len(backtest.trades[backtest.trades['type'] == 'BUY']) if not backtest.trades.empty else 0
        })
        
        print(f"   📈 결과: {total_return:.2f}% 수익률, {backtest.risk_metrics.get('sharpe_ratio', 0):.2f} 샤프 비율")
        
    def _analyze_final_results(self):
        """최종 결과 분석"""
        print("\n" + "=" * 80)
        print("📊 최종 결과 분석")
        print("=" * 80)
        
        if not self.trading_results:
            print("⚠️  백테스팅 결과가 없습니다.")
            return
        
        # 전체 성과 계산
        total_initial = sum(result['initial_capital'] for result in self.trading_results)
        total_final = sum(result['final_value'] for result in self.trading_results)
        overall_return = (total_final / total_initial - 1) * 100
        
        print(f"💰 전체 성과:")
        print(f"   - 총 초기 자본: {total_initial:,.0f}원")
        print(f"   - 총 최종 자산: {total_final:,.0f}원")
        print(f"   - 전체 수익률: {overall_return:.2f}%")
        
        # 국면별 성과 분석
        print(f"\n📊 국면별 성과:")
        regime_performance = {}
        
        for result in self.trading_results:
            regime = result['regime']
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result)
        
        for regime, results in regime_performance.items():
            avg_return = np.mean([r['total_return'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in results])
            
            print(f"   🎯 {regime} 국면:")
            print(f"      - 평균 수익률: {avg_return:.2f}%")
            print(f"      - 평균 샤프 비율: {avg_sharpe:.2f}")
            print(f"      - 평균 최대 낙폭: {avg_drawdown:.2f}%")
            print(f"      - 거래 횟수: {len(results)}회")
        
        # 최고 성과 전략
        best_result = max(self.trading_results, key=lambda x: x['total_return'])
        print(f"\n🏆 최고 성과:")
        print(f"   - 국면: {best_result['regime']}")
        print(f"   - 전략: {best_result['strategy']}")
        print(f"   - 레버리지: {best_result['leverage']}배")
        print(f"   - 수익률: {best_result['total_return']:.2f}%")
        print(f"   - 샤프 비율: {best_result['sharpe_ratio']:.2f}")
        
        # 리스크 분석
        worst_drawdown = max(self.trading_results, key=lambda x: x['max_drawdown'])
        print(f"\n⚠️  최고 리스크:")
        print(f"   - 국면: {worst_drawdown['regime']}")
        print(f"   - 최대 낙폭: {worst_drawdown['max_drawdown']:.2f}%")
        
        print(f"\n🎯 상위 0.01% 장중매매 시스템 분석 완료!")
        print("=" * 80)

def run_ultimate_system():
    """상위 0.01% 장중매매 시스템 실행"""
    
    # 데이터 로드
    data_file = 'data/historical_ohlcv/ADVANCED_CRYPTO_SAMPLE.csv'
    try:
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"✅ 고급 샘플 데이터 로드 완료: {len(data):,}개 데이터 포인트")
    except FileNotFoundError:
        print(f"❌ 오류: '{data_file}' 파일을 찾을 수 없습니다.")
        print("💡 먼저 'python create_advanced_sample_data.py'를 실행하여 고급 샘플 데이터를 생성해주세요.")
        return
    
    # 시스템 실행
    system = UltimateTradingSystem()
    system.run_complete_analysis(data)

if __name__ == '__main__':
    start_time = time.time()
    run_ultimate_system()
    end_time = time.time()
    print(f"\n⏱️  총 실행 시간: {end_time - start_time:.1f}초") 