import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 파이썬 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.strategy_manager.advanced_volatility_momentum import AdvancedVolatilityMomentumStrategy
from src.strategy_manager.ai_mean_reversion import AIMeanReversionStrategy
from tests.advanced_backtest_engine import AdvancedBacktestEngine

def run_advanced_backtest():
    """코인선물 전세계 상위 0.1% 고급 백테스트 실행"""
    
    print("🚀 코인선물 전세계 상위 0.1% 전략급 백테스팅 시스템")
    print("=" * 60)
    
    # 데이터 로드
    data_file = 'data/historical_ohlcv/ADVANCED_CRYPTO_SAMPLE.csv'
    try:
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"✅ 고급 샘플 데이터 로드 완료: {len(data):,}개 데이터 포인트")
    except FileNotFoundError:
        print(f"❌ 오류: '{data_file}' 파일을 찾을 수 없습니다.")
        print("💡 먼저 'python create_advanced_sample_data.py'를 실행하여 고급 샘플 데이터를 생성해주세요.")
        return
    
    # 백테스트 설정
    initial_capital = 100_000_000  # 1억원
    print(f"💰 초기 자본: {initial_capital:,.0f}원")
    print(f"📊 데이터 기간: {data.index[0]} ~ {data.index[-1]}")
    print(f"📈 시작가: ${data['Open'].iloc[0]:,.2f} | 종가: ${data['Close'].iloc[-1]:,.2f}")
    print(f"🔄 총 수익률: {(data['Close'].iloc[-1] / data['Open'].iloc[0] - 1) * 100:.2f}%")
    print("=" * 60)
    
    # 시나리오 1: 고급 변동성 돌파 전략
    print("\n🔥 [시나리오 1: 고급 변동성 돌파 전략]")
    print("-" * 50)
    
    vol_strategy = AdvancedVolatilityMomentumStrategy(
        k_base=0.5,
        volume_weight=0.3,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )
    
    backtest_vol = AdvancedBacktestEngine(
        data=data.copy(),
        strategy=vol_strategy,
        initial_capital=initial_capital,
        commission_rate=0.0005,  # 0.05% 수수료
        slippage_rate=0.0002,    # 0.02% 슬리피지
        max_position_size=0.1,   # 최대 10% 포지션
        stop_loss_pct=0.02,      # 2% 스탑로스
        take_profit_pct=0.05     # 5% 익절
    )
    
    print("🔄 고급 변동성 돌파 전략 백테스팅 실행 중...")
    backtest_vol.run_backtest()
    backtest_vol.generate_report()
    
    # 시나리오 2: AI 기반 평균 회귀 전략
    print("\n🤖 [시나리오 2: AI 기반 평균 회귀 전략]")
    print("-" * 50)
    
    ai_strategy = AIMeanReversionStrategy(
        window=20,
        std_dev=2.0,
        rsi_period=14,
        stoch_k=14,
        stoch_d=3,
        ml_lookback=50
    )
    
    backtest_ai = AdvancedBacktestEngine(
        data=data.copy(),
        strategy=ai_strategy,
        initial_capital=initial_capital,
        commission_rate=0.0005,
        slippage_rate=0.0002,
        max_position_size=0.08,  # AI 전략은 더 보수적
        stop_loss_pct=0.015,     # 1.5% 스탑로스
        take_profit_pct=0.04     # 4% 익절
    )
    
    print("🔄 AI 기반 평균 회귀 전략 백테스팅 실행 중...")
    backtest_ai.run_backtest()
    backtest_ai.generate_report()
    
    # 전략 비교 분석
    print("\n📊 [전략 비교 분석]")
    print("=" * 60)
    
    vol_final = backtest_vol.results['portfolio_value'].iloc[-1]
    ai_final = backtest_ai.results['portfolio_value'].iloc[-1]
    
    vol_return = (vol_final / initial_capital - 1) * 100
    ai_return = (ai_final / initial_capital - 1) * 100
    
    vol_trades = len(backtest_vol.trades[backtest_vol.trades['type'] == 'BUY']) if not backtest_vol.trades.empty else 0
    ai_trades = len(backtest_ai.trades[backtest_ai.trades['type'] == 'BUY']) if not backtest_ai.trades.empty else 0
    
    print(f"📈 변동성 돌파 전략:")
    print(f"   - 최종 자산: {vol_final:,.0f}원")
    print(f"   - 수익률: {vol_return:.2f}%")
    print(f"   - 거래 횟수: {vol_trades}회")
    print(f"   - 샤프 비율: {backtest_vol.risk_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   - 최대 낙폭: {backtest_vol.risk_metrics.get('max_drawdown', 0)*100:.2f}%")
    
    print(f"\n🤖 AI 평균 회귀 전략:")
    print(f"   - 최종 자산: {ai_final:,.0f}원")
    print(f"   - 수익률: {ai_return:.2f}%")
    print(f"   - 거래 횟수: {ai_trades}회")
    print(f"   - 샤프 비율: {backtest_ai.risk_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   - 최대 낙폭: {backtest_ai.risk_metrics.get('max_drawdown', 0)*100:.2f}%")
    
    # 승자 결정
    if vol_return > ai_return:
        winner = "변동성 돌파 전략"
        winner_return = vol_return
    elif ai_return > vol_return:
        winner = "AI 평균 회귀 전략"
        winner_return = ai_return
    else:
        winner = "동점"
        winner_return = vol_return
    
    print(f"\n🏆 승자: {winner}")
    if winner != "동점":
        print(f"   - 수익률: {winner_return:.2f}%")
    
    print("\n🎯 백테스팅 완료! 고급 차트 파일들을 확인해보세요.")
    print("=" * 60)

if __name__ == '__main__':
    run_advanced_backtest() 