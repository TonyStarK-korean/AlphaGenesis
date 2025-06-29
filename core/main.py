import pandas as pd
import sys
import os

# src 폴더를 파이썬 경로에 추가 (모듈을 찾을 수 있도록)
# 이 스크립트가 src 내부에 있으므로 상위 폴더(프로젝트 루트)를 경로에 추가해야 함
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.strategy_manager.volatility_momentum import VolatilityMomentumStrategy
from src.strategy_manager.mean_reversion import MeanReversionStrategy
from tests.backtest_engine import BacktestEngine

def run():
    data_file = 'data/historical_ohlcv/SAMPLE_STOCK.csv'
    try:
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"오류: '{data_file}' 파일을 찾을 수 없습니다.")
        print("먼저 'python create_sample_data.py'를 실행하여 샘플 데이터를 생성해주세요.")
        return

    initial_capital = 100_000_000

    print("\n[시나리오 1: 변동성 돌파 전략 백테스팅]")
    vol_strategy = VolatilityMomentumStrategy(k=0.5)
    backtest_vol = BacktestEngine(data.copy(), vol_strategy, initial_capital)
    backtest_vol.run_backtest()
    backtest_vol.generate_report()

    print("\n\n[시나리오 2: 평균 회귀 전략 백테스팅]")
    mean_rev_data = data.copy().dropna() 
    mean_rev_strategy = MeanReversionStrategy(window=10, std_dev=1.0)
    backtest_mr = BacktestEngine(mean_rev_data, mean_rev_strategy, initial_capital)
    backtest_mr.run_backtest()
    backtest_mr.generate_report()

if __name__ == '__main__':
    run()