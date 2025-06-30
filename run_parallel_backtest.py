import sys
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from run_ml_backtest import run_ml_backtest, generate_historical_data
import json

def run_single_symbol(symbol):
    print(f"{symbol} 데이터 생성 및 백테스트 시작")
    # 3년치 히스토리컬 데이터 생성 (심볼 정보는 데이터에 추가)
    df = generate_historical_data(years=3)
    df['symbol'] = symbol  # 심볼 정보 추가
    results = run_ml_backtest(df, initial_capital=10000000)
    # 결과를 파일로 저장
    out_path = f"results_{symbol.replace('/', '_')}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{symbol} 결과 저장: {out_path}")
    return out_path

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']  # 전체 심볼 리스트로 교체
    with Pool(cpu_count()) as pool:
        all_results = pool.map(run_single_symbol, symbols)
    print("모든 종목 백테스트 완료! 결과 파일:", all_results) 