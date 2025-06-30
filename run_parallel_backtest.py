import sys
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from run_ml_backtest import run_ml_backtest, generate_historical_data
import json

def run_single_symbol(symbol):
    print(f"{symbol} 데이터 생성 및 백테스트 시작")
    # 심볼별로 다른 시드값을 사용하여 다양한 데이터 생성
    import numpy as np
    seed_map = {
        'BTC/USDT': 42,
        'ETH/USDT': 123,
        'XRP/USDT': 456,
        'BNB/USDT': 789,
        'ADA/USDT': 101,
        'DOT/USDT': 202
    }
    np.random.seed(seed_map.get(symbol, 42))
    
    # 심볼별 특성을 반영한 데이터 생성
    df = generate_symbol_specific_data(symbol, years=3)
    df['symbol'] = symbol
    results = run_ml_backtest(df, initial_capital=10000000)
    
    # 결과를 파일로 저장
    out_path = f"results_{symbol.replace('/', '_')}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        # results가 직렬화 가능하도록 변환
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f"{symbol} 결과 저장: {out_path}")
    return out_path

def generate_symbol_specific_data(symbol: str, years: int = 3) -> pd.DataFrame:
    """심볼별 특성을 반영한 데이터 생성"""
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd
    
    # 심볼별 특성 정의
    symbol_params = {
        'BTC/USDT': {'base_price': 50000, 'volatility': 0.03, 'trend': 0.0002},
        'ETH/USDT': {'base_price': 3000, 'volatility': 0.04, 'trend': 0.0003},
        'XRP/USDT': {'base_price': 0.5, 'volatility': 0.05, 'trend': 0.0001},
        'BNB/USDT': {'base_price': 400, 'volatility': 0.035, 'trend': 0.0002},
        'ADA/USDT': {'base_price': 1.0, 'volatility': 0.045, 'trend': 0.0001},
        'DOT/USDT': {'base_price': 15, 'volatility': 0.04, 'trend': 0.0002}
    }
    
    params = symbol_params.get(symbol, symbol_params['BTC/USDT'])
    
    # 시간 범위 설정
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    time_delta = timedelta(hours=1)
    
    current_date = start_date
    data = []
    current_price = params['base_price']
    
    while current_date <= end_date:
        # 트렌드 + 랜덤 변동
        price_change = np.random.normal(params['trend'], params['volatility'])
        current_price = max(current_price * (1 + price_change), params['base_price'] * 0.1)
        
        # OHLCV 데이터 생성
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        high_price = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = current_price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = current_price
        volume = max(int(np.random.normal(10000, 5000)), 1000)
        
        data.append({
            'timestamp': current_date,
            'open': abs(open_price),
            'high': abs(high_price),
            'low': abs(low_price),
            'close': abs(close_price),
            'volume': volume
        })
        
        current_date += time_delta
    
    df = pd.DataFrame(data)
    print(f"{symbol} 데이터 생성 완료: {len(df)}개 (가격범위: {df['close'].min():.2f} ~ {df['close'].max():.2f})")
    return df

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']  
    print(f"=== 병렬 백테스트 시작 ===")
    print(f"대상 종목: {', '.join(symbols)}")
    print(f"프로세서 수: {cpu_count()}개")
    print(f"예상 소요시간: 5-10분")
    print("=" * 50)
    
    with Pool(cpu_count()) as pool:
        all_results = pool.map(run_single_symbol, symbols)
    
    print("\n" + "=" * 50)
    print("🎉 모든 종목 백테스트 완료!")
    print(f"📊 처리된 종목: {len(symbols)}개")
    print(f"📁 생성된 결과 파일:")
    for i, result_file in enumerate(all_results, 1):
        symbol = symbols[i-1]
        print(f"  {i}. {symbol}: {result_file}")
    print("=" * 50) 