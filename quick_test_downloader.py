#!/usr/bin/env python3
"""
빠른 테스트용 다운로더 - 확실히 작동하는 버전
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def quick_test():
    print("🔧 빠른 테스트 다운로더")
    print("=" * 40)
    
    # 바이낸스 초기화
    exchange = ccxt.binance()
    
    try:
        # 최근 100개 1시간봉 데이터 가져오기 (확실히 존재하는 데이터)
        print("📊 BTC/USDT 1시간봉 최근 100개 다운로드 중...")
        
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 저장
            Path("data/market_data").mkdir(parents=True, exist_ok=True)
            df.to_csv("data/market_data/BTC_USDT_1h_quick.csv")
            
            print(f"✅ 성공! {len(df)}개 레코드 다운로드")
            print(f"📅 기간: {df.index[0]} ~ {df.index[-1]}")
            print(f"💾 저장 경로: data/market_data/BTC_USDT_1h_quick.csv")
            
    except Exception as e:
        print(f"❌ 다운로드 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test() 