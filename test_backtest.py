#!/usr/bin/env python3
"""
테스트 백테스트
"""

import pandas as pd
import numpy as np

# 간단한 데이터 생성
print("📊 데이터 생성 중...")
dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
prices = [50000]
for _ in range(999):
    change = np.random.normal(0, 0.01)
    prices.append(prices[-1] * (1 + change))

df = pd.DataFrame({
    'close': prices,
    'volume': np.random.randint(1000, 5000, 1000)
}, index=dates)

print(f"✅ {len(df)}개 캔들 생성 완료")

# 간단한 지표
df['ma_20'] = df['close'].rolling(20).mean()
df['rsi'] = 50 + np.random.normal(0, 15, len(df))  # 간단한 RSI 시뮬레이션

# 백테스트
print("💰 초기 자본: 10,000,000원")
print("📊 거래 시작...")

capital = 10000000
position = 0
entry_price = 0
trades = 0

for i, (timestamp, row) in enumerate(df.iterrows()):
    if i < 50:  # 처음 50개는 건너뛰기
        continue
        
    # 간단한 신호
    rsi = row['rsi']
    ma_20 = row['ma_20']
    
    signal = 0
    if rsi < 30 and row['close'] > ma_20:
        signal = 1
    elif rsi > 70 and row['close'] < ma_20:
        signal = -1
    
    # 거래 실행
    if position == 0 and signal != 0:
        position = signal
        entry_price = row['close']
        print(f"🎯 {'롱' if signal == 1 else '숏'} 진입: {entry_price:.0f}")
        trades += 1
        
    elif position != 0:
        current_price = row['close']
        
        # 손익 계산
        if position == 1:
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # 청산 조건
        if profit_pct >= 0.01 or profit_pct <= -0.005:  # 1% 익절 또는 0.5% 손절
            # 수수료 차감
            profit_pct -= 0.0006
            
            # 손익 계산
            pnl = capital * 0.1 * profit_pct
            capital += pnl
            
            reason = "익절" if profit_pct > 0 else "손절"
            print(f"   {reason}: {current_price:.0f} | 손익: {profit_pct:.2%} ({pnl:,.0f}원)")
            
            position = 0

# 결과
total_return = (capital - 10000000) / 10000000
print(f"\n{'='*40}")
print(f"🎉 백테스트 결과")
print(f"{'='*40}")
print(f"💰 최종 자본: {capital:,.0f}원")
print(f"📈 총 수익률: {total_return:.2%}")
print(f"🎯 총 거래 수: {trades}건")
print(f"{'='*40}") 