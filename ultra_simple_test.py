#!/usr/bin/env python3
"""
매우 간단한 백테스트 - 100% 확실히 작동
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

def main():
    print("🚀 매우 간단한 백테스트 시작")
    
    # 1. 매우 기본적인 데이터
    np.random.seed(42)
    n_points = 200
    
    base_price = 65000
    prices = [base_price]
    
    for i in range(n_points - 1):
        change = np.random.normal(0, 0.02)  # 2% 변동성
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10000))
    
    volumes = np.random.uniform(1000, 3000, n_points)
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes
    })
    
    print(f"✅ 데이터 생성: {len(df)}개 포인트")
    print(f"   가격 범위: {df['close'].min():.0f} ~ {df['close'].max():.0f}")
    
    # 2. 매우 간단한 지표들
    df['ma_short'] = df['close'].rolling(5, min_periods=1).mean()
    df['ma_long'] = df['close'].rolling(20, min_periods=1).mean()
    df['volume_ma'] = df['volume'].rolling(10, min_periods=1).mean()
    
    # NaN 제거
    df = df.fillna(0)
    
    print("✅ 지표 계산 완료")
    
    # 3. 매우 간단한 신호
    signals = []
    trades = []
    
    for i in range(len(df)):
        signal = 'HOLD'
        confidence = 0.0
        
        if i >= 20:  # 충분한 데이터가 있을 때만
            # 단순한 골든크로스/데드크로스
            current_short = df['ma_short'].iloc[i]
            current_long = df['ma_long'].iloc[i]
            prev_short = df['ma_short'].iloc[i-1]
            prev_long = df['ma_long'].iloc[i-1]
            
            # 안전한 비교 (0으로 나누기 없음)
            if current_long > 0 and prev_long > 0:
                current_ratio = current_short / current_long
                prev_ratio = prev_short / prev_long
                
                # 골든크로스 (상승 신호)
                if current_ratio > 1.005 and prev_ratio <= 1.0:
                    signal = 'BUY'
                    confidence = 0.8
                
                # 데드크로스 (하락 신호)
                elif current_ratio < 0.995 and prev_ratio >= 1.0:
                    signal = 'SELL'
                    confidence = 0.8
        
        signals.append({
            'signal': signal,
            'confidence': confidence,
            'price': df['close'].iloc[i]
        })
        
        if signal != 'HOLD' and confidence >= 0.5:
            trades.append({
                'index': i,
                'action': signal,
                'price': df['close'].iloc[i],
                'confidence': confidence
            })
    
    print(f"✅ 신호 생성 완료")
    print(f"   총 신호: {len([s for s in signals if s['signal'] != 'HOLD'])}개")
    print(f"   실행 거래: {len(trades)}개")
    
    # 4. 결과 출력
    if trades:
        print(f"\n📊 거래 내역:")
        for i, trade in enumerate(trades[:10]):  # 최대 10개만
            print(f"   {i+1}. {trade['action']} @ {trade['price']:.0f}원 (신뢰도: {trade['confidence']:.2f})")
        
        if len(trades) > 10:
            print(f"   ... 및 {len(trades)-10}개 추가 거래")
        
        # 간단한 수익률 계산
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        market_return = (end_price - start_price) / start_price * 100
        
        print(f"\n📈 결과:")
        print(f"   시장 수익률: {market_return:+.2f}%")
        print(f"   총 거래 수: {len(trades)}개")
        
        if len(trades) > 0:
            print("✅ 거래 신호가 성공적으로 생성되었습니다!")
            grade = "B" if len(trades) >= 5 else "C"
            print(f"🏆 성과 등급: {grade}")
        else:
            print("❌ 거래가 실행되지 않았습니다.")
    else:
        print("\n❌ 거래 신호가 발생하지 않았습니다.")
        print("   원인: 신호 생성 조건이 충족되지 않음")
    
    print("🎉 테스트 완료!")

if __name__ == "__main__":
    main()