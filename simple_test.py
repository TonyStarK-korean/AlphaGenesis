#!/usr/bin/env python3
"""
간단한 테스트 버전 - 확실히 작동하는 최소 구현
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def simple_test():
    print("🚀 간단한 테스트 시작")
    
    # 1. 기본 데이터 생성
    print("📊 데이터 생성 중...")
    dates = pd.date_range('2025-06-01', periods=100, freq='h')
    
    np.random.seed(42)
    prices = [65000]
    for i in range(99):
        change = np.random.normal(0, 0.02)
        price = prices[-1] * (1 + change)
        prices.append(max(price, 10000))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.uniform(1000, 3000, 100)
    })
    
    print(f"✅ 데이터 생성 완료: {len(df)}개")
    
    # 2. 간단한 지표
    print("📈 지표 계산 중...")
    df['ma_5'] = df['close'].rolling(5, min_periods=1).mean()
    df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['rsi'] = 50 + np.random.normal(0, 10, len(df))  # 간단한 RSI
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10, min_periods=1).mean()
    
    print("✅ 지표 계산 완료")
    
    # 3. 간단한 예측
    print("🤖 예측 모델 실행 중...")
    predictions = []
    for i in range(len(df)):
        if i < 10:
            predictions.append(0.0)
        else:
            # 단순한 모멘텀 기반 예측
            momentum = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
            ma_signal = (df['ma_5'].iloc[i] - df['ma_20'].iloc[i]) / df['ma_20'].iloc[i]
            pred = (momentum + ma_signal) * 2  # 신호 증폭
            predictions.append(max(min(pred, 0.05), -0.05))
    
    strong_preds = [p for p in predictions if abs(p) > 0.005]
    print(f"✅ 예측 완료: 강한 신호 {len(strong_preds)}개")
    
    # 4. 간단한 백테스트
    print("💰 백테스트 실행 중...")
    
    capital = 10000000
    position = 0
    trades = 0
    
    for i in range(len(df)):
        pred = predictions[i]
        price = df['close'].iloc[i]
        
        # 간단한 매매 조건
        if abs(pred) > 0.008:  # 0.8% 이상 예측
            if pred > 0 and position <= 0:  # 매수
                position = 1
                trades += 1
                print(f"   매수: {price:.0f}원 (예측: {pred:+.3f})")
            elif pred < 0 and position >= 0:  # 매도
                position = -1
                trades += 1
                print(f"   매도: {price:.0f}원 (예측: {pred:+.3f})")
    
    print(f"✅ 백테스트 완료: 총 {trades}개 거래")
    
    # 5. 결과
    final_price = df['close'].iloc[-1]
    initial_price = df['close'].iloc[0]
    market_return = (final_price - initial_price) / initial_price * 100
    
    print(f"\n📊 결과:")
    print(f"   시장 수익률: {market_return:+.2f}%")
    print(f"   총 거래 수: {trades}개")
    print(f"   데이터 포인트: {len(df)}개")
    print(f"   강한 예측 신호: {len(strong_preds)}개")
    
    if trades > 0:
        print("✅ 거래 신호가 정상적으로 생성되었습니다!")
    else:
        print("❌ 거래 신호가 생성되지 않았습니다.")
    
    print("🎉 테스트 완료!")

if __name__ == "__main__":
    simple_test()