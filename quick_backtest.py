#!/usr/bin/env python3
"""
빠른 백테스트 실행 파일
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    """간단한 데이터 생성"""
    print("📊 데이터 생성 중...")
    
    # 1년치 시간 인덱스
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # 가격 시뮬레이션
    np.random.seed(42)
    prices = [50000.0]
    
    for _ in range(len(date_range) - 1):
        change = np.random.normal(0, 0.015)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLCV 데이터
    data = []
    for i, timestamp in enumerate(date_range):
        price = prices[i]
        high = price * (1 + np.random.uniform(0, 0.02))
        low = price * (1 - np.random.uniform(0, 0.02))
        volume = 1000 + np.random.exponential(2000)
        
        data.append({
            'timestamp': timestamp,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ {len(df)}개 캔들 생성 완료")
    return df

def add_indicators(df):
    """기본 지표 추가"""
    # 이동평균
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def run_backtest(df, capital=10000000):
    """백테스트 실행"""
    print(f"💰 초기 자본: {capital:,.0f}원")
    
    df = add_indicators(df)
    df = df.dropna()
    
    position = 0
    entry_price = 0
    trades = []
    
    print("📊 거래 시작...")
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        # 매 2000개마다 진행률 표시
        if i % 2000 == 0 and i > 0:
            progress = (i / len(df)) * 100
            print(f"   진행률: {progress:.1f}% | 자본: {capital:,.0f}원")
        
        # 신호 생성
        rsi = row['rsi']
        ma_20 = row['ma_20']
        ma_50 = row['ma_50']
        
        signal = 0
        if rsi < 30 and ma_20 > ma_50:  # 과매도 + 상승추세
            signal = 1
        elif rsi > 70 and ma_20 < ma_50:  # 과매수 + 하락추세
            signal = -1
        
        # 거래 실행
        if position == 0 and signal != 0:  # 진입
            position = signal
            entry_price = row['close']
            print(f"🎯 {'롱' if signal == 1 else '숏'} 진입: {entry_price:.0f}")
            
        elif position != 0:  # 청산 조건 확인
            current_price = row['close']
            
            # 손익 계산
            if position == 1:  # 롱
                profit_pct = (current_price - entry_price) / entry_price
            else:  # 숏
                profit_pct = (entry_price - current_price) / entry_price
            
            # 청산 조건
            should_close = False
            reason = ""
            
            if profit_pct >= 0.015:  # 1.5% 익절
                should_close = True
                reason = "익절"
            elif profit_pct <= -0.01:  # 1% 손절
                should_close = True
                reason = "손절"
            elif (position == 1 and signal == -1) or (position == -1 and signal == 1):  # 신호 전환
                should_close = True
                reason = "신호전환"
            
            if should_close:
                # 수수료 차감
                profit_pct -= 0.0006
                
                # 손익 계산
                position_size = capital * 0.1
                pnl = position_size * profit_pct
                capital += pnl
                
                print(f"   {reason}: {current_price:.0f} | 손익: {profit_pct:.2%} ({pnl:,.0f}원)")
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': profit_pct,
                    'pnl': pnl,
                    'reason': reason
                })
                
                position = 0
    
    # 결과 계산
    total_return = (capital - 10000000) / 10000000
    total_trades = len(trades)
    
    if total_trades > 0:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = wins / total_trades
    else:
        win_rate = 0
    
    return {
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'trades': trades
    }

def print_results(results):
    """결과 출력"""
    print(f"\n{'='*50}")
    print(f"🎉 백테스트 결과")
    print(f"{'='*50}")
    
    print(f"💰 최종 자본: {results['final_capital']:,.0f}원")
    print(f"📈 총 수익률: {results['total_return']:.2%}")
    print(f"🎯 총 거래 수: {results['total_trades']}건")
    print(f"📊 승률: {results['win_rate']:.2%}")
    
    # 성과 평가
    if results['total_return'] > 0.05:
        grade = "🏆 우수"
    elif results['total_return'] > 0.02:
        grade = "👍 양호"
    elif results['total_return'] > 0:
        grade = "📈 플러스"
    else:
        grade = "📉 손실"
    
    print(f"🏆 성과 등급: {grade}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("🚀 빠른 백테스트 시작!")
    
    # 데이터 생성
    df = generate_data()
    
    # 백테스트 실행
    results = run_backtest(df)
    
    # 결과 출력
    print_results(results) 