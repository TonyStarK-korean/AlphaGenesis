#!/usr/bin/env python3
"""
간단한 로컬 백테스트 (웹대시보드 없음)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rsi(prices, period=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_simple_backtest():
    """간단한 백테스트 실행"""
    
    print("🏠 간단한 로컬 백테스트")
    print("=" * 50)
    
    # 데이터 경로
    data_path = Path("data/market_data")
    
    # 사용 가능한 파일 확인
    csv_files = list(data_path.glob("*.csv"))
    csv_files = [f for f in csv_files if "data_generator" not in f.name]
    
    if not csv_files:
        print("❌ 데이터 파일을 찾을 수 없습니다!")
        return
    
    print(f"✅ {len(csv_files)}개 데이터 파일 발견")
    
    # BTC 1시간 데이터 우선 사용
    btc_file = None
    for file in csv_files:
        if "BTC_USDT_1h" in file.name:
            btc_file = file
            break
    
    if not btc_file:
        btc_file = csv_files[0]  # 첫 번째 파일 사용
    
    print(f"📊 사용할 데이터: {btc_file.name}")
    
    # 데이터 로드
    try:
        df = pd.read_csv(btc_file)
        
        # timestamp 컬럼 처리
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
        
        df = df.sort_index()
        print(f"✅ 데이터 로드 완료: {len(df)}개 레코드")
        print(f"📅 기간: {df.index[0]} ~ {df.index[-1]}")
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {e}")
        return
    
    # 기술적 지표 계산
    print("🤖 기술적 지표 계산 중...")
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # 매매 신호 생성
    df['signal'] = 0
    df.loc[(df['ma_20'] > df['ma_50']) & (df['rsi'] < 70), 'signal'] = 1  # 매수
    df.loc[(df['ma_20'] < df['ma_50']) | (df['rsi'] > 80), 'signal'] = -1  # 매도
    
    buy_signals = len(df[df['signal'] == 1])
    sell_signals = len(df[df['signal'] == -1])
    print(f"✅ 신호 생성 완료: 매수 {buy_signals}개, 매도 {sell_signals}개")
    
    # 백테스트 실행
    print("🚀 백테스트 실행 중...")
    
    initial_capital = 10000000  # 1천만원
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    commission = 0.001  # 0.1%
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        if i < 50:  # 지표 계산을 위해 스킵
            continue
        
        current_price = row['close']
        signal = row['signal']
        
        # 진행률 표시
        if i % (len(df) // 10) == 0:
            progress = (i / len(df)) * 100
            print(f"⏱️ 진행률: {progress:.1f}%")
        
        # 매수 신호
        if position == 0 and signal == 1:
            position = (capital * 0.95) / current_price
            entry_price = current_price
            capital -= position * current_price * (1 + commission)
            
            trades.append({
                'type': 'BUY',
                'price': current_price,
                'time': timestamp,
                'amount': position
            })
            
            print(f"📈 매수: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')}")
        
        # 매도 신호
        elif position > 0 and signal == -1:
            sell_value = position * current_price * (1 - commission)
            capital += sell_value
            
            pnl = (current_price - entry_price) / entry_price * 100
            
            trades.append({
                'type': 'SELL',
                'price': current_price,
                'time': timestamp,
                'amount': position,
                'pnl': pnl
            })
            
            pnl_symbol = "💰" if pnl > 0 else "💸"
            print(f"📉 매도: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')} {pnl_symbol} {pnl:+.2f}%")
            
            position = 0
    
    # 마지막 포지션 청산
    if position > 0:
        final_value = position * df['close'].iloc[-1]
        capital += final_value
        print(f"🔄 최종 청산: ${df['close'].iloc[-1]:,.2f}")
    
    # 결과 계산
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # 승률 계산
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    total_trades_with_pnl = [t for t in trades if 'pnl' in t]
    win_rate = (len(winning_trades) / len(total_trades_with_pnl)) * 100 if total_trades_with_pnl else 0
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎉 백테스트 완료!")
    print(f"💰 초기 자본: ₩{initial_capital:,}")
    print(f"💰 최종 자본: ₩{final_capital:,.0f}")
    print(f"📈 총 수익률: {total_return:+.2f}%")
    print(f"🎯 승률: {win_rate:.1f}%")
    print(f"🔄 총 거래: {len(trades)}회")
    print(f"✅ 수익 거래: {len(winning_trades)}회")
    print("="*60)
    
    # 최근 거래 내역 표시
    if trades:
        print("\n📋 최근 거래 내역 (마지막 5개):")
        for trade in trades[-5:]:
            if trade['type'] == 'BUY':
                print(f"  📈 {trade['time'].strftime('%m-%d %H:%M')} 매수 ${trade['price']:,.2f}")
            else:
                pnl_symbol = "💰" if trade.get('pnl', 0) > 0 else "💸"
                print(f"  📉 {trade['time'].strftime('%m-%d %H:%M')} 매도 ${trade['price']:,.2f} {pnl_symbol} {trade.get('pnl', 0):+.2f}%")

if __name__ == "__main__":
    run_simple_backtest() 