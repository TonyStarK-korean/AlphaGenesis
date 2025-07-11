#!/usr/bin/env python3
"""
라이브러리 의존성 없는 간단한 백테스트 테스트
"""

import sys
import os
import random
import math
from datetime import datetime, timedelta

def generate_simple_data(days=30):
    """간단한 가격 데이터 생성"""
    print("📊 시뮬레이션 데이터 생성 중...")
    
    data = []
    base_price = 50000  # 초기 가격 (BTC 기준)
    
    for i in range(days * 24):  # 1시간 단위
        # 랜덤 워크 생성
        change = random.uniform(-0.02, 0.02)  # -2% ~ +2%
        base_price *= (1 + change)
        
        # OHLC 데이터 생성
        high = base_price * (1 + random.uniform(0, 0.01))
        low = base_price * (1 - random.uniform(0, 0.01))
        close = base_price
        volume = random.uniform(100, 1000)
        
        data.append({
            'timestamp': datetime.now() - timedelta(hours=days*24-i),
            'open': base_price,
            'high': high,
            'low': low, 
            'close': close,
            'volume': volume
        })
    
    print(f"✅ {len(data)}개 데이터 포인트 생성 완료")
    return data

def simple_moving_average(data, window):
    """간단한 이동평균 계산"""
    if len(data) < window:
        return [data[-1]['close']] * len(data)
    
    ma_values = []
    for i in range(len(data)):
        if i < window - 1:
            ma_values.append(data[i]['close'])
        else:
            avg = sum(data[j]['close'] for j in range(i - window + 1, i + 1)) / window
            ma_values.append(avg)
    
    return ma_values

def simple_strategy_backtest(data, initial_capital=10000000):
    """간단한 이동평균 교차 전략"""
    print("🚀 간단한 백테스트 실행 중...")
    
    capital = initial_capital
    position = 0  # 0: 현금, 1: 매수
    entry_price = 0
    trades = []
    
    # 이동평균 계산
    ma_short = simple_moving_average(data, 5)   # 5시간 이동평균
    ma_long = simple_moving_average(data, 20)   # 20시간 이동평균
    
    for i in range(20, len(data)):  # 20개부터 시작
        current_price = data[i]['close']
        
        # 매수 신호: 단기 이평이 장기 이평을 상향 돌파
        if position == 0 and ma_short[i] > ma_long[i] and ma_short[i-1] <= ma_long[i-1]:
            position = 1
            entry_price = current_price
            shares = capital / current_price
            print(f"📈 매수: {current_price:,.0f}원, 수량: {shares:.4f}")
            
        # 매도 신호: 단기 이평이 장기 이평을 하향 돌파
        elif position == 1 and ma_short[i] < ma_long[i] and ma_short[i-1] >= ma_long[i-1]:
            position = 0
            exit_price = current_price
            new_capital = shares * exit_price
            profit = new_capital - capital
            profit_pct = (profit / capital) * 100
            
            trades.append({
                'entry': entry_price,
                'exit': exit_price,
                'profit': profit,
                'profit_pct': profit_pct
            })
            
            capital = new_capital
            print(f"📉 매도: {exit_price:,.0f}원, 수익: {profit:,.0f}원 ({profit_pct:.2f}%)")
    
    # 마지막 포지션 정리
    if position == 1:
        final_price = data[-1]['close']
        final_capital = shares * final_price
        profit = final_capital - initial_capital
        profit_pct = (profit / initial_capital) * 100
        
        trades.append({
            'entry': entry_price,
            'exit': final_price,
            'profit': profit,
            'profit_pct': profit_pct
        })
        capital = final_capital
    
    return capital, trades

def analyze_results(initial_capital, final_capital, trades):
    """결과 분석"""
    print("\n" + "="*60)
    print("📊 백테스트 결과 분석")
    print("="*60)
    
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    print(f"💰 초기 자본: {initial_capital:,.0f}원")
    print(f"💰 최종 자본: {final_capital:,.0f}원")
    print(f"📈 총 수익률: {total_return:.2f}%")
    print(f"💵 절대 수익: {final_capital - initial_capital:,.0f}원")
    
    if trades:
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        print(f"\n📊 거래 통계:")
        print(f"   총 거래 횟수: {len(trades)}회")
        print(f"   승리 거래: {len(winning_trades)}회")
        print(f"   패배 거래: {len(losing_trades)}회")
        print(f"   승률: {win_rate:.1f}%")
        print(f"   평균 수익: {avg_win:,.0f}원")
        print(f"   평균 손실: {avg_loss:,.0f}원")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"   수익 팩터: {profit_factor:.2f}")
    
    # 성과 등급
    if total_return > 30:
        grade = "A+"
    elif total_return > 20:
        grade = "A"
    elif total_return > 10:
        grade = "B"
    elif total_return > 0:
        grade = "C"
    else:
        grade = "D"
    
    print(f"\n🏆 성과 등급: {grade}")
    
    if total_return > 0:
        print("✅ 수익성 있는 전략입니다!")
    else:
        print("❌ 손실이 발생했습니다. 전략 개선이 필요합니다.")
    
    print("="*60)

def main():
    """메인 실행 함수"""
    print("🚀 AlphaGenesis 간단한 백테스트 시작")
    print("📊 Python 기본 라이브러리만 사용")
    print()
    
    # 명령행 인수 처리
    initial_capital = 10000000  # 기본값
    if len(sys.argv) > 1:
        try:
            if '--initial-capital' in sys.argv:
                idx = sys.argv.index('--initial-capital')
                if idx + 1 < len(sys.argv):
                    initial_capital = int(sys.argv[idx + 1])
                    print(f"💰 초기 자본 설정: {initial_capital:,.0f}원")
        except (ValueError, IndexError):
            print("⚠️  잘못된 자본 설정, 기본값 사용")
    
    # 데이터 생성
    data = generate_simple_data(30)  # 30일치 데이터
    
    # 백테스트 실행
    final_capital, trades = simple_strategy_backtest(data, initial_capital)
    
    # 결과 분석
    analyze_results(initial_capital, final_capital, trades)
    
    print("\n💡 실제 고급 백테스트를 위해서는 pandas, numpy 등의 라이브러리가 필요합니다.")
    print("   라이브러리 설치 후 run_ml_backtest.py를 실행해보세요!")

if __name__ == "__main__":
    main()