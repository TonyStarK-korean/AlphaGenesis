#!/usr/bin/env python3
"""
간단한 백테스트 실행 파일
scikit-learn과 optuna 없이도 실행 가능
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

warnings.filterwarnings('ignore')

def generate_historical_data(years: int = 3) -> pd.DataFrame:
    """히스토리컬 데이터 생성"""
    print(f"📊 {years}년치 시뮬레이션 데이터 생성 중...")
    
    # 시간 인덱스 생성
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # 비트코인 가격 시뮬레이션
    np.random.seed(42)
    initial_price = 50000.0
    
    prices = []
    current_price = initial_price
    
    for _ in range(len(date_range)):
        # 가격 변화 (랜덤 워크)
        change = np.random.normal(0, 0.02)  # 2% 변동성
        current_price *= (1 + change)
        prices.append(current_price)
    
    # OHLCV 데이터 생성
    data = []
    for i, timestamp in enumerate(date_range):
        base_price = prices[i]
        
        # 변동성 생성
        volatility = np.random.uniform(0.005, 0.03)
        high_offset = np.random.uniform(0, volatility)
        low_offset = np.random.uniform(0, volatility)
        
        high = base_price * (1 + high_offset)
        low = base_price * (1 - low_offset)
        
        # 시가와 종가 생성
        if i == 0:
            open_price = base_price
        else:
            open_price = data[-1]['close']
        
        close_price = base_price
        
        # 거래량 생성
        volume = 1000 + np.random.exponential(2000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(open_price, high, close_price),
            'low': min(open_price, low, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"   ✅ 생성 완료: {len(df)}개 캔들")
    print(f"   📊 가격 범위: {df['close'].min():.0f} ~ {df['close'].max():.0f}")
    
    return df

def make_features(df):
    """기본 기술적 지표 생성"""
    # 이동평균
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # 변동성
    df['volatility_20'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    # 거래량 지표
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 수익률
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    
    return df

class SimplePredictor:
    """간단한 예측 모델"""
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, df):
        """모델 훈련"""
        self.is_fitted = True
        
    def predict(self, df):
        """예측 수행"""
        if not self.is_fitted:
            return np.zeros(len(df))
        
        # 간단한 신호 생성
        signals = []
        for _, row in df.iterrows():
            # RSI 기반 신호
            rsi = row.get('rsi_14', 50)
            ma_20 = row.get('ma_20', row['close'])
            ma_50 = row.get('ma_50', row['close'])
            
            signal = 0
            if rsi < 30 and ma_20 > ma_50:  # 과매도 + 상승 추세
                signal = 0.001  # 상승 신호
            elif rsi > 70 and ma_20 < ma_50:  # 과매수 + 하락 추세
                signal = -0.001  # 하락 신호
            
            signals.append(signal)
        
        return np.array(signals)

def run_simple_backtest(df, initial_capital=10000000):
    """간단한 백테스트 실행"""
    print(f"💰 초기 자본: {initial_capital:,.0f}원")
    print(f"📊 데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # 피처 생성
    df = make_features(df)
    
    # 모델 훈련
    model = SimplePredictor()
    model.fit(df)
    
    # 백테스트 실행
    capital = initial_capital
    position = 0  # 0: 중립, 1: 롱, -1: 숏
    entry_price = 0
    trades = []
    
    print(f"📊 거래 시작... (총 {len(df)}개 캔들)")
    
    for idx, (timestamp, row) in enumerate(df.iterrows()):
        # 진행률 표시 (매 5000개 캔들마다만 표시)
        if idx % 5000 == 0 and idx > 0:
            progress = (idx / len(df)) * 100
            print(f"   진행률: {progress:.1f}% | 현재가: {row['close']:.0f} | 자본: {capital:,.0f}")
        
        # 예측 신호
        pred_df = pd.DataFrame([row])
        pred_df = make_features(pred_df)
        signal = model.predict(pred_df)[0]
        
        # 거래 신호 생성
        if position == 0:  # 포지션이 없을 때
            if signal > 0.0005:  # 상승 신호
                position = 1
                entry_price = row['close']
                entry_time = timestamp
                print(f"🎯 롱 진입: {entry_price:.0f} (신호: {signal:.4f})")
            elif signal < -0.0005:  # 하락 신호
                position = -1
                entry_price = row['close']
                entry_time = timestamp
                print(f"🎯 숏 진입: {entry_price:.0f} (신호: {signal:.4f})")
        
        else:  # 포지션이 있을 때
            current_price = row['close']
            
            # 청산 조건
            should_close = False
            close_reason = ""
            
            if position == 1:  # 롱 포지션
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct >= 0.02:  # 2% 익절
                    should_close = True
                    close_reason = "익절"
                elif profit_pct <= -0.01:  # 1% 손절
                    should_close = True
                    close_reason = "손절"
                elif signal < -0.0005:  # 반대 신호
                    should_close = True
                    close_reason = "신호 전환"
                    
            else:  # 숏 포지션
                profit_pct = (entry_price - current_price) / entry_price
                if profit_pct >= 0.02:  # 2% 익절
                    should_close = True
                    close_reason = "익절"
                elif profit_pct <= -0.01:  # 1% 손절
                    should_close = True
                    close_reason = "손절"
                elif signal > 0.0005:  # 반대 신호
                    should_close = True
                    close_reason = "신호 전환"
            
            # 청산 실행
            if should_close:
                # 손익 계산
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # 거래 수수료 및 슬리피지 (0.06%)
                pnl_pct -= 0.0006
                
                # 포지션 크기 (자본의 10%)
                position_size = capital * 0.1
                pnl = position_size * pnl_pct
                capital += pnl
                
                # 거래 기록
                trade = {
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'LONG' if position == 1 else 'SHORT',
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'reason': close_reason
                }
                trades.append(trade)
                
                print(f"   {close_reason}: {current_price:.0f} | 손익: {pnl_pct:.2%} ({pnl:,.0f}원) | 자본: {capital:,.0f}원")
                
                position = 0
    
    # 최종 결과 계산
    total_return = (capital - initial_capital) / initial_capital
    total_trades = len(trades)
    
    if total_trades > 0:
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        win_rate = winning_trades / total_trades
        avg_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0) / max(winning_trades, 1)
        avg_loss = sum(trade['pnl'] for trade in trades if trade['pnl'] < 0) / max(total_trades - winning_trades, 1)
    else:
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
    
    # 최대 낙폭 계산
    peak = initial_capital
    max_drawdown = 0
    for trade in trades:
        if trade['pnl'] > 0:
            peak = max(peak, capital)
        else:
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
    
    results = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'trades': trades
    }
    
    return results

def print_results(results):
    """결과 출력"""
    print(f"\n{'='*60}")
    print(f"🎉 백테스트 결과")
    print(f"{'='*60}")
    
    print(f"💰 초기 자본: {results['initial_capital']:,.0f}원")
    print(f"💰 최종 자본: {results['final_capital']:,.0f}원")
    print(f"📈 총 수익률: {results['total_return']:.2%}")
    print(f"💵 순이익: {results['final_capital'] - results['initial_capital']:,.0f}원")
    
    print(f"\n📊 거래 통계:")
    print(f"🎯 총 거래 수: {results['total_trades']}건")
    print(f"📊 승률: {results['win_rate']:.2%}")
    print(f"📈 평균 수익: {results['avg_profit']:,.0f}원")
    print(f"📉 평균 손실: {results['avg_loss']:,.0f}원")
    print(f"📊 최대 낙폭: {results['max_drawdown']:.2%}")
    
    # 성과 평가
    if results['total_return'] > 0.10:
        grade = "🏆 우수"
    elif results['total_return'] > 0.05:
        grade = "👍 양호"
    elif results['total_return'] > 0:
        grade = "📈 플러스"
    else:
        grade = "📉 손실"
    
    print(f"\n🏆 성과 등급: {grade}")
    print(f"{'='*60}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='간단한 백테스트 실행')
    parser.add_argument('--years', type=int, default=1, help='백테스트 기간 (년)')
    parser.add_argument('--capital', type=float, default=10000000, help='초기 자본')
    
    args = parser.parse_args()
    
    print(f"🚀 간단한 백테스트 시작!")
    print(f"📅 기간: {args.years}년")
    print(f"💰 초기 자본: {args.capital:,.0f}원")
    
    # 데이터 생성
    df = generate_historical_data(args.years)
    
    # 백테스트 실행
    results = run_simple_backtest(df, args.capital)
    
    # 결과 출력
    print_results(results)

if __name__ == "__main__":
    main() 