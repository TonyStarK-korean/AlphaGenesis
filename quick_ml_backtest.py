#!/usr/bin/env python3
"""
간단하고 안정적인 ML 백테스트 실행 스크립트
문제없이 빠르게 실행되도록 최적화됨
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def print_header():
    """헤더 출력"""
    print("🚀 AlphaGenesis 간단 ML 백테스트")
    print("=" * 60)
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def generate_sample_data(days=90):
    """샘플 데이터 생성"""
    print("📊 시뮬레이션 데이터 생성 중...")
    
    # 시간 인덱스 생성 (90일, 1시간 단위)
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=days*24, freq='1H')
    
    # 가격 데이터 생성 (비트코인 기준)
    np.random.seed(42)
    base_price = 50000
    prices = []
    
    for i in range(len(timestamps)):
        if i == 0:
            price = base_price
        else:
            # 랜덤 워크 + 트렌드
            change = np.random.normal(0, 0.015)  # 1.5% 표준편차
            trend = 0.0001 * np.sin(i / 168)  # 주간 사이클
            price = prices[-1] * (1 + change + trend)
        prices.append(max(price, 1000))  # 최소 가격 제한
    
    # DataFrame 생성
    data = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(timestamps))
    })
    
    print(f"✅ {len(data)}개 데이터 포인트 생성 완료")
    return data

def calculate_technical_indicators(data):
    """기술적 지표 계산"""
    print("📈 기술적 지표 계산 중...")
    
    df = data.copy()
    
    # 이동평균
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 변동성
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    # 거래량 비율
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    print("✅ 기술적 지표 계산 완료")
    return df

def simple_ml_prediction(data, lookback=50):
    """간단한 ML 예측 모델"""
    print("🤖 ML 예측 모델 실행 중...")
    
    predictions = []
    
    for i in range(len(data)):
        if i < lookback:
            predictions.append(0.0)
        else:
            # 최근 데이터 기반 간단한 예측
            recent_data = data.iloc[i-lookback:i]
            
            # 특징들
            ma_trend = (recent_data['ma_5'].iloc[-1] - recent_data['ma_20'].iloc[-1]) / recent_data['ma_20'].iloc[-1]
            rsi_signal = (recent_data['rsi'].iloc[-1] - 50) / 50
            bb_signal = recent_data['bb_position'].iloc[-1] - 0.5
            vol_signal = recent_data['volatility'].iloc[-1]
            
            # 간단한 선형 조합
            prediction = (ma_trend * 0.4 + rsi_signal * 0.3 + bb_signal * 0.2 + vol_signal * 0.1)
            predictions.append(max(min(prediction, 0.05), -0.05))  # -5% ~ +5% 제한
    
    print("✅ ML 예측 완료")
    return predictions

def generate_trading_signals(data, predictions, confidence_threshold=0.6):
    """거래 신호 생성"""
    print("🎯 거래 신호 생성 중...")
    
    signals = []
    
    for i in range(len(data)):
        signal = {
            'action': 'HOLD',  # BUY, SELL, HOLD
            'confidence': 0.0,
            'prediction': predictions[i] if i < len(predictions) else 0.0
        }
        
        if i < 50:  # 초기 50개는 HOLD
            signals.append(signal)
            continue
        
        # 예측값 기반 신호 생성
        pred = predictions[i]
        row = data.iloc[i]
        
        # 신뢰도 계산
        confidence = 0.0
        
        # RSI 기반 신뢰도
        rsi = row.get('rsi', 50)
        if pred > 0 and rsi < 40:  # 상승 예측 + 과매도
            confidence += 0.3
        elif pred < 0 and rsi > 60:  # 하락 예측 + 과매수
            confidence += 0.3
        
        # 이동평균 기반 신뢰도
        if pred > 0 and row['ma_5'] > row['ma_20']:  # 상승 예측 + 상승 추세
            confidence += 0.3
        elif pred < 0 and row['ma_5'] < row['ma_20']:  # 하락 예측 + 하락 추세
            confidence += 0.3
        
        # 볼린저 밴드 기반 신뢰도
        bb_pos = row.get('bb_position', 0.5)
        if pred > 0 and bb_pos < 0.2:  # 상승 예측 + 하단 근처
            confidence += 0.2
        elif pred < 0 and bb_pos > 0.8:  # 하락 예측 + 상단 근처
            confidence += 0.2
        
        # 예측 강도 기반 신뢰도
        confidence += abs(pred) * 2
        
        # 최종 신호 결정
        signal['confidence'] = min(confidence, 1.0)
        
        if signal['confidence'] >= confidence_threshold:
            if pred > 0.01:
                signal['action'] = 'BUY'
            elif pred < -0.01:
                signal['action'] = 'SELL'
        
        signals.append(signal)
    
    print(f"✅ 거래 신호 생성 완료 (신뢰도 임계값: {confidence_threshold})")
    return signals

def run_backtest(data, signals, initial_capital=10000000):
    """백테스트 실행"""
    print("💰 백테스트 실행 중...")
    
    capital = initial_capital
    position = 0  # 0: 현금, 1: 보유
    shares = 0
    trades = []
    portfolio_values = []
    
    for i, (_, row) in enumerate(data.iterrows()):
        current_price = row['close']
        signal = signals[i] if i < len(signals) else {'action': 'HOLD', 'confidence': 0.0}
        
        # 포트폴리오 가치 계산
        if position == 1:
            portfolio_value = shares * current_price
        else:
            portfolio_value = capital
        portfolio_values.append(portfolio_value)
        
        # 매수 신호
        if signal['action'] == 'BUY' and position == 0:
            shares = capital / current_price
            position = 1
            trades.append({
                'type': 'BUY',
                'price': current_price,
                'timestamp': row['timestamp'],
                'confidence': signal['confidence']
            })
            print(f"📈 매수: {current_price:,.0f}원 (신뢰도: {signal['confidence']:.2f})")
        
        # 매도 신호
        elif signal['action'] == 'SELL' and position == 1:
            capital = shares * current_price
            profit = capital - initial_capital
            profit_pct = (profit / initial_capital) * 100
            
            trades.append({
                'type': 'SELL',
                'price': current_price,
                'timestamp': row['timestamp'],
                'confidence': signal['confidence'],
                'profit': profit,
                'profit_pct': profit_pct
            })
            position = 0
            shares = 0
            print(f"📉 매도: {current_price:,.0f}원 (수익: {profit:,.0f}원, {profit_pct:.2f}%)")
    
    # 마지막 포지션 정리
    if position == 1:
        final_price = data['close'].iloc[-1]
        capital = shares * final_price
    
    print("✅ 백테스트 완료")
    return capital, trades, portfolio_values

def analyze_results(initial_capital, final_capital, trades, portfolio_values):
    """결과 분석"""
    print("\n" + "="*60)
    print("📊 백테스트 결과 분석")
    print("="*60)
    
    # 기본 수익률
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    print(f"💰 초기 자본: {initial_capital:,.0f}원")
    print(f"💰 최종 자본: {final_capital:,.0f}원")
    print(f"📈 총 수익률: {total_return:.2f}%")
    print(f"💵 절대 수익: {final_capital - initial_capital:,.0f}원")
    
    # 거래 통계
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL' and 'profit' in t]
        
        print(f"\n📊 거래 통계:")
        print(f"   총 매수: {len(buy_trades)}회")
        print(f"   총 매도: {len(sell_trades)}회")
        
        if sell_trades:
            winning_trades = [t for t in sell_trades if t['profit'] > 0]
            losing_trades = [t for t in sell_trades if t['profit'] <= 0]
            
            win_rate = len(winning_trades) / len(sell_trades) * 100
            avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            print(f"   승리 거래: {len(winning_trades)}회")
            print(f"   패배 거래: {len(losing_trades)}회")
            print(f"   승률: {win_rate:.1f}%")
            print(f"   평균 수익: {avg_win:,.0f}원")
            print(f"   평균 손실: {avg_loss:,.0f}원")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                print(f"   수익 팩터: {profit_factor:.2f}")
    
    # 최대 드로우다운 계산
    if portfolio_values:
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        print(f"   최대 드로우다운: {max_drawdown*100:.2f}%")
    
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
    # 명령행 인수 처리
    parser = argparse.ArgumentParser(description='AlphaGenesis 간단 ML 백테스트')
    parser.add_argument('--initial-capital', type=int, default=10000000, help='초기 자본 (기본: 1000만원)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='신뢰도 임계값 (기본: 0.6)')
    parser.add_argument('--days', type=int, default=90, help='백테스트 기간 (기본: 90일)')
    
    args = parser.parse_args()
    
    # 헤더 출력
    print_header()
    
    print(f"⚙️  설정:")
    print(f"   초기 자본: {args.initial_capital:,.0f}원")
    print(f"   신뢰도 임계값: {args.confidence_threshold}")
    print(f"   백테스트 기간: {args.days}일")
    print()
    
    try:
        # 1. 데이터 생성
        data = generate_sample_data(args.days)
        
        # 2. 기술적 지표 계산
        data = calculate_technical_indicators(data)
        
        # 3. ML 예측
        predictions = simple_ml_prediction(data)
        
        # 4. 거래 신호 생성
        signals = generate_trading_signals(data, predictions, args.confidence_threshold)
        
        # 5. 백테스트 실행
        final_capital, trades, portfolio_values = run_backtest(data, signals, args.initial_capital)
        
        # 6. 결과 분석
        analyze_results(args.initial_capital, final_capital, trades, portfolio_values)
        
        # 7. 완료 메시지
        print(f"\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 백테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        print("🔧 문제가 지속되면 라이브러리 설치를 확인해주세요.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())