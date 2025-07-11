#!/usr/bin/env python3
"""
간단한 트리플 콤보 백테스트 테스트
문제 해결을 위한 최소한의 구현
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

def generate_test_data(days=30):
    """테스트용 데이터 생성"""
    print("📊 테스트 데이터 생성 중...")
    
    # 시간 인덱스 생성
    start_date = datetime(2025, 6, 1)
    timestamps = pd.date_range(start=start_date, periods=days*24, freq='h')
    
    # 비트코인 가격 시뮬레이션
    np.random.seed(42)
    base_price = 65000
    prices = []
    volumes = []
    
    for i in range(len(timestamps)):
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 0.02)  # 2% 변동성
            trend = 0.0001 * np.sin(i / 24)    # 일일 사이클
            price = prices[-1] * (1 + change + trend)
        
        prices.append(max(price, 1000))
        volumes.append(np.random.uniform(100, 1000))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    print(f"✅ {len(df)}개 데이터 포인트 생성 완료")
    return df

def calculate_indicators(df):
    """기술적 지표 계산"""
    print("📈 기술적 지표 계산 중...")
    
    # 이동평균
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(14).mean()
    
    # 볼린저 밴드
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 변동성
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    # 거래량 분석
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)  # 0으로 나누기 방지
    
    # NaN 처리 - pandas 최신 버전 호환
    df = df.ffill().fillna(0)
    
    print("✅ 기술적 지표 계산 완료")
    return df

def simple_ml_prediction(df):
    """간단한 ML 예측 - 신호 강화"""
    print("🤖 ML 예측 모델 실행 중...")
    
    predictions = []
    
    for i in range(len(df)):
        if i < 50:
            predictions.append(0.0)
        else:
            # 최근 데이터 기반 예측
            recent = df.iloc[i-20:i]
            
            # 특징 추출
            price_trend = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            ma_signal = (recent['ma_5'].iloc[-1] - recent['ma_20'].iloc[-1]) / recent['ma_20'].iloc[-1]
            rsi_signal = (recent['rsi'].iloc[-1] - 50) / 50
            vol_signal = recent['volatility'].iloc[-1]
            
            # 간단한 예측 - 신호 증폭
            prediction = price_trend * 0.4 + ma_signal * 0.3 + rsi_signal * 0.2 + vol_signal * 0.1
            prediction = prediction * 3.0  # 신호 증폭
            predictions.append(max(min(prediction, 0.05), -0.05))
    
    strong_signals = [p for p in predictions if abs(p) > 0.005]
    print(f"✅ ML 예측 완료 (강한 신호: {len(strong_signals)}개)")
    return predictions

def analyze_market_condition(row):
    """시장 상황 분석 - 더 민감한 감지"""
    rsi = row.get('rsi', 50)
    ma_5 = row.get('ma_5', row['close'])
    ma_20 = row.get('ma_20', row['close'])
    volatility = row.get('volatility', 0.02)
    
    # 더 민감한 조건으로 변경 - 0으로 나누기 방지
    ma_diff = (ma_5 - ma_20) / (ma_20 + 1e-10) if ma_20 != 0 else 0
    
    if ma_diff > 0.001 and volatility < 0.04:  # 조건 완화
        return 'trending_up'
    elif ma_diff < -0.001 and volatility < 0.04:  # 조건 완화
        return 'trending_down'
    elif volatility > 0.03:  # 조건 완화
        return 'volatile'
    else:
        return 'sideways'

def triple_combo_strategy(row, ml_pred, market_condition):
    """트리플 콤보 전략 시뮬레이션"""
    
    signal = {
        'action': 'HOLD',
        'confidence': 0.0,
        'strategy_used': 'none'
    }
    
    close = row['close']
    rsi = row.get('rsi', 50)
    atr = row.get('atr', close * 0.02)
    bb_width = row.get('bb_width', 0.04)
    volume_ratio = row.get('volume_ratio', 1.0)
    
    # 전략 1: 추세 순응형 (Trend Following) - 조건 완화
    if market_condition in ['trending_up', 'trending_down']:
        confidence = 0.0
        
        # 추세 확인 - 조건 대폭 완화
        if market_condition == 'trending_up' and ml_pred > 0.003:  # 0.01 -> 0.003
            if rsi < 70 and volume_ratio > 0.8:  # 조건 완화
                confidence += 0.8  # 신뢰도 증가
                signal['action'] = 'BUY'
                signal['strategy_used'] = 'trend_following'
        
        elif market_condition == 'trending_down' and ml_pred < -0.003:  # 조건 완화
            if rsi > 30 and volume_ratio > 0.8:  # 조건 완화
                confidence += 0.8  # 신뢰도 증가
                signal['action'] = 'SELL'
                signal['strategy_used'] = 'trend_following'
        
        signal['confidence'] = confidence
    
    # 전략 2: 스캘핑 (CVD Scalping) - 조건 완화
    elif market_condition == 'sideways':
        confidence = 0.0
        
        if ml_pred > 0.002 and rsi < 60:  # 조건 대폭 완화
            confidence += 0.8  # 신뢰도 증가
            signal['action'] = 'BUY'
            signal['strategy_used'] = 'scalping'
        
        elif ml_pred < -0.002 and rsi > 40:  # 조건 대폭 완화
            confidence += 0.8  # 신뢰도 증가
            signal['action'] = 'SELL'
            signal['strategy_used'] = 'scalping'
        
        signal['confidence'] = confidence
    
    # 전략 3: 변동성 돌파 (Volatility Breakout) - 조건 완화
    elif market_condition == 'volatile':
        confidence = 0.0
        
        # 조건 대폭 완화
        if abs(ml_pred) > 0.005 and volume_ratio > 1.0:  # 조건 완화
            confidence += 0.7  # 신뢰도 증가
            if ml_pred > 0:
                signal['action'] = 'BUY'
            else:
                signal['action'] = 'SELL'
            signal['strategy_used'] = 'volatility_breakout'
        
        signal['confidence'] = confidence
    
    return signal

def run_backtest(df, predictions, min_confidence=0.4):  # 신뢰도 임계값 완화
    """백테스트 실행"""
    print(f"💰 백테스트 실행 중... (최소 신뢰도: {min_confidence})")
    
    initial_capital = 10000000
    capital = initial_capital
    position = 0  # 0: 현금, 1: 보유, -1: 공매도
    shares = 0
    trades = []
    portfolio_values = []
    
    # 신호 통계 추가
    signal_count = 0
    executed_count = 0
    
    strategy_stats = {
        'trend_following': {'count': 0, 'profit': 0},
        'scalping': {'count': 0, 'profit': 0},
        'volatility_breakout': {'count': 0, 'profit': 0}
    }
    
    for i, (_, row) in enumerate(df.iterrows()):
        current_price = row['close']
        ml_pred = predictions[i] if i < len(predictions) else 0.0
        
        # 시장 상황 분석
        market_condition = analyze_market_condition(row)
        
        # 트리플 콤보 신호 생성
        signal = triple_combo_strategy(row, ml_pred, market_condition)
        
        # 신호 통계
        if signal['action'] != 'HOLD':
            signal_count += 1
            if signal_count <= 10:  # 처음 10개 신호만 로그
                print(f"   신호 {signal_count}: {signal['action']} | 신뢰도: {signal['confidence']:.3f} | 전략: {signal['strategy_used']}")
        
        # 포트폴리오 가치 계산
        if position != 0:
            portfolio_value = capital + (shares * current_price * position)
        else:
            portfolio_value = capital
        portfolio_values.append(portfolio_value)
        
        # 거래 실행
        if signal['confidence'] >= min_confidence:
            if signal['action'] == 'BUY' and position <= 0:
                # 매수
                if position == -1:  # 공매도 청산
                    capital += shares * current_price * position
                
                shares = capital / current_price
                position = 1
                capital = 0
                
                executed_count += 1
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'strategy': signal['strategy_used'],
                    'confidence': signal['confidence'],
                    'timestamp': row['timestamp']
                })
                
                print(f"   ✅ 매수 체결 #{executed_count}: {current_price:.0f}원 ({signal['strategy_used']})")
                
            elif signal['action'] == 'SELL' and position >= 0:
                # 매도
                if position == 1:  # 보유 청산
                    capital = shares * current_price
                
                shares = capital / current_price
                position = -1
                capital = 0
                
                executed_count += 1
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'strategy': signal['strategy_used'],
                    'confidence': signal['confidence'],
                    'timestamp': row['timestamp']
                })
                
                print(f"   ✅ 매도 체결 #{executed_count}: {current_price:.0f}원 ({signal['strategy_used']})")
    
    # 최종 정산
    if position != 0:
        capital += shares * df['close'].iloc[-1] * position
    
    print(f"✅ 백테스트 완료")
    print(f"   총 신호: {signal_count}개, 실행 거래: {executed_count}개")
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'strategy_stats': strategy_stats,
        'signal_count': signal_count,
        'executed_count': executed_count
    }

def analyze_results(results):
    """결과 분석"""
    print("\n" + "="*60)
    print("📊 트리플 콤보 백테스트 결과")
    print("="*60)
    
    initial = results['initial_capital']
    final = results['final_capital']
    total_return = (final - initial) / initial * 100
    
    print(f"💰 초기 자본: {initial:,.0f}원")
    print(f"💰 최종 자본: {final:,.0f}원")
    print(f"📈 총 수익률: {total_return:.2f}%")
    print(f"💵 절대 수익: {final - initial:,.0f}원")
    
    # 거래 분석
    trades = results['trades']
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        print(f"\n📊 거래 통계:")
        print(f"   총 매수: {len(buy_trades)}회")
        print(f"   총 매도: {len(sell_trades)}회")
        
        # 전략별 사용 횟수
        strategy_counts = {}
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        print(f"   전략별 사용:")
        for strategy, count in strategy_counts.items():
            print(f"      {strategy}: {count}회")
    
    # 성과 등급
    if total_return > 20:
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
        print("❌ 손실이 발생했습니다.")
    
    print("="*60)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='간단한 트리플 콤보 백테스트')
    parser.add_argument('--days', type=int, default=30, help='백테스트 기간 (기본: 30일)')
    parser.add_argument('--min-confidence', type=float, default=0.6, help='최소 신뢰도 (기본: 0.6)')
    
    args = parser.parse_args()
    
    print("🚀 간단한 트리플 콤보 백테스트 시작")
    print(f"📅 기간: {args.days}일")
    print(f"🎯 최소 신뢰도: {args.min_confidence} (수정됨)")
    print(f"🔧 개선사항: 신호 생성 조건 완화, 로깅 추가")
    print()
    
    try:
        # 1. 데이터 생성
        df = generate_test_data(args.days)
        
        # 2. 기술적 지표 계산
        df = calculate_indicators(df)
        
        # 3. ML 예측
        predictions = simple_ml_prediction(df)
        
        # 4. 백테스트 실행
        results = run_backtest(df, predictions, args.min_confidence)
        
        # 5. 결과 분석
        analyze_results(results)
        
        print(f"\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 백테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()