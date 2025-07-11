#!/usr/bin/env python3
"""
진단용 백테스트 - 신호 생성 문제 분석
매매 신호가 왜 발생하지 않는지 상세 분석
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

def generate_test_data(days=7):
    """진단용 짧은 기간 데이터 생성"""
    print("📊 진단용 데이터 생성 중...")
    
    # 시간 인덱스 생성
    start_date = datetime(2025, 6, 1)
    timestamps = pd.date_range(start=start_date, periods=days*24, freq='1H')
    
    # 더 변동성 있는 가격 시뮬레이션
    np.random.seed(42)
    base_price = 65000
    prices = []
    volumes = []
    
    for i in range(len(timestamps)):
        if i == 0:
            price = base_price
        else:
            # 더 큰 변동성 적용
            change = np.random.normal(0, 0.03)  # 3% 변동성으로 증가
            trend = 0.002 * np.sin(i / 12)     # 더 강한 트렌드
            price = prices[-1] * (1 + change + trend)
        
        prices.append(max(price, 1000))
        # 더 다양한 거래량
        volume = np.random.uniform(500, 2000) * (1 + abs(change))
        volumes.append(volume)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    print(f"✅ {len(df)}개 데이터 포인트 생성 완료")
    print(f"   가격 변동 범위: {(df['close'].max()/df['close'].min()-1)*100:.1f}%")
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
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # NaN 처리
    df = df.fillna(method='ffill').fillna(0)
    
    print("✅ 기술적 지표 계산 완료")
    return df

def diagnostic_ml_prediction(df):
    """진단용 ML 예측 - 더 강한 신호 생성"""
    print("🤖 진단용 ML 예측 모델 실행 중...")
    
    predictions = []
    
    for i in range(len(df)):
        if i < 20:
            predictions.append(0.0)
        else:
            # 최근 데이터 기반 예측
            recent = df.iloc[i-20:i]
            
            # 특징 추출
            price_trend = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            ma_signal = (recent['ma_5'].iloc[-1] - recent['ma_20'].iloc[-1]) / recent['ma_20'].iloc[-1]
            rsi_signal = (recent['rsi'].iloc[-1] - 50) / 50
            vol_signal = recent['volatility'].iloc[-1]
            
            # 더 강한 예측 신호 생성
            prediction = price_trend * 0.5 + ma_signal * 0.4 + rsi_signal * 0.3
            # 범위 확장
            prediction = max(min(prediction, 0.08), -0.08)
            
            predictions.append(prediction)
    
    print(f"✅ ML 예측 완료")
    print(f"   예측값 범위: {min(predictions):.4f} ~ {max(predictions):.4f}")
    strong_signals = [p for p in predictions if abs(p) > 0.01]
    print(f"   강한 신호(>1%): {len(strong_signals)}개")
    
    return predictions

def analyze_market_condition(row):
    """시장 상황 분석"""
    rsi = row.get('rsi', 50)
    ma_5 = row.get('ma_5', row['close'])
    ma_20 = row.get('ma_20', row['close'])
    volatility = row.get('volatility', 0.02)
    
    if ma_5 > ma_20 * 1.005 and volatility < 0.04:  # 임계값 완화
        return 'trending_up'
    elif ma_5 < ma_20 * 0.995 and volatility < 0.04:  # 임계값 완화
        return 'trending_down'
    elif volatility > 0.04:  # 임계값 완화
        return 'volatile'
    else:
        return 'sideways'

def diagnostic_strategy(row, ml_pred, market_condition):
    """진단용 전략 - 더 관대한 조건"""
    
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
    
    # 전략 1: 추세 순응형 (조건 완화)
    if market_condition in ['trending_up', 'trending_down']:
        confidence = 0.0
        
        if market_condition == 'trending_up' and ml_pred > 0.005:  # 0.01에서 0.005로 완화
            if rsi < 70 and volume_ratio > 1.0:  # 1.2에서 1.0으로 완화
                confidence = min(0.8, abs(ml_pred) * 30)  # 신뢰도 증가
                signal['action'] = 'BUY'
                signal['strategy_used'] = 'trend_following'
        
        elif market_condition == 'trending_down' and ml_pred < -0.005:  # 조건 완화
            if rsi > 30 and volume_ratio > 1.0:  # 조건 완화
                confidence = min(0.8, abs(ml_pred) * 30)
                signal['action'] = 'SELL'
                signal['strategy_used'] = 'trend_following'
        
        signal['confidence'] = confidence
    
    # 전략 2: 스캘핑 (조건 완화)
    elif market_condition == 'sideways':
        confidence = 0.0
        
        if ml_pred > 0.003 and rsi < 55:  # 조건 완화
            confidence = min(0.9, abs(ml_pred) * 40)
            signal['action'] = 'BUY'
            signal['strategy_used'] = 'scalping'
        
        elif ml_pred < -0.003 and rsi > 45:  # 조건 완화
            confidence = min(0.9, abs(ml_pred) * 40)
            signal['action'] = 'SELL'
            signal['strategy_used'] = 'scalping'
        
        signal['confidence'] = confidence
    
    # 전략 3: 변동성 돌파 (조건 완화)
    elif market_condition == 'volatile':
        confidence = 0.0
        
        if bb_width < 0.05:  # 조건 완화
            if abs(ml_pred) > 0.01 and volume_ratio > 1.2:  # 조건 완화
                confidence = min(0.7, abs(ml_pred) * 20)
                if ml_pred > 0:
                    signal['action'] = 'BUY'
                else:
                    signal['action'] = 'SELL'
                signal['strategy_used'] = 'volatility_breakout'
        
        signal['confidence'] = confidence
    
    return signal

def diagnostic_backtest(df, predictions, min_confidence=0.5):  # 신뢰도 임계값 완화
    """진단용 백테스트 실행"""
    print("💰 진단용 백테스트 실행 중...")
    
    signal_stats = {
        'total_signals': 0,
        'confident_signals': 0,
        'by_strategy': {'trend_following': 0, 'scalping': 0, 'volatility_breakout': 0},
        'by_action': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
        'by_market': {'trending_up': 0, 'trending_down': 0, 'sideways': 0, 'volatile': 0}
    }
    
    trades = []
    detailed_signals = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        current_price = row['close']
        ml_pred = predictions[i] if i < len(predictions) else 0.0
        
        # 시장 상황 분석
        market_condition = analyze_market_condition(row)
        
        # 전략 신호 생성
        signal = diagnostic_strategy(row, ml_pred, market_condition)
        
        # 통계 수집
        signal_stats['total_signals'] += 1
        signal_stats['by_action'][signal['action']] += 1
        signal_stats['by_market'][market_condition] += 1
        
        if signal['action'] != 'HOLD':
            signal_stats['by_strategy'][signal['strategy_used']] += 1
            
            if signal['confidence'] >= min_confidence:
                signal_stats['confident_signals'] += 1
                trades.append({
                    'index': i,
                    'timestamp': row['timestamp'],
                    'action': signal['action'],
                    'confidence': signal['confidence'],
                    'strategy': signal['strategy_used'],
                    'ml_pred': ml_pred,
                    'market_condition': market_condition,
                    'price': current_price,
                    'rsi': row.get('rsi', 50),
                    'volume_ratio': row.get('volume_ratio', 1.0)
                })
        
        # 상세 신호 정보 저장 (샘플링)
        if i % 10 == 0 or signal['action'] != 'HOLD':
            detailed_signals.append({
                'index': i,
                'timestamp': row['timestamp'],
                'action': signal['action'],
                'confidence': signal['confidence'],
                'strategy': signal['strategy_used'],
                'ml_pred': ml_pred,
                'market_condition': market_condition,
                'rsi': row.get('rsi', 50),
                'volume_ratio': row.get('volume_ratio', 1.0),
                'bb_width': row.get('bb_width', 0.04)
            })
    
    return {
        'signal_stats': signal_stats,
        'trades': trades,
        'detailed_signals': detailed_signals
    }

def print_diagnostic_results(results):
    """진단 결과 출력"""
    print("\n" + "="*80)
    print("🔍 신호 생성 진단 결과")
    print("="*80)
    
    stats = results['signal_stats']
    trades = results['trades']
    signals = results['detailed_signals']
    
    print(f"📊 전체 신호 통계:")
    print(f"   총 분석 포인트: {stats['total_signals']}개")
    print(f"   신뢰도 충족 신호: {stats['confident_signals']}개")
    print(f"   실제 거래 실행: {len(trades)}개")
    
    print(f"\n📈 액션별 분포:")
    for action, count in stats['by_action'].items():
        pct = (count / stats['total_signals']) * 100
        print(f"   {action}: {count}개 ({pct:.1f}%)")
    
    print(f"\n🎯 전략별 신호:")
    for strategy, count in stats['by_strategy'].items():
        print(f"   {strategy}: {count}개")
    
    print(f"\n📊 시장 상황별 분포:")
    for market, count in stats['by_market'].items():
        pct = (count / stats['total_signals']) * 100
        print(f"   {market}: {count}개 ({pct:.1f}%)")
    
    if trades:
        print(f"\n💰 실행된 거래 목록:")
        print("-" * 60)
        for trade in trades[:10]:  # 처음 10개만 표시
            print(f"   {trade['timestamp']} | {trade['action']:<4} | "
                  f"{trade['strategy']:<15} | 신뢰도: {trade['confidence']:.2f} | "
                  f"ML: {trade['ml_pred']:+.4f} | {trade['market_condition']}")
        if len(trades) > 10:
            print(f"   ... 및 {len(trades)-10}개 추가 거래")
    else:
        print(f"\n❌ 실행된 거래가 없습니다!")
        
        print(f"\n🔍 신호 생성 실패 원인 분석:")
        
        # 샘플 신호들 분석
        no_signal_count = sum(1 for s in signals if s['action'] == 'HOLD')
        low_confidence_count = sum(1 for s in signals if s['action'] != 'HOLD' and s['confidence'] < 0.5)
        
        print(f"   HOLD 신호: {no_signal_count}개")
        print(f"   낮은 신뢰도: {low_confidence_count}개")
        
        # 최근 몇 개 신호 상세 분석
        print(f"\n📋 최근 신호 상세 분석:")
        print("-" * 80)
        for signal in signals[-5:]:
            print(f"   시간: {signal['timestamp']}")
            print(f"   시장상황: {signal['market_condition']}, ML예측: {signal['ml_pred']:+.4f}")
            print(f"   RSI: {signal['rsi']:.1f}, 거래량비율: {signal['volume_ratio']:.2f}")
            print(f"   액션: {signal['action']}, 신뢰도: {signal['confidence']:.3f}")
            print(f"   전략: {signal['strategy']}")
            print("-" * 40)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='신호 생성 진단 백테스트')
    parser.add_argument('--days', type=int, default=7, help='진단 기간 (기본: 7일)')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='최소 신뢰도 (기본: 0.5)')
    
    args = parser.parse_args()
    
    print("🔍 신호 생성 진단 백테스트 시작")
    print(f"📅 기간: {args.days}일")
    print(f"🎯 최소 신뢰도: {args.min_confidence}")
    print()
    
    try:
        # 1. 진단용 데이터 생성
        df = generate_test_data(args.days)
        
        # 2. 기술적 지표 계산
        df = calculate_indicators(df)
        
        # 3. 진단용 ML 예측
        predictions = diagnostic_ml_prediction(df)
        
        # 4. 진단용 백테스트 실행
        results = diagnostic_backtest(df, predictions, args.min_confidence)
        
        # 5. 진단 결과 분석
        print_diagnostic_results(results)
        
        print(f"\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 진단이 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()