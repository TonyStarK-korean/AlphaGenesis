#!/usr/bin/env python3
"""
수정된 간단한 트리플 콤보 백테스트
신호 생성 문제를 해결한 버전
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

def generate_test_data(days=30):
    """테스트용 데이터 생성 - 더 현실적인 패턴"""
    print("📊 수정된 테스트 데이터 생성 중...")
    
    # 시간 인덱스 생성
    start_date = datetime(2025, 6, 1)
    timestamps = pd.date_range(start=start_date, periods=days*24, freq='h')
    
    # 더 현실적인 비트코인 가격 시뮬레이션
    np.random.seed(42)
    base_price = 65000
    prices = []
    volumes = []
    
    # 시장 상황 시뮬레이션 (일별 변화)
    market_phases = np.random.choice(['bull', 'bear', 'sideways', 'volatile'], 
                                   size=days, p=[0.3, 0.2, 0.3, 0.2])
    
    for i in range(len(timestamps)):
        day_index = i // 24
        phase = market_phases[min(day_index, len(market_phases)-1)]
        
        # 변화율 초기화
        change = 0.0
        
        if i == 0:
            price = base_price
        else:
            # 시장 상황에 따른 가격 변화
            if phase == 'bull':
                change = np.random.normal(0.002, 0.025)  # 상승 편향
            elif phase == 'bear':
                change = np.random.normal(-0.002, 0.025)  # 하락 편향
            elif phase == 'volatile':
                change = np.random.normal(0, 0.04)  # 높은 변동성
            else:  # sideways
                change = np.random.normal(0, 0.015)  # 낮은 변동성
            
            # 시간별 사이클 추가
            hourly_cycle = 0.0005 * np.sin(2 * np.pi * (i % 24) / 24)
            price = prices[-1] * (1 + change + hourly_cycle)
        
        prices.append(max(price, 10000))
        
        # 변동성에 따른 거래량 조정
        base_vol = 1000
        if abs(change) > 0.02:  # 큰 변동성일 때 거래량 증가
            volume_multiplier = np.random.uniform(1.5, 3.0)
        else:
            volume_multiplier = np.random.uniform(0.5, 1.5)
        
        volumes.append(base_vol * volume_multiplier)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    print(f"✅ {len(df)}개 데이터 포인트 생성 완료")
    print(f"   가격 변동: {(df['close'].max()/df['close'].min()-1)*100:.1f}%")
    print(f"   평균 거래량: {df['volume'].mean():.0f}")
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

def enhanced_ml_prediction(df):
    """향상된 ML 예측 - 더 강한 신호 생성"""
    print("🤖 향상된 ML 예측 모델 실행 중...")
    
    predictions = []
    
    for i in range(len(df)):
        if i < 50:
            predictions.append(0.0)
        else:
            # 다양한 기간의 데이터 활용
            short_term = df.iloc[i-10:i]  # 10시간
            medium_term = df.iloc[i-24:i]  # 24시간
            long_term = df.iloc[i-50:i]   # 50시간
            
            # 다중 시간대 분석
            # 1. 가격 모멘텀 - 0으로 나누기 방지
            short_base = short_term['close'].iloc[0]
            medium_base = medium_term['close'].iloc[0]
            short_momentum = (short_term['close'].iloc[-1] - short_base) / (short_base + 1e-10) if short_base != 0 else 0
            medium_momentum = (medium_term['close'].iloc[-1] - medium_base) / (medium_base + 1e-10) if medium_base != 0 else 0
            
            # 2. 이동평균 신호 - 0으로 나누기 방지
            ma_20_val = df.iloc[i]['ma_20']
            ma_50_val = df.iloc[i]['ma_50']
            ma_short = (df.iloc[i]['ma_5'] - ma_20_val) / (ma_20_val + 1e-10) if ma_20_val != 0 else 0
            ma_long = (ma_20_val - ma_50_val) / (ma_50_val + 1e-10) if ma_50_val != 0 else 0
            
            # 3. RSI 신호
            rsi_signal = (df.iloc[i]['rsi'] - 50) / 50
            
            # 4. 변동성 신호
            vol_signal = df.iloc[i]['volatility'] * np.sign(short_momentum)
            
            # 5. 거래량 신호
            volume_signal = (df.iloc[i]['volume_ratio'] - 1) * 0.1
            
            # 종합 예측 (가중 평균)
            prediction = (
                short_momentum * 0.25 + 
                medium_momentum * 0.20 + 
                ma_short * 0.20 + 
                ma_long * 0.15 + 
                rsi_signal * 0.10 + 
                vol_signal * 0.05 + 
                volume_signal * 0.05
            )
            
            # 신호 증폭 (더 강한 신호 생성)
            prediction = prediction * 2.0
            
            # 범위 제한
            prediction = max(min(prediction, 0.1), -0.1)
            
            predictions.append(prediction)
    
    print(f"✅ 향상된 ML 예측 완료")
    strong_signals = [p for p in predictions if abs(p) > 0.01]
    print(f"   강한 신호 수: {len(strong_signals)}개 (전체의 {len(strong_signals)/len(predictions)*100:.1f}%)")
    print(f"   예측값 범위: {min(predictions):.4f} ~ {max(predictions):.4f}")
    
    return predictions

def improved_market_analysis(row):
    """개선된 시장 상황 분석 - 더 민감한 감지"""
    rsi = row.get('rsi', 50)
    ma_5 = row.get('ma_5', row['close'])
    ma_20 = row.get('ma_20', row['close'])
    ma_50 = row.get('ma_50', row['close'])
    volatility = row.get('volatility', 0.02)
    volume_ratio = row.get('volume_ratio', 1.0)
    
    # 더 민감한 추세 감지 - 0으로 나누기 방지
    ma_diff_short = (ma_5 - ma_20) / (ma_20 + 1e-10) if ma_20 != 0 else 0
    ma_diff_long = (ma_20 - ma_50) / (ma_50 + 1e-10) if ma_50 != 0 else 0
    
    # 강한 상승 추세
    if ma_diff_short > 0.002 and ma_diff_long > 0.001 and volatility < 0.05:
        return 'trending_up'
    # 강한 하락 추세
    elif ma_diff_short < -0.002 and ma_diff_long < -0.001 and volatility < 0.05:
        return 'trending_down'
    # 고변동성
    elif volatility > 0.03 or volume_ratio > 1.5:
        return 'volatile'
    # 횡보
    else:
        return 'sideways'

def improved_triple_combo_strategy(row, ml_pred, market_condition):
    """개선된 트리플 콤보 전략 - 더 관대한 조건"""
    
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
    
    # 전략 1: 추세 순응형 (조건 대폭 완화)
    if market_condition in ['trending_up', 'trending_down']:
        confidence = 0.0
        
        if market_condition == 'trending_up' and ml_pred > 0.003:  # 0.01에서 0.003으로 완화
            if rsi < 75 and volume_ratio > 0.8:  # 조건 대폭 완화
                confidence = min(0.9, abs(ml_pred) * 50)  # 신뢰도 증가
                signal['action'] = 'BUY'
                signal['strategy_used'] = 'trend_following'
        
        elif market_condition == 'trending_down' and ml_pred < -0.003:
            if rsi > 25 and volume_ratio > 0.8:
                confidence = min(0.9, abs(ml_pred) * 50)
                signal['action'] = 'SELL'
                signal['strategy_used'] = 'trend_following'
        
        signal['confidence'] = confidence
    
    # 전략 2: 스캘핑 (조건 완화)
    elif market_condition == 'sideways':
        confidence = 0.0
        
        if ml_pred > 0.002:  # 0.005에서 0.002로 완화
            if rsi < 60:  # RSI 조건 완화
                confidence = min(0.8, abs(ml_pred) * 60)
                signal['action'] = 'BUY'
                signal['strategy_used'] = 'scalping'
        
        elif ml_pred < -0.002:
            if rsi > 40:
                confidence = min(0.8, abs(ml_pred) * 60)
                signal['action'] = 'SELL'
                signal['strategy_used'] = 'scalping'
        
        signal['confidence'] = confidence
    
    # 전략 3: 변동성 돌파 (조건 완화)
    elif market_condition == 'volatile':
        confidence = 0.0
        
        if abs(ml_pred) > 0.005:  # ML 예측 조건만 확인
            if volume_ratio > 1.0:  # 거래량 조건 완화
                confidence = min(0.7, abs(ml_pred) * 30)
                if ml_pred > 0:
                    signal['action'] = 'BUY'
                else:
                    signal['action'] = 'SELL'
                signal['strategy_used'] = 'volatility_breakout'
        
        signal['confidence'] = confidence
    
    return signal

def run_improved_backtest(df, predictions, min_confidence=0.4):  # 신뢰도 임계값 대폭 완화
    """개선된 백테스트 실행"""
    print(f"💰 개선된 백테스트 실행 중... (최소 신뢰도: {min_confidence})")
    
    initial_capital = 10000000
    capital = initial_capital
    position = 0  # 0: 현금, 1: 보유, -1: 공매도
    shares = 0
    trades = []
    portfolio_values = []
    
    # 신호 통계
    signal_count = 0
    executed_trades = 0
    
    strategy_stats = {
        'trend_following': {'count': 0, 'profit': 0},
        'scalping': {'count': 0, 'profit': 0},
        'volatility_breakout': {'count': 0, 'profit': 0}
    }
    
    print("📊 거래 진행 상황:")
    print("-" * 60)
    
    for i, (_, row) in enumerate(df.iterrows()):
        current_price = row['close']
        ml_pred = predictions[i] if i < len(predictions) else 0.0
        
        # 시장 상황 분석
        market_condition = improved_market_analysis(row)
        
        # 트리플 콤보 신호 생성
        signal = improved_triple_combo_strategy(row, ml_pred, market_condition)
        
        # 신호 통계
        if signal['action'] != 'HOLD':
            signal_count += 1
            
            # 신호 로그 (처음 몇 개만)
            if signal_count <= 10:
                print(f"   신호 {signal_count}: {signal['action']} | 신뢰도: {signal['confidence']:.3f} | "
                      f"전략: {signal['strategy_used']} | ML: {ml_pred:+.4f}")
        
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
                executed_trades += 1
                
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'strategy': signal['strategy_used'],
                    'confidence': signal['confidence'],
                    'timestamp': row['timestamp'],
                    'ml_pred': ml_pred,
                    'market_condition': market_condition
                })
                
                print(f"   ✅ 매수 체결 #{executed_trades}: {current_price:.0f}원 "
                      f"({signal['strategy_used']}, 신뢰도: {signal['confidence']:.2f})")
                
            elif signal['action'] == 'SELL' and position >= 0:
                # 매도
                if position == 1:  # 보유 청산
                    capital = shares * current_price
                
                shares = capital / current_price
                position = -1
                capital = 0
                executed_trades += 1
                
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'strategy': signal['strategy_used'],
                    'confidence': signal['confidence'],
                    'timestamp': row['timestamp'],
                    'ml_pred': ml_pred,
                    'market_condition': market_condition
                })
                
                print(f"   ✅ 매도 체결 #{executed_trades}: {current_price:.0f}원 "
                      f"({signal['strategy_used']}, 신뢰도: {signal['confidence']:.2f})")
    
    # 최종 정산
    if position != 0:
        capital += shares * df['close'].iloc[-1] * position
    
    print(f"\n📊 신호 생성 통계:")
    print(f"   총 신호 발생: {signal_count}개")
    print(f"   실행된 거래: {executed_trades}개")
    print(f"   신호 실행률: {executed_trades/signal_count*100 if signal_count > 0 else 0:.1f}%")
    
    print("✅ 개선된 백테스트 완료")
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'strategy_stats': strategy_stats,
        'signal_count': signal_count,
        'executed_trades': executed_trades
    }

def analyze_improved_results(results):
    """개선된 결과 분석"""
    print("\n" + "="*70)
    print("📊 개선된 트리플 콤보 백테스트 결과")
    print("="*70)
    
    initial = results['initial_capital']
    final = results['final_capital']
    total_return = (final - initial) / initial * 100
    
    print(f"💰 초기 자본: {initial:,.0f}원")
    print(f"💰 최종 자본: {final:,.0f}원")
    print(f"📈 총 수익률: {total_return:+.2f}%")
    print(f"💵 절대 수익: {final - initial:+,.0f}원")
    
    # 거래 분석
    trades = results['trades']
    print(f"\n📊 거래 통계:")
    print(f"   신호 발생: {results['signal_count']}개")
    print(f"   실행 거래: {results['executed_trades']}개")
    print(f"   실행률: {results['executed_trades']/results['signal_count']*100 if results['signal_count'] > 0 else 0:.1f}%")
    
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
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
        
        # 최근 거래 내역
        print(f"\n📋 최근 거래 내역 (최대 5개):")
        for i, trade in enumerate(trades[-5:]):
            print(f"   {i+1}. {trade['type']} @ {trade['price']:.0f}원 "
                  f"({trade['strategy']}, 신뢰도: {trade['confidence']:.2f})")
    
    # 성과 등급
    if total_return > 20:
        grade = "A+ (탁월)"
    elif total_return > 10:
        grade = "A (우수)"
    elif total_return > 5:
        grade = "B+ (양호)"
    elif total_return > 0:
        grade = "B (보통)"
    elif total_return > -5:
        grade = "C (개선 필요)"
    else:
        grade = "D (부족)"
    
    print(f"\n🏆 성과 등급: {grade}")
    
    if total_return > 0:
        print("✅ 수익성 있는 전략입니다!")
    elif results['executed_trades'] > 0:
        print("⚠️ 거래는 발생했으나 손실이 발생했습니다.")
    else:
        print("❌ 거래가 발생하지 않았습니다.")
    
    print("="*70)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='개선된 트리플 콤보 백테스트')
    parser.add_argument('--days', type=int, default=30, help='백테스트 기간 (기본: 30일)')
    parser.add_argument('--min-confidence', type=float, default=0.4, help='최소 신뢰도 (기본: 0.4)')
    
    args = parser.parse_args()
    
    print("🚀 개선된 트리플 콤보 백테스트 시작")
    print(f"📅 기간: {args.days}일")
    print(f"🎯 최소 신뢰도: {args.min_confidence} (완화됨)")
    print(f"🔧 개선사항: 신호 생성 조건 완화, ML 예측 강화")
    print()
    
    try:
        # 1. 개선된 데이터 생성
        df = generate_test_data(args.days)
        
        # 2. 기술적 지표 계산
        df = calculate_indicators(df)
        
        # 3. 향상된 ML 예측
        predictions = enhanced_ml_prediction(df)
        
        # 4. 개선된 백테스트 실행
        results = run_improved_backtest(df, predictions, args.min_confidence)
        
        # 5. 결과 분석
        analyze_improved_results(results)
        
        print(f"\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 개선된 백테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()