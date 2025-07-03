#!/usr/bin/env python3
"""
🚀 트리플 콤보 백테스트 실행기
간단한 명령으로 트리플 콤보 전략을 테스트할 수 있습니다.
"""

import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np

# 기존 모듈들 임포트
try:
    from run_ml_backtest import (
        PricePredictionModel, make_features, generate_crypto_features, 
        generate_advanced_features, detect_market_condition_simple,
        generate_historical_data
    )
    print("✅ 기존 ML 백테스트 모듈 로드 성공!")
except ImportError as e:
    print(f"❌ 기존 ML 백테스트 모듈 로드 실패: {e}")
    sys.exit(1)

# 트리플 콤보 전략 임포트
try:
    from triple_combo_strategy import (
        TripleComboStrategy, print_detailed_trade_log, 
        check_position_exit, calculate_pnl
    )
    print("✅ 트리플 콤보 전략 모듈 로드 성공!")
except ImportError as e:
    print(f"❌ 트리플 콤보 전략 모듈 로드 실패: {e}")
    print("   triple_combo_strategy.py 파일이 있는지 확인해주세요.")
    sys.exit(1)

warnings.filterwarnings('ignore')


def run_triple_combo_backtest():
    """트리플 콤보 백테스트 실행"""
    try:
        print(f"\n{'='*80}")
        print(f"🚀 트리플 콤보 백테스트 시작!")
        print(f"📅 기간: 2025년 6월 시뮬레이션")
        print(f"💰 초기 자본: 10,000,000원")
        print(f"🎯 3가지 전략 조합으로 모든 시장 상황 대응")
        print(f"{'='*80}")
        
        # 1. 데이터 생성 (720개 캔들 = 30일 * 24시간)
        print("📊 시뮬레이션 데이터 생성 중...")
        df = generate_historical_data(years=1)
        df = df.tail(720)  # 최신 720개 캔들 사용
        print(f"   ✅ 데이터 준비: {len(df)}개 캔들")
        
        # 2. 피처 생성
        print("🔧 고급 피처 생성 중...")
        df = make_features(df)
        df = generate_crypto_features(df)
        df = generate_advanced_features(df)
        print(f"   ✅ 총 피처 수: {len(df.columns)}개")
        
        # 3. ML 모델 훈련
        print("🤖 강화된 ML 모델 훈련 중...")
        model = PricePredictionModel(top_n_features=30)
        model.fit(df)
        
        # 4. 트리플 콤보 전략 초기화
        print("🎯 트리플 콤보 전략 초기화...")
        strategy = TripleComboStrategy({
            'min_confidence': 0.6,
            'trend_priority': 0.4,
            'scalping_priority': 0.35,
            'breakout_priority': 0.25
        })
        
        # 5. 백테스트 실행
        print("📈 백테스트 실행 중...")
        results = simple_backtest_execution(df, strategy, model)
        
        # 6. 결과 출력
        print_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ 백테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def simple_backtest_execution(df, strategy, model):
    """간단한 백테스트 실행"""
    try:
        # 백테스트 설정
        initial_capital = 10000000
        capital = initial_capital
        position = 0
        trades = []
        
        print(f"\n📊 백테스트 진행 중... (총 {len(df)}개 캔들)")
        
        for idx, (_, row) in enumerate(df.iterrows()):
            # 진행률 표시
            if idx % 100 == 0:
                progress = (idx / len(df)) * 100
                print(f"   진행률: {progress:.1f}%")
            
            # ML 예측
            ml_pred = 0
            if model and model.is_fitted:
                try:
                    pred_result = model.predict(pd.DataFrame([row]))
                    ml_pred = pred_result[0] if len(pred_result) > 0 else 0
                except:
                    ml_pred = 0
            
            # 신호 생성 (간단한 버전)
            if position == 0:  # 포지션이 없을 때만 새 신호 확인
                market_condition = detect_market_condition_simple(
                    df['close'].iloc[max(0, idx-20):idx+1]
                )
                
                signal = strategy.generate_signal(
                    row, ml_pred, market_condition, 
                    df.iloc[max(0, idx-20):idx+1]
                )
                
                if signal['signal'] != 0 and signal['confidence'] >= 0.6:
                    # 간단한 포지션 관리
                    position = signal['signal']
                    entry_price = row['close']
                    
                    # 거래 기록
                    trade_info = {
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'signal': signal['signal'],
                        'confidence': signal['confidence'],
                        'strategy': signal.get('strategy', 'unknown'),
                        'ml_pred': ml_pred
                    }
                    
                    print(f"\n🎯 신호 발생! [{signal.get('strategy', 'unknown')}]")
                    print(f"   포지션: {'롱' if position == 1 else '숏'}")
                    print(f"   신뢰도: {signal['confidence']:.2f}")
                    print(f"   진입가: {entry_price:.2f}")
                    
            else:
                # 간단한 청산 조건 (5캔들 후 자동 청산)
                if len(trades) > 0 and (idx - trades[-1]['entry_time']) >= 5:
                    exit_price = row['close']
                    
                    # 손익 계산
                    if position == 1:  # 롱
                        pnl_pct = (exit_price - trades[-1]['entry_price']) / trades[-1]['entry_price']
                    else:  # 숏
                        pnl_pct = (trades[-1]['entry_price'] - exit_price) / trades[-1]['entry_price']
                    
                    # 거래 완료
                    trades[-1]['exit_time'] = idx
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['pnl_pct'] = pnl_pct
                    trades[-1]['pnl'] = capital * 0.02 * pnl_pct  # 2% 포지션 크기
                    
                    capital += trades[-1]['pnl']
                    
                    print(f"   청산가: {exit_price:.2f}")
                    print(f"   손익: {pnl_pct:.2%} ({trades[-1]['pnl']:,.0f}원)")
                    print(f"   자본: {capital:,.0f}원")
                    
                    position = 0
            
            # 새 거래 시작 시 기록
            if position != 0 and len(trades) == 0:
                trades.append(trade_info)
            elif position != 0 and trades[-1].get('exit_time') is not None:
                trades.append(trade_info)
        
        # 최종 결과 계산
        total_return = (capital - initial_capital) / initial_capital
        completed_trades = [t for t in trades if 'exit_time' in t]
        
        if completed_trades:
            winning_trades = sum(1 for t in completed_trades if t['pnl'] > 0)
            win_rate = winning_trades / len(completed_trades)
        else:
            win_rate = 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': len(completed_trades),
            'win_rate': win_rate,
            'trades': completed_trades
        }
        
        return results
        
    except Exception as e:
        print(f"❌ 백테스트 실행 오류: {e}")
        return {'error': str(e)}


def print_results(results):
    """결과 출력"""
    if 'error' in results:
        print(f"❌ 백테스트 실패: {results['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"🎉 트리플 콤보 백테스트 결과")
    print(f"{'='*60}")
    
    print(f"💰 초기 자본: {results['initial_capital']:,.0f}원")
    print(f"💰 최종 자본: {results['final_capital']:,.0f}원")
    print(f"📈 총 수익률: {results['total_return']:.2%}")
    print(f"💵 순이익: {results['final_capital'] - results['initial_capital']:,.0f}원")
    
    print(f"\n📊 거래 통계:")
    print(f"🎯 총 거래 수: {results['total_trades']}건")
    print(f"📊 승률: {results['win_rate']:.2%}")
    
    # 전략별 분석
    if results['trades']:
        strategy_stats = {}
        for trade in results['trades']:
            strategy = trade.get('strategy', 'unknown')
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            
            strategy_stats[strategy]['count'] += 1
            if trade['pnl'] > 0:
                strategy_stats[strategy]['wins'] += 1
            strategy_stats[strategy]['total_pnl'] += trade['pnl']
        
        print(f"\n🎯 전략별 성과:")
        for strategy, stats in strategy_stats.items():
            win_rate = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
            print(f"   {strategy}: {stats['count']}건, 승률 {win_rate:.1f}%, 총 {stats['total_pnl']:,.0f}원")
    
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


if __name__ == "__main__":
    print("🚀 트리플 콤보 백테스트 실행기")
    print("=" * 60)
    
    # 실행
    results = run_triple_combo_backtest()
    
    if results and 'error' not in results:
        print(f"\n✅ 백테스트 완료!")
    else:
        print(f"\n❌ 백테스트 실패")
        
    input("\n아무 키나 누르면 종료합니다...") 