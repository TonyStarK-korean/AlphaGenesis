<<<<<<< HEAD
#!/usr/bin/env python3
"""
🚀 트리플 콤보 백테스트 실행 스크립트
ML 신뢰도 극대화 + 3가지 전략 조합으로 2025년 6월 백테스트
"""

import sys
import os
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

# 기존 모듈들 임포트
from run_ml_backtest import (
    PricePredictionModel, make_features, generate_crypto_features, 
    generate_advanced_features, detect_market_condition_simple,
    generate_historical_data, setup_logging
)

# 트리플 콤보 전략 임포트
try:
    from triple_combo_strategy import (
        TripleComboStrategy, print_detailed_trade_log, 
        check_position_exit, calculate_pnl
    )
    TRIPLE_COMBO_AVAILABLE = True
    print("🚀 트리플 콤보 전략 모듈 로드 성공!")
except ImportError as e:
    print(f"❌ 트리플 콤보 전략 모듈 로드 실패: {e}")
    TRIPLE_COMBO_AVAILABLE = False

warnings.filterwarnings('ignore')

def generate_june_2025_data():
    """2025년 6월 시뮬레이션 데이터 생성"""
    try:
        print("📊 2025년 6월 데이터 생성 중...")
        
        # 2025년 6월 1일 ~ 30일 (30일 * 24시간 = 720개 캔들)
        start_date = datetime(2025, 6, 1)
        end_date = datetime(2025, 6, 30, 23, 0, 0)
        
        # 시간 인덱스 생성
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # 비트코인 가격 시뮬레이션 (더 현실적인 패턴)
        np.random.seed(42)  # 재현 가능한 결과
        
        # 초기 가격 설정
        initial_price = 70000.0  # 2025년 예상 BTC 가격
        
        # 다양한 시장 국면 시뮬레이션
        market_phases = np.random.choice(['trending_up', 'trending_down', 'sideways', 'volatile'], 
                                       size=len(date_range)//24, 
                                       p=[0.3, 0.2, 0.3, 0.2])
        
        # 각 국면별 가격 생성
        prices = []
        current_price = initial_price
        
        for day in range(len(date_range)//24):
            phase = market_phases[day] if day < len(market_phases) else 'sideways'
            
            for hour in range(24):
                if phase == 'trending_up':
                    # 상승 추세: 평균 +0.5%, 변동성 2%
                    change = np.random.normal(0.005, 0.02)
                elif phase == 'trending_down':
                    # 하락 추세: 평균 -0.3%, 변동성 2.5%
                    change = np.random.normal(-0.003, 0.025)
                elif phase == 'sideways':
                    # 횡보: 평균 0%, 변동성 1%
                    change = np.random.normal(0, 0.01)
                else:  # volatile
                    # 변동성: 평균 0%, 변동성 4%
                    change = np.random.normal(0, 0.04)
                
                current_price *= (1 + change)
                prices.append(current_price)
        
        # 나머지 시간 채우기
        while len(prices) < len(date_range):
            change = np.random.normal(0, 0.02)
            current_price *= (1 + change)
            prices.append(current_price)
        
        prices = np.array(prices[:len(date_range)])
        
        # OHLCV 데이터 생성
        data = []
        for i in range(len(date_range)):
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
            
            # 거래량 생성 (변동성과 연관)
            base_volume = 1000 + np.random.exponential(2000)
            if abs(close_price - open_price) / open_price > 0.02:
                base_volume *= np.random.uniform(1.5, 3.0)  # 변동성 클 때 거래량 증가
            
            data.append({
                'datetime': date_range[i],
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': base_volume
            })
        
        df = pd.DataFrame(data)
        
        # 추가 정보
        df['timestamp'] = df['datetime'].astype('int64') // 10**9
        
        print(f"   ✅ 생성 완료: {len(df)}개 캔들")
        print(f"   📊 가격 범위: {df['close'].min():.0f} ~ {df['close'].max():.0f}")
        print(f"   📈 평균 가격: {df['close'].mean():.0f}")
        print(f"   📊 평균 거래량: {df['volume'].mean():.0f}")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 생성 오류: {e}")
        return generate_historical_data(years=1).tail(720)


def execute_triple_combo_backtest_with_logs(df, strategy, model):
    """상세 로그를 포함한 트리플 콤보 백테스트 실행"""
    try:
        # 백테스트 설정
        initial_capital = 10000000
        commission_rate = 0.0004
        slippage_rate = 0.0002
        
        # 포지션 관리
        capital = initial_capital
        position = 0  # 0: 중립, 1: 롱, -1: 숏
        position_size = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        current_leverage = 1.0
        entry_time = None
        
        # 결과 추적
        trades = []
        equity_curve = []
        daily_pnl = []
        
        # 전략별 성과 추적
        strategy_performance = {
            'trend': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_volume': 0},
            'scalping': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_volume': 0},
            'breakout': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_volume': 0}
        }
        
        print(f"\n🎯 백테스트 실행 (총 {len(df)}개 캔들)")
        print("=" * 80)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                current_time = row.get('datetime', f"캔들_{idx}")
                current_price = row['close']
                
                # 진행률 표시
                if idx % 100 == 0:
                    progress = (idx / len(df)) * 100
                    print(f"📊 진행률: {progress:.1f}% | 현재가: {current_price:.0f} | 자본: {capital:,.0f}")
                
                # ML 예측 수행
                ml_pred = 0
                if model and model.is_fitted:
                    try:
                        pred_result = model.predict(pd.DataFrame([row]))
                        ml_pred = pred_result[0] if len(pred_result) > 0 else 0
                    except:
                        ml_pred = 0
                
                # 포지션 관리 (기존 포지션 청산 확인)
                if position != 0:
                    should_close, close_reason = check_position_exit(
                        row, position, entry_price, stop_loss, take_profit
                    )
                    
                    if should_close:
                        # 포지션 청산
                        exit_price = current_price
                        exit_time = current_time
                        
                        # 손익 계산
                        pnl = calculate_pnl(position, entry_price, exit_price, position_size, current_leverage)
                        
                        # 수수료 및 슬리피지
                        commission = abs(position_size) * commission_rate
                        slippage = abs(position_size) * slippage_rate
                        net_pnl = pnl - commission - slippage
                        
                        capital += net_pnl
                        
                        # 거래 기록
                        trade_record = {
                            'trade_id': len(trades) + 1,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'size': position_size,
                            'leverage': current_leverage,
                            'pnl': pnl,
                            'commission': commission,
                            'slippage': slippage,
                            'net_pnl': net_pnl,
                            'reason': close_reason,
                            'strategy': getattr(strategy, 'last_strategy', 'unknown'),
                            'ml_pred': ml_pred,
                            'duration_hours': 1  # 시간 단위 거래
                        }
                        
                        trades.append(trade_record)
                        
                        # 전략별 성과 업데이트
                        strategy_name = trade_record['strategy']
                        if strategy_name in strategy_performance:
                            perf = strategy_performance[strategy_name]
                            perf['trades'] += 1
                            perf['total_pnl'] += net_pnl
                            perf['total_volume'] += abs(position_size)
                            if net_pnl > 0:
                                perf['wins'] += 1
                        
                        # 상세 거래 로그 출력
                        print_detailed_trade_log(trade_record)
                        
                        # 포지션 초기화
                        position = 0
                        position_size = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                        current_leverage = 1.0
                        entry_time = None
                
                # 새 신호 확인 (포지션이 없을 때만)
                if position == 0:
                    market_condition = detect_market_condition_simple(
                        df['close'].iloc[max(0, idx-20):idx+1]
                    )
                    
                    signal = strategy.generate_signal(
                        row, ml_pred, market_condition, 
                        df.iloc[max(0, idx-50):idx+1]
                    )
                    
                    if signal['signal'] != 0 and signal['confidence'] >= 0.6:
                        # 리스크 관리
                        risk_capital = capital * 0.02  # 거래당 2% 리스크
                        leverage = min(signal['leverage_suggestion'], 5.0)
                        
                        # 포지션 크기 계산
                        position_size = risk_capital * leverage
                        
                        # 포지션 진입
                        position = signal['signal']
                        entry_price = current_price
                        entry_time = current_time
                        stop_loss = signal['stop_loss']
                        take_profit = signal['take_profit']
                        current_leverage = leverage
                        
                        # 진입 로그
                        print(f"\n🎯 신호 발생!")
                        print(f"   ⏰ 시간: {current_time}")
                        print(f"   🎯 전략: {signal.get('strategy', 'unknown')}")
                        print(f"   📍 포지션: {'롱' if position == 1 else '숏'}")
                        print(f"   💰 진입가: {entry_price:.2f}")
                        print(f"   🛑 손절가: {stop_loss:.2f}")
                        print(f"   🎯 익절가: {take_profit:.2f}")
                        print(f"   ⚖️  레버리지: {leverage:.1f}x")
                        print(f"   🎲 신뢰도: {signal['confidence']:.2f}")
                        print(f"   🤖 ML 예측: {ml_pred:.4f}")
                
                # 자본 곡선 업데이트
                current_equity = capital
                unrealized_pnl = 0
                if position != 0:
                    unrealized_pnl = calculate_pnl(
                        position, entry_price, current_price, position_size, current_leverage
                    )
                    current_equity += unrealized_pnl
                
                equity_curve.append({
                    'datetime': current_time,
                    'equity': current_equity,
                    'position': position,
                    'price': current_price,
                    'unrealized_pnl': unrealized_pnl
                })
                
                # 일일 수익률 계산
                if len(equity_curve) > 1:
                    prev_equity = equity_curve[-2]['equity']
                    daily_return = (current_equity - prev_equity) / prev_equity
                    daily_pnl.append(daily_return)
                
            except Exception as e:
                print(f"   ⚠️  행 처리 오류 (idx={idx}): {e}")
                continue
        
        # 최종 포지션 강제 청산
        if position != 0:
            final_row = df.iloc[-1]
            exit_price = final_row['close']
            pnl = calculate_pnl(position, entry_price, exit_price, position_size, current_leverage)
            capital += pnl
            
            # 최종 거래 기록
            final_trade = {
                'trade_id': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': final_row.get('datetime', 'final'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'size': position_size,
                'leverage': current_leverage,
                'pnl': pnl,
                'commission': 0,
                'slippage': 0,
                'net_pnl': pnl,
                'reason': 'final_close',
                'strategy': getattr(strategy, 'last_strategy', 'unknown'),
                'ml_pred': 0,
                'duration_hours': 1
            }
            trades.append(final_trade)
            print_detailed_trade_log(final_trade)
        
        # 결과 계산
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # 성과 지표
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['net_pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        if total_trades > 0:
            profits = [t['net_pnl'] for t in trades if t['net_pnl'] > 0]
            losses = [t['net_pnl'] for t in trades if t['net_pnl'] < 0]
            
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0
        
        # 샤프 비율
        if len(daily_pnl) > 0:
            sharpe_ratio = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(24*30) if np.std(daily_pnl) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 최대 드로우다운
        equity_values = [e['equity'] for e in equity_curve]
        if len(equity_values) > 0:
            peak = equity_values[0]
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # 결과 패키징
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve,
            'strategy_performance': strategy_performance,
            'daily_pnl': daily_pnl
        }
        
        return results
        
    except Exception as e:
        print(f"❌ 백테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def print_triple_combo_results(results):
    """트리플 콤보 백테스트 결과 출력"""
    if 'error' in results:
        print(f"❌ 백테스트 실패: {results['error']}")
        return
    
    print(f"\n{'='*80}")
    print(f"🎉 트리플 콤보 백테스트 완료!")
    print(f"{'='*80}")
    
    # 기본 성과 지표
    print(f"💰 초기 자본: {results['initial_capital']:,.0f}원")
    print(f"💰 최종 자본: {results['final_capital']:,.0f}원")
    print(f"📈 총 수익률: {results['total_return']:.2%}")
    print(f"💵 순이익: {results['final_capital'] - results['initial_capital']:,.0f}원")
    
    print(f"\n📊 거래 통계:")
    print(f"🎯 총 거래 수: {results['total_trades']}건")
    print(f"✅ 승리 거래: {results['winning_trades']}건")
    print(f"❌ 패배 거래: {results['total_trades'] - results['winning_trades']}건")
    print(f"📊 승률: {results['win_rate']:.2%}")
    
    print(f"\n💰 손익 분석:")
    print(f"📈 평균 수익: {results['avg_win']:,.0f}원")
    print(f"📉 평균 손실: {results['avg_loss']:,.0f}원")
    print(f"⚖️  수익 팩터: {results['profit_factor']:.2f}")
    
    print(f"\n📊 리스크 지표:")
    print(f"📈 샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"📉 최대 드로우다운: {results['max_drawdown']:.2%}")
    
    # 전략별 성과
    print(f"\n🎯 전략별 성과:")
    print("=" * 60)
    for strategy_name, perf in results['strategy_performance'].items():
        if perf['trades'] > 0:
            win_rate = (perf['wins'] / perf['trades']) * 100
            avg_pnl = perf['total_pnl'] / perf['trades']
            print(f"📊 {strategy_name.upper():<12}: {perf['trades']:2d}건 | 승률 {win_rate:5.1f}% | 평균 {avg_pnl:8,.0f}원 | 총 {perf['total_pnl']:10,.0f}원")
    
    # 성과 등급 평가
    print(f"\n🏆 종합 평가:")
    grade = evaluate_performance_grade(results)
    print(f"   성과 등급: {grade['grade']}")
    print(f"   평가 점수: {grade['score']:.1f}/100")
    print(f"   핵심 강점: {grade['strengths']}")
    print(f"   개선 포인트: {grade['weaknesses']}")
    
    print(f"\n{'='*80}")


def evaluate_performance_grade(results):
    """성과 등급 평가"""
    score = 0
    strengths = []
    weaknesses = []
    
    # 수익률 평가 (30점)
    if results['total_return'] > 0.20:  # 20% 이상
        score += 30
        strengths.append("높은 수익률")
    elif results['total_return'] > 0.10:  # 10% 이상
        score += 20
        strengths.append("양호한 수익률")
    elif results['total_return'] > 0:  # 플러스 수익
        score += 10
        strengths.append("플러스 수익")
    else:
        weaknesses.append("손실 발생")
    
    # 승률 평가 (20점)
    if results['win_rate'] > 0.60:  # 60% 이상
        score += 20
        strengths.append("높은 승률")
    elif results['win_rate'] > 0.50:  # 50% 이상
        score += 15
        strengths.append("양호한 승률")
    elif results['win_rate'] > 0.40:  # 40% 이상
        score += 10
    else:
        weaknesses.append("낮은 승률")
    
    # 수익 팩터 평가 (20점)
    if results['profit_factor'] > 2.0:
        score += 20
        strengths.append("우수한 수익 팩터")
    elif results['profit_factor'] > 1.5:
        score += 15
        strengths.append("양호한 수익 팩터")
    elif results['profit_factor'] > 1.0:
        score += 10
    else:
        weaknesses.append("수익 팩터 부족")
    
    # 샤프 비율 평가 (15점)
    if results['sharpe_ratio'] > 1.5:
        score += 15
        strengths.append("높은 샤프 비율")
    elif results['sharpe_ratio'] > 1.0:
        score += 10
        strengths.append("양호한 샤프 비율")
    elif results['sharpe_ratio'] > 0.5:
        score += 5
    else:
        weaknesses.append("낮은 샤프 비율")
    
    # 드로우다운 평가 (15점)
    if results['max_drawdown'] < 0.05:  # 5% 미만
        score += 15
        strengths.append("낮은 드로우다운")
    elif results['max_drawdown'] < 0.10:  # 10% 미만
        score += 10
        strengths.append("관리 가능한 드로우다운")
    elif results['max_drawdown'] < 0.15:  # 15% 미만
        score += 5
    else:
        weaknesses.append("높은 드로우다운")
    
    # 등급 결정
    if score >= 90:
        grade = "A+ (탁월)"
    elif score >= 80:
        grade = "A (우수)"
    elif score >= 70:
        grade = "B+ (양호)"
    elif score >= 60:
        grade = "B (보통)"
    elif score >= 50:
        grade = "C+ (개선 필요)"
    elif score >= 40:
        grade = "C (미흡)"
    else:
        grade = "D (부족)"
    
    return {
        'grade': grade,
        'score': score,
        'strengths': ', '.join(strengths) if strengths else "없음",
        'weaknesses': ', '.join(weaknesses) if weaknesses else "없음"
    }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='🚀 트리플 콤보 백테스트 실행')
    parser.add_argument('--initial-capital', type=float, default=10000000, help='초기 자본 (기본값: 10,000,000원)')
    parser.add_argument('--ml-features', type=int, default=40, help='ML 모델에 사용할 피처 수 (기본값: 40)')
    parser.add_argument('--min-confidence', type=float, default=0.6, help='최소 신뢰도 임계값 (기본값: 0.6)')
    parser.add_argument('--verbose', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 트리플 콤보 백테스트 시작!")
        print(f"📅 기간: 2025년 6월 1일 ~ 6월 30일")
        print(f"💰 초기 자본: {args.initial_capital:,.0f}원")
        print(f"🎯 목표: 3가지 전략 조합으로 모든 시장 상황 대응")
        print(f"{'='*80}")
        
        if not TRIPLE_COMBO_AVAILABLE:
            print("❌ 트리플 콤보 전략을 사용할 수 없습니다.")
            print("   triple_combo_strategy.py 파일이 있는지 확인해주세요.")
            return
        
        # 1. 데이터 생성
        df = generate_june_2025_data()
        
        # 2. 피처 생성 (모든 고급 피처 포함)
        print("🔧 고급 피처 생성 중...")
        df = make_features(df)
        df = generate_crypto_features(df)
        df = generate_advanced_features(df)
        print(f"   ✅ 총 피처 수: {len(df.columns)}개")
        
        # 3. ML 모델 훈련
        print("🤖 강화된 ML 모델 훈련 중...")
        model = PricePredictionModel(n_splits=5)  # top_n_features 파라미터 제거
        model.fit(df)
        
        # 4. 트리플 콤보 전략 초기화
        print("🎯 트리플 콤보 전략 초기화...")
        strategy = TripleComboStrategy({
            'min_confidence': args.min_confidence,
            'trend_priority': 0.4,
            'scalping_priority': 0.35,
            'breakout_priority': 0.25
        })
        
        # 5. 백테스트 실행
        print("📈 백테스트 실행 중...")
        results = execute_triple_combo_backtest_with_logs(df, strategy, model)
        
        # 6. 결과 분석 및 출력
        print_triple_combo_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ 트리플 콤보 백테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 로깅 설정
    setup_logging()
    
    # 메인 실행
    results = main()
    
    if results and 'error' not in results:
        print(f"\n🎉 백테스트 성공적으로 완료!")
        print(f"   최종 수익률: {results['total_return']:.2%}")
        print(f"   총 거래 수: {results['total_trades']}건")
        print(f"   승률: {results['win_rate']:.2%}")
    else:
=======
#!/usr/bin/env python3
"""
🚀 트리플 콤보 백테스트 실행 스크립트
ML 신뢰도 극대화 + 3가지 전략 조합으로 2025년 6월 백테스트
"""

import sys
import os
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

# 기존 모듈들 임포트
from run_ml_backtest import (
    PricePredictionModel, make_features, generate_crypto_features, 
    generate_advanced_features, detect_market_condition_simple,
    generate_historical_data, setup_logging
)

# 트리플 콤보 전략 임포트
try:
    from triple_combo_strategy import (
        TripleComboStrategy, print_detailed_trade_log, 
        check_position_exit, calculate_pnl
    )
    TRIPLE_COMBO_AVAILABLE = True
    print("🚀 트리플 콤보 전략 모듈 로드 성공!")
except ImportError as e:
    print(f"❌ 트리플 콤보 전략 모듈 로드 실패: {e}")
    TRIPLE_COMBO_AVAILABLE = False

warnings.filterwarnings('ignore')

def generate_june_2025_data():
    """2025년 6월 시뮬레이션 데이터 생성"""
    try:
        print("📊 2025년 6월 데이터 생성 중...")
        
        # 2025년 6월 1일 ~ 30일 (30일 * 24시간 = 720개 캔들)
        start_date = datetime(2025, 6, 1)
        end_date = datetime(2025, 6, 30, 23, 0, 0)
        
        # 시간 인덱스 생성
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # 비트코인 가격 시뮬레이션 (더 현실적인 패턴)
        np.random.seed(42)  # 재현 가능한 결과
        
        # 초기 가격 설정
        initial_price = 70000.0  # 2025년 예상 BTC 가격
        
        # 다양한 시장 국면 시뮬레이션
        market_phases = np.random.choice(['trending_up', 'trending_down', 'sideways', 'volatile'], 
                                       size=len(date_range)//24, 
                                       p=[0.3, 0.2, 0.3, 0.2])
        
        # 각 국면별 가격 생성
        prices = []
        current_price = initial_price
        
        for day in range(len(date_range)//24):
            phase = market_phases[day] if day < len(market_phases) else 'sideways'
            
            for hour in range(24):
                if phase == 'trending_up':
                    # 상승 추세: 평균 +0.5%, 변동성 2%
                    change = np.random.normal(0.005, 0.02)
                elif phase == 'trending_down':
                    # 하락 추세: 평균 -0.3%, 변동성 2.5%
                    change = np.random.normal(-0.003, 0.025)
                elif phase == 'sideways':
                    # 횡보: 평균 0%, 변동성 1%
                    change = np.random.normal(0, 0.01)
                else:  # volatile
                    # 변동성: 평균 0%, 변동성 4%
                    change = np.random.normal(0, 0.04)
                
                current_price *= (1 + change)
                prices.append(current_price)
        
        # 나머지 시간 채우기
        while len(prices) < len(date_range):
            change = np.random.normal(0, 0.02)
            current_price *= (1 + change)
            prices.append(current_price)
        
        prices = np.array(prices[:len(date_range)])
        
        # OHLCV 데이터 생성
        data = []
        for i in range(len(date_range)):
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
            
            # 거래량 생성 (변동성과 연관)
            base_volume = 1000 + np.random.exponential(2000)
            if abs(close_price - open_price) / open_price > 0.02:
                base_volume *= np.random.uniform(1.5, 3.0)  # 변동성 클 때 거래량 증가
            
            data.append({
                'datetime': date_range[i],
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': base_volume
            })
        
        df = pd.DataFrame(data)
        
        # 추가 정보
        df['timestamp'] = df['datetime'].astype('int64') // 10**9
        
        print(f"   ✅ 생성 완료: {len(df)}개 캔들")
        print(f"   📊 가격 범위: {df['close'].min():.0f} ~ {df['close'].max():.0f}")
        print(f"   📈 평균 가격: {df['close'].mean():.0f}")
        print(f"   📊 평균 거래량: {df['volume'].mean():.0f}")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 생성 오류: {e}")
        return generate_historical_data(years=1).tail(720)


def execute_triple_combo_backtest_with_logs(df, strategy, model):
    """상세 로그를 포함한 트리플 콤보 백테스트 실행"""
    try:
        # 백테스트 설정
        initial_capital = 10000000
        commission_rate = 0.0004
        slippage_rate = 0.0002
        
        # 포지션 관리
        capital = initial_capital
        position = 0  # 0: 중립, 1: 롱, -1: 숏
        position_size = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        current_leverage = 1.0
        entry_time = None
        
        # 결과 추적
        trades = []
        equity_curve = []
        daily_pnl = []
        
        # 전략별 성과 추적
        strategy_performance = {
            'trend': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_volume': 0},
            'scalping': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_volume': 0},
            'breakout': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_volume': 0}
        }
        
        print(f"\n🎯 백테스트 실행 (총 {len(df)}개 캔들)")
        print("=" * 80)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                current_time = row.get('datetime', f"캔들_{idx}")
                current_price = row['close']
                
                # 진행률 표시
                if idx % 100 == 0:
                    progress = (idx / len(df)) * 100
                    print(f"📊 진행률: {progress:.1f}% | 현재가: {current_price:.0f} | 자본: {capital:,.0f}")
                
                # ML 예측 수행
                ml_pred = 0
                if model and model.is_fitted:
                    try:
                        pred_result = model.predict(pd.DataFrame([row]))
                        ml_pred = pred_result[0] if len(pred_result) > 0 else 0
                    except:
                        ml_pred = 0
                
                # 포지션 관리 (기존 포지션 청산 확인)
                if position != 0:
                    should_close, close_reason = check_position_exit(
                        row, position, entry_price, stop_loss, take_profit
                    )
                    
                    if should_close:
                        # 포지션 청산
                        exit_price = current_price
                        exit_time = current_time
                        
                        # 손익 계산
                        pnl = calculate_pnl(position, entry_price, exit_price, position_size, current_leverage)
                        
                        # 수수료 및 슬리피지
                        commission = abs(position_size) * commission_rate
                        slippage = abs(position_size) * slippage_rate
                        net_pnl = pnl - commission - slippage
                        
                        capital += net_pnl
                        
                        # 거래 기록
                        trade_record = {
                            'trade_id': len(trades) + 1,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'size': position_size,
                            'leverage': current_leverage,
                            'pnl': pnl,
                            'commission': commission,
                            'slippage': slippage,
                            'net_pnl': net_pnl,
                            'reason': close_reason,
                            'strategy': getattr(strategy, 'last_strategy', 'unknown'),
                            'ml_pred': ml_pred,
                            'duration_hours': 1  # 시간 단위 거래
                        }
                        
                        trades.append(trade_record)
                        
                        # 전략별 성과 업데이트
                        strategy_name = trade_record['strategy']
                        if strategy_name in strategy_performance:
                            perf = strategy_performance[strategy_name]
                            perf['trades'] += 1
                            perf['total_pnl'] += net_pnl
                            perf['total_volume'] += abs(position_size)
                            if net_pnl > 0:
                                perf['wins'] += 1
                        
                        # 상세 거래 로그 출력
                        print_detailed_trade_log(trade_record)
                        
                        # 포지션 초기화
                        position = 0
                        position_size = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                        current_leverage = 1.0
                        entry_time = None
                
                # 새 신호 확인 (포지션이 없을 때만)
                if position == 0:
                    market_condition = detect_market_condition_simple(
                        df['close'].iloc[max(0, idx-20):idx+1]
                    )
                    
                    signal = strategy.generate_signal(
                        row, ml_pred, market_condition, 
                        df.iloc[max(0, idx-50):idx+1]
                    )
                    
                    if signal['signal'] != 0 and signal['confidence'] >= 0.6:
                        # 리스크 관리
                        risk_capital = capital * 0.02  # 거래당 2% 리스크
                        leverage = min(signal['leverage_suggestion'], 5.0)
                        
                        # 포지션 크기 계산
                        position_size = risk_capital * leverage
                        
                        # 포지션 진입
                        position = signal['signal']
                        entry_price = current_price
                        entry_time = current_time
                        stop_loss = signal['stop_loss']
                        take_profit = signal['take_profit']
                        current_leverage = leverage
                        
                        # 진입 로그
                        print(f"\n🎯 신호 발생!")
                        print(f"   ⏰ 시간: {current_time}")
                        print(f"   🎯 전략: {signal.get('strategy', 'unknown')}")
                        print(f"   📍 포지션: {'롱' if position == 1 else '숏'}")
                        print(f"   💰 진입가: {entry_price:.2f}")
                        print(f"   🛑 손절가: {stop_loss:.2f}")
                        print(f"   🎯 익절가: {take_profit:.2f}")
                        print(f"   ⚖️  레버리지: {leverage:.1f}x")
                        print(f"   🎲 신뢰도: {signal['confidence']:.2f}")
                        print(f"   🤖 ML 예측: {ml_pred:.4f}")
                
                # 자본 곡선 업데이트
                current_equity = capital
                unrealized_pnl = 0
                if position != 0:
                    unrealized_pnl = calculate_pnl(
                        position, entry_price, current_price, position_size, current_leverage
                    )
                    current_equity += unrealized_pnl
                
                equity_curve.append({
                    'datetime': current_time,
                    'equity': current_equity,
                    'position': position,
                    'price': current_price,
                    'unrealized_pnl': unrealized_pnl
                })
                
                # 일일 수익률 계산
                if len(equity_curve) > 1:
                    prev_equity = equity_curve[-2]['equity']
                    daily_return = (current_equity - prev_equity) / prev_equity
                    daily_pnl.append(daily_return)
                
            except Exception as e:
                print(f"   ⚠️  행 처리 오류 (idx={idx}): {e}")
                continue
        
        # 최종 포지션 강제 청산
        if position != 0:
            final_row = df.iloc[-1]
            exit_price = final_row['close']
            pnl = calculate_pnl(position, entry_price, exit_price, position_size, current_leverage)
            capital += pnl
            
            # 최종 거래 기록
            final_trade = {
                'trade_id': len(trades) + 1,
                'entry_time': entry_time,
                'exit_time': final_row.get('datetime', 'final'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'size': position_size,
                'leverage': current_leverage,
                'pnl': pnl,
                'commission': 0,
                'slippage': 0,
                'net_pnl': pnl,
                'reason': 'final_close',
                'strategy': getattr(strategy, 'last_strategy', 'unknown'),
                'ml_pred': 0,
                'duration_hours': 1
            }
            trades.append(final_trade)
            print_detailed_trade_log(final_trade)
        
        # 결과 계산
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # 성과 지표
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['net_pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        if total_trades > 0:
            profits = [t['net_pnl'] for t in trades if t['net_pnl'] > 0]
            losses = [t['net_pnl'] for t in trades if t['net_pnl'] < 0]
            
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0
        
        # 샤프 비율
        if len(daily_pnl) > 0:
            sharpe_ratio = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(24*30) if np.std(daily_pnl) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 최대 드로우다운
        equity_values = [e['equity'] for e in equity_curve]
        if len(equity_values) > 0:
            peak = equity_values[0]
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # 결과 패키징
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve,
            'strategy_performance': strategy_performance,
            'daily_pnl': daily_pnl
        }
        
        return results
        
    except Exception as e:
        print(f"❌ 백테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def print_triple_combo_results(results):
    """트리플 콤보 백테스트 결과 출력"""
    if 'error' in results:
        print(f"❌ 백테스트 실패: {results['error']}")
        return
    
    print(f"\n{'='*80}")
    print(f"🎉 트리플 콤보 백테스트 완료!")
    print(f"{'='*80}")
    
    # 기본 성과 지표
    print(f"💰 초기 자본: {results['initial_capital']:,.0f}원")
    print(f"💰 최종 자본: {results['final_capital']:,.0f}원")
    print(f"📈 총 수익률: {results['total_return']:.2%}")
    print(f"💵 순이익: {results['final_capital'] - results['initial_capital']:,.0f}원")
    
    print(f"\n📊 거래 통계:")
    print(f"🎯 총 거래 수: {results['total_trades']}건")
    print(f"✅ 승리 거래: {results['winning_trades']}건")
    print(f"❌ 패배 거래: {results['total_trades'] - results['winning_trades']}건")
    print(f"📊 승률: {results['win_rate']:.2%}")
    
    print(f"\n💰 손익 분석:")
    print(f"📈 평균 수익: {results['avg_win']:,.0f}원")
    print(f"📉 평균 손실: {results['avg_loss']:,.0f}원")
    print(f"⚖️  수익 팩터: {results['profit_factor']:.2f}")
    
    print(f"\n📊 리스크 지표:")
    print(f"📈 샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"📉 최대 드로우다운: {results['max_drawdown']:.2%}")
    
    # 전략별 성과
    print(f"\n🎯 전략별 성과:")
    print("=" * 60)
    for strategy_name, perf in results['strategy_performance'].items():
        if perf['trades'] > 0:
            win_rate = (perf['wins'] / perf['trades']) * 100
            avg_pnl = perf['total_pnl'] / perf['trades']
            print(f"📊 {strategy_name.upper():<12}: {perf['trades']:2d}건 | 승률 {win_rate:5.1f}% | 평균 {avg_pnl:8,.0f}원 | 총 {perf['total_pnl']:10,.0f}원")
    
    # 성과 등급 평가
    print(f"\n🏆 종합 평가:")
    grade = evaluate_performance_grade(results)
    print(f"   성과 등급: {grade['grade']}")
    print(f"   평가 점수: {grade['score']:.1f}/100")
    print(f"   핵심 강점: {grade['strengths']}")
    print(f"   개선 포인트: {grade['weaknesses']}")
    
    print(f"\n{'='*80}")


def evaluate_performance_grade(results):
    """성과 등급 평가"""
    score = 0
    strengths = []
    weaknesses = []
    
    # 수익률 평가 (30점)
    if results['total_return'] > 0.20:  # 20% 이상
        score += 30
        strengths.append("높은 수익률")
    elif results['total_return'] > 0.10:  # 10% 이상
        score += 20
        strengths.append("양호한 수익률")
    elif results['total_return'] > 0:  # 플러스 수익
        score += 10
        strengths.append("플러스 수익")
    else:
        weaknesses.append("손실 발생")
    
    # 승률 평가 (20점)
    if results['win_rate'] > 0.60:  # 60% 이상
        score += 20
        strengths.append("높은 승률")
    elif results['win_rate'] > 0.50:  # 50% 이상
        score += 15
        strengths.append("양호한 승률")
    elif results['win_rate'] > 0.40:  # 40% 이상
        score += 10
    else:
        weaknesses.append("낮은 승률")
    
    # 수익 팩터 평가 (20점)
    if results['profit_factor'] > 2.0:
        score += 20
        strengths.append("우수한 수익 팩터")
    elif results['profit_factor'] > 1.5:
        score += 15
        strengths.append("양호한 수익 팩터")
    elif results['profit_factor'] > 1.0:
        score += 10
    else:
        weaknesses.append("수익 팩터 부족")
    
    # 샤프 비율 평가 (15점)
    if results['sharpe_ratio'] > 1.5:
        score += 15
        strengths.append("높은 샤프 비율")
    elif results['sharpe_ratio'] > 1.0:
        score += 10
        strengths.append("양호한 샤프 비율")
    elif results['sharpe_ratio'] > 0.5:
        score += 5
    else:
        weaknesses.append("낮은 샤프 비율")
    
    # 드로우다운 평가 (15점)
    if results['max_drawdown'] < 0.05:  # 5% 미만
        score += 15
        strengths.append("낮은 드로우다운")
    elif results['max_drawdown'] < 0.10:  # 10% 미만
        score += 10
        strengths.append("관리 가능한 드로우다운")
    elif results['max_drawdown'] < 0.15:  # 15% 미만
        score += 5
    else:
        weaknesses.append("높은 드로우다운")
    
    # 등급 결정
    if score >= 90:
        grade = "A+ (탁월)"
    elif score >= 80:
        grade = "A (우수)"
    elif score >= 70:
        grade = "B+ (양호)"
    elif score >= 60:
        grade = "B (보통)"
    elif score >= 50:
        grade = "C+ (개선 필요)"
    elif score >= 40:
        grade = "C (미흡)"
    else:
        grade = "D (부족)"
    
    return {
        'grade': grade,
        'score': score,
        'strengths': ', '.join(strengths) if strengths else "없음",
        'weaknesses': ', '.join(weaknesses) if weaknesses else "없음"
    }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='🚀 트리플 콤보 백테스트 실행')
    parser.add_argument('--initial-capital', type=float, default=10000000, help='초기 자본 (기본값: 10,000,000원)')
    parser.add_argument('--ml-features', type=int, default=40, help='ML 모델에 사용할 피처 수 (기본값: 40)')
    parser.add_argument('--min-confidence', type=float, default=0.6, help='최소 신뢰도 임계값 (기본값: 0.6)')
    parser.add_argument('--verbose', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 트리플 콤보 백테스트 시작!")
        print(f"📅 기간: 2025년 6월 1일 ~ 6월 30일")
        print(f"💰 초기 자본: {args.initial_capital:,.0f}원")
        print(f"🎯 목표: 3가지 전략 조합으로 모든 시장 상황 대응")
        print(f"{'='*80}")
        
        if not TRIPLE_COMBO_AVAILABLE:
            print("❌ 트리플 콤보 전략을 사용할 수 없습니다.")
            print("   triple_combo_strategy.py 파일이 있는지 확인해주세요.")
            return
        
        # 1. 데이터 생성
        df = generate_june_2025_data()
        
        # 2. 피처 생성 (모든 고급 피처 포함)
        print("🔧 고급 피처 생성 중...")
        df = make_features(df)
        df = generate_crypto_features(df)
        df = generate_advanced_features(df)
        print(f"   ✅ 총 피처 수: {len(df.columns)}개")
        
        # 3. ML 모델 훈련
        print("🤖 강화된 ML 모델 훈련 중...")
        model = PricePredictionModel(top_n_features=args.ml_features)
        model.fit(df)
        
        # 4. 트리플 콤보 전략 초기화
        print("🎯 트리플 콤보 전략 초기화...")
        strategy = TripleComboStrategy({
            'min_confidence': args.min_confidence,
            'trend_priority': 0.4,
            'scalping_priority': 0.35,
            'breakout_priority': 0.25
        })
        
        # 5. 백테스트 실행
        print("📈 백테스트 실행 중...")
        results = execute_triple_combo_backtest_with_logs(df, strategy, model)
        
        # 6. 결과 분석 및 출력
        print_triple_combo_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ 트리플 콤보 백테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 로깅 설정
    setup_logging()
    
    # 메인 실행
    results = main()
    
    if results and 'error' not in results:
        print(f"\n🎉 백테스트 성공적으로 완료!")
        print(f"   최종 수익률: {results['total_return']:.2%}")
        print(f"   총 거래 수: {results['total_trades']}건")
        print(f"   승률: {results['win_rate']:.2%}")
    else:
>>>>>>> febb08c8d864666b98f9587b4eb4ce3a55eed692
        print(f"\n❌ 백테스트 실패") 