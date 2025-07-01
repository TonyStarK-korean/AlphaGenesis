#!/usr/bin/env python3
"""
🚀 AlphaGenesis-V3: 최종 통합 백테스트 실행기
모든 전략과 ML 모델, 동적 리스크 관리를 결합한 최종 시스템
"""

import sys
import os
import logging
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 필요한 모듈 임포트 ---
try:
    # 기능별로 분리된 모듈에서 함수 임포트
    from run_ml_backtest import (
        PricePredictionModel, make_features, generate_crypto_features,
        generate_advanced_features, setup_logging, generate_historical_data
    )
    from triple_combo_strategy import (
        TripleComboStrategy, calculate_dynamic_risk_settings,
        print_detailed_trade_log, check_position_exit, calculate_pnl,
        evaluate_performance_grade
    )
    MODULES_LOADED = True
    print("✅ 모든 필수 모듈 로드 성공!")
except ImportError as e:
    print(f"❌ 필수 모듈 로드 실패: {e}")
    print("   run_ml_backtest.py와 triple_combo_strategy.py 파일이 있는지 확인해주세요.")
    MODULES_LOADED = False

warnings.filterwarnings('ignore')


def run_v3_backtest(df: pd.DataFrame, initial_capital: float, model: PricePredictionModel, strategy_manager: TripleComboStrategy):
    """
    AlphaGenesis-V3 시스템의 핵심 백테스트 실행 함수
    """
    try:
        logger = logging.getLogger(__name__)
        
        # --- 1. 피처 엔지니어링 ---
        print("🔧 고급 피처 생성 중...")
        df_features = make_features(df.copy())
        df_features = generate_crypto_features(df_features)
        df_features = generate_advanced_features(df_features)
        df_features.dropna(inplace=True)
        print(f"   ✅ 피처 생성 완료: {len(df_features.columns)}개 피처")

        # --- 2. ML 예측 수행 ---
        print("🤖 ML 예측 생성 중...")
        if model and hasattr(model, 'is_fitted') and model.is_fitted:
            df_features['ml_prediction'] = model.predict(df_features)
        else:
            print("   ⚠️  훈련된 ML 모델이 없어, 예측 없이 진행합니다.")
            df_features['ml_prediction'] = 0.0
        print("   ✅ ML 예측 완료!")

        # --- 3. 백테스트 변수 초기화 ---
        capital = initial_capital
        position = 0
        position_info = {}
        trades = []
        equity_curve = [{'time': df_features.index[0], 'capital': capital}]

        print(f"\n📈 백테스트 실행 중 (총 {len(df_features)}개 캔들)...")
        
        # --- 4. 메인 백테스트 루프 ---
        for idx, row in tqdm(df_features.iterrows(), total=len(df_features), desc="AlphaGenesis-V3"):
            current_time = idx
            current_price = row['close']

            # --- 4a. 포지션 청산 확인 ---
            if position != 0:
                should_close, close_reason = check_position_exit(
                    row, position, position_info['entry_price'], 
                    position_info['stop_loss'], position_info['take_profit']
                )
                if should_close:
                    pnl = calculate_pnl(position, position_info['entry_price'], current_price, 
                                        position_info['size'], position_info['leverage'])
                    net_pnl = pnl - (abs(pnl) * 0.0006) # 수수료/슬리피지 근사치
                    capital += net_pnl
                    
                    trade_record = {**position_info, 'exit_time': current_time, 'exit_price': current_price, 'net_pnl': net_pnl, 'reason': close_reason}
                    trades.append(trade_record)
                    print_detailed_trade_log(trade_record)
                    
                    position = 0
                    position_info = {}

            # --- 4b. 신규 진입 신호 확인 ---
            if position == 0:
                ml_pred = row['ml_prediction']
                
                # 트리플 콤보 전략 매니저를 통해 신호 생성
                signal = strategy_manager.generate_signal(row, ml_pred, None, df_features.iloc[max(0, idx-50):idx+1])

                if signal['signal'] != 0:
                    # 동적 리스크 설정 계산
                    risk_settings = calculate_dynamic_risk_settings(
                        signal.get('market_phase', 'mixed'),
                        ml_pred,
                        signal['confidence']
                    )
                    
                    # 포지션 진입
                    position = signal['signal']
                    entry_price = current_price
                    
                    position_info = {
                        'entry_time': current_time,
                        'strategy': signal['strategy'],
                        'position': position,
                        'entry_price': entry_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'size': capital * risk_settings['position_size'],
                        'leverage': risk_settings['leverage'],
                        'confidence': signal['confidence'],
                        'ml_pred': ml_pred
                    }

            # --- 4c. 자본 곡선 기록 ---
            equity_curve.append({'time': current_time, 'capital': capital})

        # --- 5. 최종 결과 계산 ---
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # 성과 지표 계산
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['net_pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # ... (더 상세한 결과 계산 로직 추가 가능) ...

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trades': trades,
            'equity_curve': equity_curve
        }

    except Exception as e:
        logging.error(f"❌ V3 백테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def print_v3_results(results):
    """AlphaGenesis-V3 결과 출력"""
    if 'error' in results:
        print(f"❌ 백테스트 실패: {results['error']}")
        return

    print(f"\n{'='*80}")
    print(f"🎉 AlphaGenesis-V3 백테스트 완료!")
    print(f"{'='*80}")

    # 기본 성과 지표
    print(f"💰 초기 자본: {results['initial_capital']:,.0f}원")
    print(f"💰 최종 자본: {results['final_capital']:,.0f}원")
    print(f"📈 총 수익률: {results['total_return']:.2%}")
    print(f"💵 순이익: {results['final_capital'] - results['initial_capital']:,.0f}원")

    print(f"\n📊 거래 통계:")
    print(f"🎯 총 거래 수: {results['total_trades']}건")
    print(f"📊 승률: {results['win_rate']:.2%}")

    # 전략별 성과 분석
    strategy_stats = {}
    for trade in results['trades']:
        strategy_name = trade['strategy'].split('_')[-1] # 'Triple_Combo_trend' -> 'trend'
        if strategy_name not in strategy_stats:
            strategy_stats[strategy_name] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        
        stats = strategy_stats[strategy_name]
        stats['trades'] += 1
        stats['total_pnl'] += trade['net_pnl']
        if trade['net_pnl'] > 0:
            stats['wins'] += 1

    print(f"\n🎯 전략별 성과:")
    print("=" * 60)
    for strategy_name, stats in strategy_stats.items():
        if stats['trades'] > 0:
            win_rate = (stats['wins'] / stats['trades']) * 100
            avg_pnl = stats['total_pnl'] / stats['trades']
            print(f"📊 {strategy_name.upper():<12}: {stats['trades']:3d}건 | 승률 {win_rate:5.1f}% | 평균 {avg_pnl:8,.0f}원 | 총 {stats['total_pnl']:10,.0f}원")

    # 성과 등급 평가
    grade = evaluate_performance_grade(results)
    print(f"\n🏆 종합 평가:")
    print(f"   성과 등급: {grade['grade']}")
    print(f"   평가 점수: {grade['score']:.1f}/100")
    print(f"   핵심 강점: {grade['strengths']}")
    print(f"   개선 포인트: {grade['weaknesses']}")

    print(f"\n{'='*80}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='🚀 AlphaGenesis-V3 최종 백테스트 실행기')
    parser.add_argument('--data', type=str, default='data/market_data/BNB_USDT_1h.csv', help='백테스트용 데이터 파일 경로')
    parser.add_argument('--capital', type=float, default=10000000, help='초기 자본')
    parser.add_argument('--features', type=int, default=50, help='ML 모델에 사용할 상위 피처 수')
    
    args = parser.parse_args()

    if not MODULES_LOADED:
        sys.exit(1)

    # --- 1. 데이터 로드 ---
    try:
        print(f"💾 데이터 로드 중: {args.data}")
        df = pd.read_csv(args.data, index_col='timestamp', parse_dates=True)
        # 최근 1년 데이터만 사용 (테스트 속도 및 최신 경향 반영)
        df = df.last('1Y')
        print(f"   ✅ 데이터 로드 완료: {len(df)}개 캔들")
    except FileNotFoundError:
        print(f"   ⚠️  데이터 파일({args.data})을 찾을 수 없어, 시뮬레이션 데이터를 생성합니다.")
        df = generate_historical_data(years=1)
        df.set_index('timestamp', inplace=True)

    # --- 2. ML 모델 훈련 ---
    print("\n🤖 강화된 ML 모델 훈련 중...")
    model = PricePredictionModel(top_n_features=args.features)
    # 훈련을 위해 모든 피처가 포함된 데이터프레임 생성
    df_for_training = make_features(df.copy())
    df_for_training = generate_crypto_features(df_for_training)
    df_for_training = generate_advanced_features(df_for_training)
    model.fit(df_for_training)

    # --- 3. 트리플 콤보 전략 초기화 ---
    print("\n🎯 트리플 콤보 전략 초기화...")
    strategy_manager = TripleComboStrategy()

    # --- 4. 최종 백테스트 실행 ---
    results = run_v3_backtest(df, args.capital, model, strategy_manager)

    # --- 5. 결과 출력 ---
    print_v3_results(results)

if __name__ == "__main__":
    # 로깅 설정
    setup_logging()
    
    # 메인 실행
    main()