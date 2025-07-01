#!/usr/bin/env python3
"""
🔬 워크-포워드 최적화 개선 시스템
다양한 훈련 기간을 테스트하여 최적의 학습 기간을 찾는 시스템
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 기존 모듈 임포트
try:
    from run_ml_backtest import (
        PricePredictionModel, make_features, generate_crypto_features, 
        generate_advanced_features, generate_historical_data, run_crypto_backtest
    )
    
    # Optuna 임포트 시도
    try:
        import optuna
        OPTUNA_AVAILABLE = True
    except ImportError:
        OPTUNA_AVAILABLE = False
        print("⚠️  Optuna가 설치되지 않았습니다. 기본 파라미터를 사용합니다.")
        
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    sys.exit(1)


def optimize_strategy_parameters_simple(train_df: pd.DataFrame, model, n_trials: int = 20) -> dict:
    """간단한 전략 파라미터 최적화"""
    try:
        if not OPTUNA_AVAILABLE:
            return {
                'rsi_buy_threshold': 30,
                'rsi_sell_threshold': 70,
                'bb_threshold': 0.8,
                'volume_threshold': 1.5,
                'min_confidence': 0.6
            }
        
        def objective(trial):
            params = {
                'rsi_buy_threshold': trial.suggest_int('rsi_buy_threshold', 20, 40),
                'rsi_sell_threshold': trial.suggest_int('rsi_sell_threshold', 60, 80),
                'bb_threshold': trial.suggest_float('bb_threshold', 0.5, 1.0),
                'volume_threshold': trial.suggest_float('volume_threshold', 1.0, 2.5),
                'min_confidence': trial.suggest_float('min_confidence', 0.4, 0.8)
            }
            
            try:
                result = run_crypto_backtest(
                    df=train_df.copy(),
                    model=model,
                    params=params,
                    is_optimization=True
                )
                
                # 목적 함수: 수익률 / 최대 드로우다운
                total_return = result.get('total_return', 0)
                max_drawdown = max(result.get('max_drawdown', 1.0), 0.01)
                
                return total_return / max_drawdown
                
            except Exception:
                return -1000  # 실패 시 매우 낮은 값
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
        
    except Exception as e:
        print(f"파라미터 최적화 오류: {e}")
        return {
            'rsi_buy_threshold': 30,
            'rsi_sell_threshold': 70,
            'bb_threshold': 0.8,
            'volume_threshold': 1.5,
            'min_confidence': 0.6
        }


def run_walk_forward_optimization(
    df: pd.DataFrame,
    initial_capital: float = 10000000,
    fold_months: int = 6,
    test_months: int = 3,
    training_periods: list = [180, 365, 540],  # 6개월, 1년, 1.5년
    optimization_trials: int = 30
):
    """
    🔬 워크-포워드 최적화 with 훈련 기간 탐색
    각 Fold에서 여러 훈련 기간으로 모델을 훈련시킨 후 최적 기간 선택
    """
    try:
        print(f"\n{'='*80}")
        print(f"🔬 워크-포워드 최적화 with 훈련 기간 탐색")
        print(f"{'='*80}")
        print(f"📊 전체 데이터: {len(df)}개 캔들")
        print(f"🔄 훈련 Fold: {fold_months}개월")
        print(f"🧪 테스트 기간: {test_months}개월")
        print(f"⏱️  탐색할 훈련 기간: {training_periods}일")
        print(f"🎯 최적화 시도: {optimization_trials}회")
        print(f"{'='*80}")
        
        # 1. 데이터를 월별로 분할
        df_monthly = df.resample('M').last()
        total_months = len(df_monthly)
        
        if total_months < fold_months + test_months:
            raise ValueError(f"데이터가 부족합니다. 필요: {fold_months + test_months}개월, 보유: {total_months}개월")
        
        # 워크-포워드 결과 저장
        wf_results = []
        fold_performances = []
        
        # 2. 각 Fold별 워크-포워드 실행
        current_month = 0
        fold_num = 1
        
        while current_month + fold_months + test_months <= total_months:
            try:
                print(f"\n🔄 Fold {fold_num}: 월 {current_month} ~ {current_month + fold_months + test_months}")
                
                # 훈련 기간 설정
                train_start_month = current_month
                train_end_month = current_month + fold_months
                test_start_month = train_end_month
                test_end_month = test_start_month + test_months
                
                # 해당 기간 데이터 추출
                train_start_date = df_monthly.index[train_start_month]
                train_end_date = df_monthly.index[train_end_month - 1]
                test_start_date = df_monthly.index[test_start_month]
                test_end_date = df_monthly.index[test_end_month - 1]
                
                fold_train_df = df[(df.index >= train_start_date) & (df.index <= train_end_date)]
                fold_test_df = df[(df.index >= test_start_date) & (df.index <= test_end_date)]
                
                print(f"   📈 훈련 데이터: {len(fold_train_df)}개 ({train_start_date.strftime('%Y-%m-%d')} ~ {train_end_date.strftime('%Y-%m-%d')})")
                print(f"   🧪 테스트 데이터: {len(fold_test_df)}개 ({test_start_date.strftime('%Y-%m-%d')} ~ {test_end_date.strftime('%Y-%m-%d')})")
                
                # 3. 최적의 훈련 기간 탐색
                best_performance = -np.inf
                best_model = None
                best_params = None
                best_period = None
                period_results = {}
                
                print(f"   🔬 최적 훈련 기간 탐색 중...")
                
                for period_days in training_periods:
                    try:
                        # 현재 훈련 기간으로 서브셋 생성
                        if len(fold_train_df) > period_days * 24:  # 1일 = 24시간 가정
                            period_hours = period_days * 24
                            sub_train_df = fold_train_df.tail(period_hours)
                        else:
                            sub_train_df = fold_train_df.copy()
                        
                        print(f"      📊 {period_days}일 기간 ({len(sub_train_df)}개 캔들)로 훈련...")
                        
                        # ML 모델 훈련
                        model = PricePredictionModel(top_n_features=30)
                        
                        # 피처 생성
                        train_features_df = make_features(sub_train_df.copy())
                        train_features_df = generate_crypto_features(train_features_df)
                        train_features_df = generate_advanced_features(train_features_df)
                        
                        # 모델 훈련
                        model.fit(train_features_df)
                        
                        # 파라미터 최적화 (간단 버전)
                        params = optimize_strategy_parameters_simple(sub_train_df, model, n_trials=20)
                        
                        # In-Sample 성능 평가
                        in_sample_result = run_crypto_backtest(
                            df=sub_train_df.copy(),
                            model=model,
                            params=params,
                            is_optimization=True
                        )
                        
                        # 성과 점수 계산 (수익률 / 드로우다운)
                        total_return = in_sample_result.get('total_return', 0)
                        max_drawdown = max(in_sample_result.get('max_drawdown', 1.0), 0.01)
                        performance_score = total_return / max_drawdown
                        
                        period_results[period_days] = {
                            'model': model,
                            'params': params,
                            'performance_score': performance_score,
                            'total_return': total_return,
                            'max_drawdown': max_drawdown,
                            'in_sample_result': in_sample_result
                        }
                        
                        print(f"         ✅ 성과 점수: {performance_score:.2f} (수익률: {total_return:.2%}, DD: {max_drawdown:.2%})")
                        
                        # 최고 성과 업데이트
                        if performance_score > best_performance:
                            best_performance = performance_score
                            best_model = model
                            best_params = params
                            best_period = period_days
                            
                    except Exception as e:
                        print(f"         ❌ {period_days}일 기간 훈련 실패: {e}")
                        continue
                
                if best_model is None:
                    print(f"   ❌ Fold {fold_num}: 모든 훈련 기간 실패")
                    current_month += test_months
                    fold_num += 1
                    continue
                
                print(f"   🎯 최적 훈련 기간: {best_period}일 (성과 점수: {best_performance:.2f})")
                
                # 4. Out-of-Sample 테스트
                print(f"   🧪 Out-of-Sample 테스트 실행...")
                
                # 테스트 데이터 피처 생성
                test_features_df = make_features(fold_test_df.copy())
                test_features_df = generate_crypto_features(test_features_df)
                test_features_df = generate_advanced_features(test_features_df)
                
                # 백테스트 실행
                oos_result = run_crypto_backtest(
                    df=fold_test_df.copy(),
                    model=best_model,
                    params=best_params,
                    is_optimization=False
                )
                
                # Fold 결과 저장
                fold_result = {
                    'fold_num': fold_num,
                    'train_period': best_period,
                    'train_start': train_start_date,
                    'train_end': train_end_date,
                    'test_start': test_start_date,
                    'test_end': test_end_date,
                    'train_candles': len(fold_train_df),
                    'test_candles': len(fold_test_df),
                    'best_params': best_params,
                    'period_results': period_results,
                    'oos_result': oos_result,
                    'oos_return': oos_result.get('total_return', 0),
                    'oos_sharpe': oos_result.get('sharpe_ratio', 0),
                    'oos_max_dd': oos_result.get('max_drawdown', 1.0)
                }
                
                wf_results.append(fold_result)
                fold_performances.append({
                    'fold': fold_num,
                    'period': best_period,
                    'return': oos_result.get('total_return', 0),
                    'sharpe': oos_result.get('sharpe_ratio', 0),
                    'max_dd': oos_result.get('max_drawdown', 1.0)
                })
                
                print(f"   📊 Out-of-Sample 결과:")
                print(f"      💰 수익률: {oos_result.get('total_return', 0):.2%}")
                print(f"      📈 샤프 비율: {oos_result.get('sharpe_ratio', 0):.2f}")
                print(f"      📉 최대 DD: {oos_result.get('max_drawdown', 0):.2%}")
                
                # 다음 Fold로
                current_month += test_months
                fold_num += 1
                
            except Exception as e:
                print(f"   ❌ Fold {fold_num} 처리 실패: {e}")
                current_month += test_months
                fold_num += 1
                continue
        
        # 5. 전체 결과 분석
        if fold_performances:
            avg_return = np.mean([f['return'] for f in fold_performances])
            avg_sharpe = np.mean([f['sharpe'] for f in fold_performances])
            avg_dd = np.mean([f['max_dd'] for f in fold_performances])
            
            # 최적 훈련 기간 분석
            period_counts = {}
            for f in fold_performances:
                period = f['period']
                period_counts[period] = period_counts.get(period, 0) + 1
            
            best_period_overall = max(period_counts.keys(), key=lambda x: period_counts[x])
            
            print(f"\n🎉 워크-포워드 최적화 완료!")
            print(f"📊 총 Fold 수: {len(fold_performances)}")
            print(f"📈 평균 수익률: {avg_return:.2%}")
            print(f"⚖️  평균 샤프 비율: {avg_sharpe:.2f}")
            print(f"📉 평균 최대 DD: {avg_dd:.2%}")
            print(f"🎯 가장 많이 선택된 훈련 기간: {best_period_overall}일 ({period_counts[best_period_overall]}회)")
            
            # 훈련 기간별 성과 통계
            print(f"\n📊 훈련 기간별 선택 횟수:")
            for period in sorted(period_counts.keys()):
                count = period_counts[period]
                period_returns = [f['return'] for f in fold_performances if f['period'] == period]
                avg_return_for_period = np.mean(period_returns) if period_returns else 0
                print(f"   {period:3d}일: {count:2d}회 선택 (평균 수익률: {avg_return_for_period:6.2%})")
        
        return {
            'wf_results': wf_results,
            'fold_performances': fold_performances,
            'summary': {
                'total_folds': len(fold_performances),
                'avg_return': avg_return if fold_performances else 0,
                'avg_sharpe': avg_sharpe if fold_performances else 0,
                'avg_max_dd': avg_dd if fold_performances else 0,
                'best_period_overall': best_period_overall if fold_performances else None,
                'period_counts': period_counts if fold_performances else {}
            }
        }
        
    except Exception as e:
        print(f"❌ 워크-포워드 최적화 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_walk_forward_results(wf_result):
    """워크-포워드 결과 시각화"""
    try:
        if not wf_result or not wf_result['fold_performances']:
            print("❌ 시각화할 워크-포워드 결과가 없습니다.")
            return
        
        fold_performances = wf_result['fold_performances']
        
        print(f"\n{'='*80}")
        print(f"📊 워크-포워드 결과 시각화")
        print(f"{'='*80}")
        
        # 1. Fold별 성과 테이블
        print(f"\n📈 Fold별 성과:")
        print("-" * 70)
        print(f"{'Fold':<4} {'기간':<6} {'수익률':<8} {'샤프비율':<8} {'최대DD':<8} {'등급':<6}")
        print("-" * 70)
        
        for perf in fold_performances:
            fold_num = perf['fold']
            period = perf['period']
            ret = perf['return']
            sharpe = perf['sharpe']
            dd = perf['max_dd']
            
            # 등급 계산
            if ret > 0.5 and sharpe > 1.0 and dd < 0.15:
                grade = "A+"
            elif ret > 0.3 and sharpe > 0.5 and dd < 0.20:
                grade = "A"
            elif ret > 0.2 and sharpe > 0.3 and dd < 0.25:
                grade = "B+"
            elif ret > 0.1 and sharpe > 0 and dd < 0.30:
                grade = "B"
            elif ret > 0:
                grade = "C+"
            else:
                grade = "C"
            
            print(f"{fold_num:<4} {period:<6} {ret:<8.2%} {sharpe:<8.2f} {dd:<8.2%} {grade:<6}")
        
        # 2. 훈련 기간별 통계
        period_stats = {}
        for perf in fold_performances:
            period = perf['period']
            if period not in period_stats:
                period_stats[period] = {'count': 0, 'returns': [], 'sharpes': [], 'dds': []}
            
            period_stats[period]['count'] += 1
            period_stats[period]['returns'].append(perf['return'])
            period_stats[period]['sharpes'].append(perf['sharpe'])
            period_stats[period]['dds'].append(perf['max_dd'])
        
        print(f"\n📊 훈련 기간별 성과 통계:")
        print("-" * 80)
        print(f"{'기간':<6} {'선택':<4} {'평균수익률':<10} {'평균샤프':<8} {'평균DD':<8} {'안정성':<6}")
        print("-" * 80)
        
        for period in sorted(period_stats.keys()):
            stats = period_stats[period]
            count = stats['count']
            avg_ret = np.mean(stats['returns'])
            avg_sharpe = np.mean(stats['sharpes'])
            avg_dd = np.mean(stats['dds'])
            
            # 안정성 점수 (수익률 표준편차의 역수)
            ret_std = np.std(stats['returns']) if len(stats['returns']) > 1 else 1.0
            stability = 1.0 / (ret_std + 0.01)  # 0으로 나누기 방지
            
            stability_grade = "높음" if stability > 10 else "보통" if stability > 5 else "낮음"
            
            print(f"{period:<6} {count:<4} {avg_ret:<10.2%} {avg_sharpe:<8.2f} {avg_dd:<8.2%} {stability_grade:<6}")
        
        # 3. 전체 요약
        summary = wf_result['summary']
        print(f"\n🎯 전체 요약:")
        print(f"📊 총 Fold 수: {summary['total_folds']}")
        print(f"📈 평균 수익률: {summary['avg_return']:.2%}")
        print(f"⚖️  평균 샤프 비율: {summary['avg_sharpe']:.2f}")
        print(f"📉 평균 최대 DD: {summary['avg_max_dd']:.2%}")
        print(f"🎯 최적 훈련 기간: {summary['best_period_overall']}일")
        
        # 4. 권장사항
        print(f"\n💡 권장사항:")
        best_period = summary['best_period_overall']
        if best_period:
            best_stats = period_stats[best_period]
            best_avg_return = np.mean(best_stats['returns'])
            best_avg_sharpe = np.mean(best_stats['sharpes'])
            
            print(f"   🎯 권장 훈련 기간: {best_period}일")
            print(f"   📈 예상 성과: 수익률 {best_avg_return:.2%}, 샤프 비율 {best_avg_sharpe:.2f}")
            
            if best_avg_return > 0.3:
                print(f"   ✅ 우수한 성과가 예상됩니다!")
            elif best_avg_return > 0.15:
                print(f"   ✅ 양호한 성과가 예상됩니다.")
            else:
                print(f"   ⚠️  성과 개선이 필요할 수 있습니다.")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        print(f"❌ 워크-포워드 시각화 오류: {e}")


def analyze_training_period_trends(wf_result):
    """훈련 기간 트렌드 분석"""
    try:
        if not wf_result or not wf_result['wf_results']:
            print("❌ 분석할 워크-포워드 결과가 없습니다.")
            return
        
        print(f"\n{'='*80}")
        print(f"📈 훈련 기간 트렌드 분석")
        print(f"{'='*80}")
        
        wf_results = wf_result['wf_results']
        
        # 시간에 따른 최적 훈련 기간 변화
        print(f"⏰ 시간에 따른 최적 훈련 기간 변화:")
        print("-" * 60)
        print(f"{'Fold':<4} {'테스트 기간':<20} {'최적 기간':<8} {'성과':<8}")
        print("-" * 60)
        
        for result in wf_results:
            fold_num = result['fold_num']
            test_start = result['test_start'].strftime('%Y-%m-%d')
            best_period = result['train_period']
            oos_return = result['oos_return']
            
            print(f"{fold_num:<4} {test_start:<20} {best_period:<8} {oos_return:<8.2%}")
        
        # 시장 상황별 최적 훈련 기간 분석
        print(f"\n📊 시장 변동성별 최적 훈련 기간:")
        
        # 각 Fold의 변동성 계산 (Out-of-Sample 기간)
        volatility_periods = []
        for result in wf_results:
            oos_result = result['oos_result']
            volatility = oos_result.get('volatility', 0.05)  # 기본값
            
            if volatility > 0.08:
                vol_regime = "고변동성"
            elif volatility > 0.05:
                vol_regime = "중변동성"
            else:
                vol_regime = "저변동성"
            
            volatility_periods.append({
                'fold': result['fold_num'],
                'volatility': volatility,
                'vol_regime': vol_regime,
                'best_period': result['train_period'],
                'oos_return': result['oos_return']
            })
        
        # 변동성 구간별 통계
        vol_stats = {}
        for vp in volatility_periods:
            regime = vp['vol_regime']
            if regime not in vol_stats:
                vol_stats[regime] = {'periods': [], 'returns': []}
            
            vol_stats[regime]['periods'].append(vp['best_period'])
            vol_stats[regime]['returns'].append(vp['oos_return'])
        
        print("-" * 60)
        print(f"{'변동성 구간':<10} {'평균 기간':<8} {'평균 수익률':<10} {'선호 기간':<10}")
        print("-" * 60)
        
        for regime, stats in vol_stats.items():
            if stats['periods']:
                avg_period = np.mean(stats['periods'])
                avg_return = np.mean(stats['returns'])
                # 가장 많이 선택된 기간
                from collections import Counter
                period_counts = Counter(stats['periods'])
                preferred_period = period_counts.most_common(1)[0][0]
                
                print(f"{regime:<10} {avg_period:<8.0f} {avg_return:<10.2%} {preferred_period:<10}")
        
        # 성과 분포 분석
        print(f"\n📊 훈련 기간별 성과 분포:")
        
        period_performance = {}
        for result in wf_results:
            period = result['train_period']
            oos_return = result['oos_return']
            
            if period not in period_performance:
                period_performance[period] = []
            
            period_performance[period].append(oos_return)
        
        print("-" * 70)
        print(f"{'기간':<6} {'사용횟수':<8} {'평균수익률':<10} {'최고수익률':<10} {'최저수익률':<10}")
        print("-" * 70)
        
        for period in sorted(period_performance.keys()):
            returns = period_performance[period]
            count = len(returns)
            avg_return = np.mean(returns)
            max_return = max(returns)
            min_return = min(returns)
            
            print(f"{period:<6} {count:<8} {avg_return:<10.2%} {max_return:<10.2%} {min_return:<10.2%}")
        
        # 결론 및 권장사항
        print(f"\n💡 분석 결론:")
        
        # 가장 안정적인 훈련 기간 찾기
        stability_scores = {}
        for period, returns in period_performance.items():
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                # 샤프 비율과 유사한 안정성 점수
                stability_score = avg_return / (std_return + 0.01)
                stability_scores[period] = stability_score
        
        if stability_scores:
            most_stable_period = max(stability_scores.keys(), key=lambda x: stability_scores[x])
            print(f"   🎯 가장 안정적인 훈련 기간: {most_stable_period}일")
            print(f"   📈 해당 기간의 평균 수익률: {np.mean(period_performance[most_stable_period]):.2%}")
        
        # 시장 상황별 권장사항
        print(f"\n🎯 시장 상황별 권장 훈련 기간:")
        for regime, stats in vol_stats.items():
            if stats['periods']:
                from collections import Counter
                period_counts = Counter(stats['periods'])
                top_period = period_counts.most_common(1)[0][0]
                print(f"   {regime}: {top_period}일 권장")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        print(f"❌ 훈련 기간 트렌드 분석 오류: {e}")
        import traceback
        traceback.print_exc()


def run_comprehensive_walk_forward_test():
    """포괄적 워크-포워드 테스트 실행"""
    try:
        print("🔬 포괄적 워크-포워드 최적화 테스트 시작!")
        
        # 1. 데이터 생성
        print("📊 테스트 데이터 생성 중...")
        df = generate_historical_data(years=3)  # 3년치 데이터
        
        # 2. 워크-포워드 최적화 실행
        print("🔄 워크-포워드 최적화 실행 중...")
        wf_results = run_walk_forward_optimization(
            df=df,
            fold_months=6,      # 6개월 훈련 Fold
            test_months=2,      # 2개월 테스트
            training_periods=[180, 365, 540, 730],  # 6개월, 1년, 1.5년, 2년
            optimization_trials=25
        )
        
        if not wf_results:
            print("❌ 워크-포워드 최적화 실패")
            return None
        
        # 3. 결과 시각화
        print("📊 결과 시각화 중...")
        visualize_walk_forward_results(wf_results)
        
        # 4. 훈련 기간 트렌드 분석
        print("📈 훈련 기간 트렌드 분석 중...")
        analyze_training_period_trends(wf_results)
        
        print("🎉 포괄적 워크-포워드 테스트 완료!")
        
        return wf_results
        
    except Exception as e:
        print(f"❌ 포괄적 워크-포워드 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """워크-포워드 최적화 시스템 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='워크-포워드 최적화 시스템')
    parser.add_argument('--mode', choices=['basic', 'comprehensive'], 
                       default='comprehensive', help='실행 모드')
    parser.add_argument('--years', type=int, default=3, help='데이터 연수')
    parser.add_argument('--fold_months', type=int, default=6, help='훈련 Fold 개월수')
    parser.add_argument('--test_months', type=int, default=2, help='테스트 개월수')
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        run_comprehensive_walk_forward_test()
    else:
        df = generate_historical_data(years=args.years)
        results = run_walk_forward_optimization(
            df=df,
            fold_months=args.fold_months,
            test_months=args.test_months
        )
        if results:
            visualize_walk_forward_results(results) 