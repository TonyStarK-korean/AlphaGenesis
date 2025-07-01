#!/usr/bin/env python3
"""
🤖 ML 모델 예측 정확도 분석 시스템
ML 예측 결과와 실제 가격 움직임을 비교하여 모델의 정확도를 시각적으로 분석
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
        generate_advanced_features, generate_historical_data
    )
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    sys.exit(1)


def analyze_ml_prediction_accuracy(df, model, window_days=30):
    """ML 모델 예측 정확도 분석"""
    try:
        print(f"\n{'='*80}")
        print(f"🤖 ML 모델 예측 정확도 분석")
        print(f"{'='*80}")
        
        if not model or not hasattr(model, 'is_fitted') or not model.is_fitted:
            print("❌ 훈련된 ML 모델이 없습니다.")
            return
        
        # 피처 생성
        df_features = make_features(df.copy())
        df_features = generate_crypto_features(df_features)
        df_features = generate_advanced_features(df_features)
        df_features.dropna(inplace=True)
        
        if len(df_features) < window_days:
            print(f"❌ 분석할 데이터가 부족합니다. (필요: {window_days}일, 보유: {len(df_features)}일)")
            return
        
        # ML 예측 생성
        predictions = model.predict(df_features)
        df_features['ml_prediction'] = predictions
        
        # 실제 수익률 계산 (다음 캔들 수익률)
        df_features['actual_return'] = df_features['close'].pct_change().shift(-1)
        df_features.dropna(inplace=True)
        
        # 예측 방향 vs 실제 방향
        df_features['pred_direction'] = np.where(df_features['ml_prediction'] > 0, 1, -1)
        df_features['actual_direction'] = np.where(df_features['actual_return'] > 0, 1, -1)
        df_features['direction_correct'] = (df_features['pred_direction'] == df_features['actual_direction'])
        
        # 전체 정확도
        total_predictions = len(df_features)
        direction_accuracy = df_features['direction_correct'].mean()
        
        print(f"📊 전체 예측 정확도:")
        print(f"   총 예측 횟수: {total_predictions:,}회")
        print(f"   방향 정확도: {direction_accuracy:.2%}")
        
        # 신뢰도별 정확도 분석
        confidence_ranges = [
            (0.0, 0.2, "매우 낮음"),
            (0.2, 0.4, "낮음"),
            (0.4, 0.6, "보통"),
            (0.6, 0.8, "높음"),
            (0.8, 1.0, "매우 높음")
        ]
        
        print(f"\n📊 신뢰도별 정확도:")
        print("-" * 60)
        print(f"{'신뢰도':<12} {'예측수':<8} {'정확도':<8} {'평균수익률':<12}")
        print("-" * 60)
        
        for min_conf, max_conf, label in confidence_ranges:
            mask = (df_features['ml_prediction'].abs() >= min_conf) & (df_features['ml_prediction'].abs() < max_conf)
            subset = df_features[mask]
            
            if len(subset) > 0:
                subset_accuracy = subset['direction_correct'].mean()
                avg_return = subset['actual_return'].mean()
                count = len(subset)
                
                print(f"{label:<12} {count:<8} {subset_accuracy:<8.2%} {avg_return:<12.4%}")
        
        # 시간대별 정확도 분석
        df_features['hour'] = df_features.index.hour
        hourly_accuracy = df_features.groupby('hour')['direction_correct'].agg(['count', 'mean'])
        
        print(f"\n⏰ 시간대별 정확도 (상위 5개):")
        top_hours = hourly_accuracy.sort_values('mean', ascending=False).head(5)
        print("-" * 40)
        print(f"{'시간':<6} {'예측수':<8} {'정확도':<8}")
        print("-" * 40)
        for hour, (count, accuracy) in top_hours.iterrows():
            if count >= 10:  # 최소 10개 예측이 있는 시간대만
                print(f"{hour:02d}:00  {count:<8.0f} {accuracy:<8.2%}")
        
        # 예측 강도별 실제 수익률 분포
        df_features['pred_strength'] = df_features['ml_prediction'].abs()
        strength_ranges = [
            (0.0, 0.3, "약함"),
            (0.3, 0.6, "보통"), 
            (0.6, 1.0, "강함")
        ]
        
        print(f"\n💪 예측 강도별 실제 수익률:")
        print("-" * 50)
        print(f"{'강도':<8} {'예측수':<8} {'평균수익률':<12} {'승률':<8}")
        print("-" * 50)
        
        for min_str, max_str, label in strength_ranges:
            mask = (df_features['pred_strength'] >= min_str) & (df_features['pred_strength'] < max_str)
            subset = df_features[mask]
            
            if len(subset) > 0:
                avg_return = subset['actual_return'].mean()
                win_rate = (subset['actual_return'] > 0).mean()
                count = len(subset)
                
                print(f"{label:<8} {count:<8} {avg_return:<12.4%} {win_rate:<8.2%}")
        
        # 최고/최악 예측 사례
        df_features['pred_return_product'] = df_features['ml_prediction'] * df_features['actual_return']
        
        print(f"\n🎯 예측 성과 분석:")
        
        # 가장 정확했던 예측들 (예측과 실제가 같은 방향이면서 큰 수익)
        best_predictions = df_features.nlargest(5, 'pred_return_product')
        print(f"   🟢 최고 예측 5건:")
        for idx, row in best_predictions.iterrows():
            print(f"      {idx}: 예측 {row['ml_prediction']:+.3f} → 실제 {row['actual_return']:+.4%}")
        
        # 가장 틀렸던 예측들
        worst_predictions = df_features.nsmallest(5, 'pred_return_product')
        print(f"   🔴 최악 예측 5건:")
        for idx, row in worst_predictions.iterrows():
            print(f"      {idx}: 예측 {row['ml_prediction']:+.3f} → 실제 {row['actual_return']:+.4%}")
        
        # 개선 권장사항
        print(f"\n💡 모델 개선 권장사항:")
        
        if direction_accuracy < 0.55:
            print(f"   ⚠️  전체 정확도가 낮습니다 ({direction_accuracy:.2%}). 피처 추가나 모델 변경을 고려해보세요.")
        elif direction_accuracy > 0.65:
            print(f"   ✅ 전체 정확도가 우수합니다 ({direction_accuracy:.2%})!")
        else:
            print(f"   ✅ 전체 정확도가 양호합니다 ({direction_accuracy:.2%}).")
        
        # 높은 신뢰도 예측의 비율
        high_conf_ratio = (df_features['pred_strength'] > 0.6).mean()
        if high_conf_ratio < 0.1:
            print(f"   ⚠️  높은 신뢰도 예측이 부족합니다 ({high_conf_ratio:.1%}). 모델 확신도를 높여보세요.")
        
        print(f"\n{'='*80}")
        
        return {
            'total_predictions': total_predictions,
            'direction_accuracy': direction_accuracy,
            'confidence_analysis': confidence_ranges,
            'hourly_accuracy': hourly_accuracy,
            'strength_analysis': strength_ranges,
            'best_predictions': best_predictions,
            'worst_predictions': worst_predictions,
            'high_conf_ratio': high_conf_ratio
        }
        
    except Exception as e:
        print(f"❌ ML 예측 정확도 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_prediction_vs_market_regime(df, model):
    """시장 국면별 ML 예측 정확도 분석"""
    try:
        print(f"\n{'='*80}")
        print(f"📊 시장 국면별 ML 예측 정확도 분석")
        print(f"{'='*80}")
        
        # 피처 생성
        df_features = make_features(df.copy())
        df_features = generate_crypto_features(df_features)
        df_features = generate_advanced_features(df_features)
        df_features.dropna(inplace=True)
        
        # ML 예측 생성
        predictions = model.predict(df_features)
        df_features['ml_prediction'] = predictions
        
        # 실제 수익률
        df_features['actual_return'] = df_features['close'].pct_change().shift(-1)
        df_features.dropna(inplace=True)
        
        # 예측 방향 vs 실제 방향
        df_features['pred_direction'] = np.where(df_features['ml_prediction'] > 0, 1, -1)
        df_features['actual_direction'] = np.where(df_features['actual_return'] > 0, 1, -1)
        df_features['direction_correct'] = (df_features['pred_direction'] == df_features['actual_direction'])
        
        # 시장 국면 분류
        def classify_market_regime(row):
            """시장 국면 분류"""
            rsi = row.get('rsi_14', 50)
            volatility = row.get('volatility_20', 0.05)
            ma_20 = row.get('ma_20', row['close'])
            ma_50 = row.get('ma_50', row['close'])
            volume_ratio = row.get('volume_ratio', 1.0)
            
            # 변동성 기준
            if volatility > 0.08:
                return "고변동성"
            elif volatility < 0.03:
                return "저변동성"
            
            # 추세 기준
            if ma_20 > ma_50 * 1.02 and rsi > 50:
                return "상승추세"
            elif ma_20 < ma_50 * 0.98 and rsi < 50:
                return "하락추세"
            else:
                return "횡보"
        
        df_features['market_regime'] = df_features.apply(classify_market_regime, axis=1)
        
        # 국면별 정확도 분석
        regime_analysis = df_features.groupby('market_regime').agg({
            'direction_correct': ['count', 'mean'],
            'actual_return': 'mean',
            'ml_prediction': ['mean', 'std']
        }).round(4)
        
        print(f"📊 시장 국면별 예측 정확도:")
        print("-" * 70)
        print(f"{'국면':<10} {'예측수':<8} {'정확도':<8} {'평균수익률':<12} {'예측평균':<10} {'예측편차':<10}")
        print("-" * 70)
        
        for regime in regime_analysis.index:
            count = regime_analysis.loc[regime, ('direction_correct', 'count')]
            accuracy = regime_analysis.loc[regime, ('direction_correct', 'mean')]
            avg_return = regime_analysis.loc[regime, ('actual_return', 'mean')]
            pred_mean = regime_analysis.loc[regime, ('ml_prediction', 'mean')]
            pred_std = regime_analysis.loc[regime, ('ml_prediction', 'std')]
            
            print(f"{regime:<10} {count:<8.0f} {accuracy:<8.2%} {avg_return:<12.4%} {pred_mean:<10.3f} {pred_std:<10.3f}")
        
        # 국면 전환 시점의 예측 정확도
        print(f"\n🔄 국면 전환 시점 분석:")
        
        # 국면 변화 감지
        df_features['regime_change'] = df_features['market_regime'] != df_features['market_regime'].shift(1)
        
        # 전환 시점과 일반 시점 비교
        transition_accuracy = df_features[df_features['regime_change']]['direction_correct'].mean()
        normal_accuracy = df_features[~df_features['regime_change']]['direction_correct'].mean()
        
        print(f"   국면 전환 시점 정확도: {transition_accuracy:.2%}")
        print(f"   일반 시점 정확도: {normal_accuracy:.2%}")
        print(f"   정확도 차이: {transition_accuracy - normal_accuracy:+.2%}")
        
        # 국면별 최고/최악 예측
        print(f"\n🎯 국면별 예측 성과:")
        
        for regime in df_features['market_regime'].unique():
            regime_data = df_features[df_features['market_regime'] == regime]
            if len(regime_data) > 0:
                # 예측과 실제의 곱 (같은 방향이면 양수, 다른 방향이면 음수)
                regime_data['pred_score'] = regime_data['ml_prediction'] * regime_data['actual_return']
                
                best_pred = regime_data.nlargest(1, 'pred_score').iloc[0]
                worst_pred = regime_data.nsmallest(1, 'pred_score').iloc[0]
                
                print(f"   {regime}:")
                print(f"      최고: 예측 {best_pred['ml_prediction']:+.3f} → 실제 {best_pred['actual_return']:+.4%}")
                print(f"      최악: 예측 {worst_pred['ml_prediction']:+.3f} → 실제 {worst_pred['actual_return']:+.4%}")
        
        return {
            'regime_analysis': regime_analysis,
            'transition_accuracy': transition_accuracy,
            'normal_accuracy': normal_accuracy
        }
        
    except Exception as e:
        print(f"❌ 시장 국면별 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_prediction_performance_report(df, model):
    """ML 예측 성과 종합 보고서 생성"""
    try:
        print(f"\n{'='*80}")
        print(f"📋 ML 예측 성과 종합 보고서")
        print(f"{'='*80}")
        
        # 1. 기본 정확도 분석
        basic_analysis = analyze_ml_prediction_accuracy(df, model)
        
        if not basic_analysis:
            print("❌ 기본 분석 실패")
            return None
        
        # 2. 시장 국면별 분석
        regime_analysis = analyze_prediction_vs_market_regime(df, model)
        
        # 3. 시간 흐름에 따른 정확도 변화
        print(f"\n📈 시간 흐름에 따른 정확도 변화:")
        
        # 피처 생성
        df_features = make_features(df.copy())
        df_features = generate_crypto_features(df_features)
        df_features = generate_advanced_features(df_features)
        df_features.dropna(inplace=True)
        
        # ML 예측
        predictions = model.predict(df_features)
        df_features['ml_prediction'] = predictions
        df_features['actual_return'] = df_features['close'].pct_change().shift(-1)
        df_features.dropna(inplace=True)
        
        df_features['pred_direction'] = np.where(df_features['ml_prediction'] > 0, 1, -1)
        df_features['actual_direction'] = np.where(df_features['actual_return'] > 0, 1, -1)
        df_features['direction_correct'] = (df_features['pred_direction'] == df_features['actual_direction'])
        
        # 월별 정확도
        df_features['month'] = df_features.index.to_period('M')
        monthly_accuracy = df_features.groupby('month')['direction_correct'].agg(['count', 'mean'])
        
        print("-" * 40)
        print(f"{'월':<10} {'예측수':<8} {'정확도':<8}")
        print("-" * 40)
        
        for month, (count, accuracy) in monthly_accuracy.tail(6).iterrows():  # 최근 6개월
            print(f"{month}  {count:<8.0f} {accuracy:<8.2%}")
        
        # 4. 종합 평가 및 권장사항
        print(f"\n🏆 종합 평가:")
        
        overall_accuracy = basic_analysis['direction_accuracy']
        high_conf_ratio = basic_analysis['high_conf_ratio']
        
        # 평가 점수 계산
        score = 0
        
        # 정확도 점수 (40점)
        if overall_accuracy > 0.65:
            score += 40
            accuracy_grade = "탁월"
        elif overall_accuracy > 0.60:
            score += 35
            accuracy_grade = "우수"
        elif overall_accuracy > 0.55:
            score += 30
            accuracy_grade = "양호"
        elif overall_accuracy > 0.50:
            score += 20
            accuracy_grade = "보통"
        else:
            score += 10
            accuracy_grade = "개선 필요"
        
        # 신뢰도 점수 (30점)
        if high_conf_ratio > 0.2:
            score += 30
            confidence_grade = "탁월"
        elif high_conf_ratio > 0.15:
            score += 25
            confidence_grade = "우수"
        elif high_conf_ratio > 0.10:
            score += 20
            confidence_grade = "양호"
        else:
            score += 10
            confidence_grade = "개선 필요"
        
        # 안정성 점수 (30점) - 국면별 정확도 편차
        if regime_analysis:
            regime_accuracies = []
            for regime in regime_analysis['regime_analysis'].index:
                acc = regime_analysis['regime_analysis'].loc[regime, ('direction_correct', 'mean')]
                regime_accuracies.append(acc)
            
            accuracy_std = np.std(regime_accuracies)
            if accuracy_std < 0.05:
                score += 30
                stability_grade = "매우 안정적"
            elif accuracy_std < 0.10:
                score += 25
                stability_grade = "안정적"
            elif accuracy_std < 0.15:
                score += 20
                stability_grade = "보통"
            else:
                score += 10
                stability_grade = "불안정"
        else:
            score += 15
            stability_grade = "평가 불가"
        
        # 최종 등급
        if score >= 90:
            final_grade = "A+ (최우수)"
        elif score >= 80:
            final_grade = "A (우수)"
        elif score >= 70:
            final_grade = "B+ (양호)"
        elif score >= 60:
            final_grade = "B (보통)"
        else:
            final_grade = "C (개선 필요)"
        
        print(f"   📊 전체 정확도: {overall_accuracy:.2%} ({accuracy_grade})")
        print(f"   🎯 높은 신뢰도 비율: {high_conf_ratio:.1%} ({confidence_grade})")
        print(f"   ⚖️  국면별 안정성: {stability_grade}")
        print(f"   🏆 종합 등급: {final_grade} ({score}/100점)")
        
        # 개선 권장사항
        print(f"\n💡 개선 권장사항:")
        
        if overall_accuracy < 0.60:
            print(f"   1. 예측 정확도 개선:")
            print(f"      - 더 많은 피처 추가 (예: 감정 지표, 거시경제 데이터)")
            print(f"      - 앙상블 모델 적용")
            print(f"      - 하이퍼파라미터 튜닝")
        
        if high_conf_ratio < 0.15:
            print(f"   2. 신뢰도 개선:")
            print(f"      - 불확실성 추정 모델 적용")
            print(f"      - 베이지안 신경망 고려")
            print(f"      - 예측 강도 캘리브레이션")
        
        if score < 70:
            print(f"   3. 전반적 개선:")
            print(f"      - 더 길고 다양한 데이터셋 사용")
            print(f"      - 정기적인 모델 재훈련")
            print(f"      - A/B 테스트를 통한 전략 검증")
        
        print(f"\n{'='*80}")
        
        return {
            'basic_analysis': basic_analysis,
            'regime_analysis': regime_analysis,
            'monthly_accuracy': monthly_accuracy,
            'final_score': score,
            'final_grade': final_grade
        }
        
    except Exception as e:
        print(f"❌ 종합 보고서 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ml_accuracy_test():
    """ML 정확도 분석 테스트 실행"""
    try:
        print("🤖 ML 예측 정확도 분석 테스트 시작!")
        
        # 1. 데이터 생성
        print("📊 테스트 데이터 생성 중...")
        df = generate_historical_data(years=2)
        
        # 2. ML 모델 훈련
        print("🤖 ML 모델 훈련 중...")
        model = PricePredictionModel(top_n_features=50)
        
        # 피처 생성
        df_features = make_features(df.copy())
        df_features = generate_crypto_features(df_features)
        df_features = generate_advanced_features(df_features)
        
        # 훈련/테스트 분할
        train_size = int(len(df_features) * 0.8)
        train_df = df_features.iloc[:train_size]
        test_df = df_features.iloc[train_size:]
        
        # 모델 훈련
        model.fit(train_df)
        
        # 3. 테스트 데이터로 정확도 분석
        print("📊 예측 정확도 분석 중...")
        
        # 기본 정확도 분석
        basic_result = analyze_ml_prediction_accuracy(test_df, model)
        
        # 시장 국면별 분석
        regime_result = analyze_prediction_vs_market_regime(test_df, model)
        
        # 종합 보고서
        comprehensive_result = create_prediction_performance_report(test_df, model)
        
        print("🎉 ML 정확도 분석 완료!")
        
        return {
            'basic_result': basic_result,
            'regime_result': regime_result,
            'comprehensive_result': comprehensive_result
        }
        
    except Exception as e:
        print(f"❌ ML 정확도 분석 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """ML 예측 정확도 분석 시스템 실행"""
    run_ml_accuracy_test() 