#!/usr/bin/env python3
"""
ML 모델 백테스트 실행 파일
몇 년치 백테스트가 가능한 ML 모델 테스트
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
from tqdm import tqdm
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.models.price_prediction_model import PricePredictionModel
from core.trading_engine.dynamic_leverage_manager import DynamicLeverageManager, MarketCondition, PhaseType
from data.market_data.data_generator import MarketDataGenerator
from utils.indicators.technical_indicators import TechnicalIndicators

def setup_logging():
    """
    로그 설정 (한국시간)
    """
    seoul_tz = pytz.timezone('Asia/Seoul')
    class SeoulFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, seoul_tz)
            if datefmt:
                s = dt.strftime(datefmt)
            else:
                s = dt.strftime("%Y-%m-%d %H:%M:%S")
            return s
    formatter = SeoulFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ml_backtest.log'),
            logging.StreamHandler()
        ]
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    return logging.getLogger(__name__)

def generate_historical_data(years: int = 3) -> pd.DataFrame:
    """과거 데이터 생성 (몇 년치)"""
    logger = logging.getLogger(__name__)
    logger.info(f"{years}년치 과거 데이터 생성 시작")
    
    # 데이터 생성기 초기화
    data_generator = MarketDataGenerator()
    
    # 시작 날짜 (3년 전)
    start_date = datetime.now() - timedelta(days=365 * years)
    
    # 데이터 생성
    df = data_generator.generate_historical_data(
        start_date=start_date,
        end_date=datetime.now(),
        symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        timeframe='1h'
    )
    
    logger.info(f"데이터 생성 완료: {len(df)} 개 데이터 포인트")
    # 데이터 생성 후 바로 아래 코드 추가
    return df

def run_ml_backtest(df: pd.DataFrame, initial_capital: float = 10000000):
    """ML 모델 백테스트 실행"""
    logger = logging.getLogger(__name__)
    logger.info("ML 모델 백테스트 시작")
    
    # ML 모델 초기화
    ml_model = PricePredictionModel()
    
    # 동적 레버리지 관리자 초기화
    leverage_manager = DynamicLeverageManager()
    
    # 기술적 지표 계산
    indicators = TechnicalIndicators()
    df_with_indicators = indicators.add_all_indicators(df.copy())
    
    # 백테스트 결과 저장
    results = {
        'timestamp': [],
        'capital': [],
        'leverage': [],
        'position': [],
        'prediction': [],
        'actual_return': [],
        'cumulative_return': []
    }
    
    current_capital = initial_capital
    peak_capital = initial_capital
    position = 0  # 0: 없음, 1: 롱, -1: 숏
    consecutive_wins = 0
    consecutive_losses = 0
    
    # 훈련 데이터 크기 (전체 데이터의 70%)
    train_size = int(len(df_with_indicators) * 0.7)
    train_data = df_with_indicators.iloc[:train_size]
    test_data = df_with_indicators.iloc[train_size:]
    
    logger.info(f"훈련 데이터: {len(train_data)} 개, 테스트 데이터: {len(test_data)} 개")
    
    # 모델 훈련
    logger.info("ML 모델 훈련 시작")
    ml_model.fit(train_data, tune=True)
    logger.info("ML 모델 훈련 완료")
    
    # 백테스트 실행
    start_time = time.time()
    for i in tqdm(
        range(len(test_data)),
        desc='백테스트 진행중',
        ncols=80,
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
        mininterval=0.1,
        ascii=True,
        disable=False
    ):
        idx, row = test_data.iloc[i].name, test_data.iloc[i]
        try:
            # 현재 시장 상황 분석
            market_condition = analyze_market_condition(row)
            
            # 변동성 계산 (20일)
            volatility = row.get('volatility_20', 0.05)
            
            # RSI
            rsi = row.get('rsi_14', 50)
            
            # 동적 레버리지 계산
            current_leverage = leverage_manager.update_leverage(
                phase=PhaseType.PHASE1_AGGRESSIVE,
                market_condition=market_condition,
                current_capital=current_capital,
                peak_capital=peak_capital,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                volatility=volatility,
                rsi=rsi
            )
            
            # ML 예측
            prediction_data = df_with_indicators.iloc[:train_size + i + 1]
            if len(prediction_data) > 60:  # 최소 데이터 필요
                prediction = ml_model.predict(prediction_data, model_type='ensemble')
                if len(prediction) > 0:
                    predicted_return = prediction[-1]
                else:
                    predicted_return = 0
            else:
                predicted_return = 0
            
            # 거래 신호 생성
            signal = generate_trading_signal(predicted_return, row, current_leverage)
            
            # 포지션 업데이트
            if signal == 1 and position <= 0:  # 롱 진입
                position = 1
                logger.info(f"{idx}: 롱 진입 (예측: {predicted_return:.4f}, 레버리지: {current_leverage})")
            elif signal == -1 and position >= 0:  # 숏 진입
                position = -1
                logger.info(f"{idx}: 숏 진입 (예측: {predicted_return:.4f}, 레버리지: {current_leverage})")
            elif signal == 0 and position != 0:  # 청산
                position = 0
                logger.info(f"{idx}: 포지션 청산")
            
            # 수익률 계산
            if i > 0:
                actual_return = (row['close'] - test_data.iloc[i-1]['close']) / test_data.iloc[i-1]['close']
                if position == 1:  # 롱
                    capital_change = actual_return * current_leverage
                elif position == -1:  # 숏
                    capital_change = -actual_return * current_leverage
                else:
                    capital_change = 0
                
                current_capital *= (1 + capital_change)
                peak_capital = max(peak_capital, current_capital)
                
                # 연속 승/패 업데이트
                if capital_change > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                elif capital_change < 0:
                    consecutive_losses += 1
                    consecutive_wins = 0
                else:
                    consecutive_wins = 0
                    consecutive_losses = 0
            else:
                actual_return = 0
                capital_change = 0
            
            # 결과 저장
            results['timestamp'].append(idx)
            results['capital'].append(current_capital)
            results['leverage'].append(current_leverage)
            results['position'].append(position)
            results['prediction'].append(predicted_return)
            results['actual_return'].append(actual_return)
            results['cumulative_return'].append((current_capital - initial_capital) / initial_capital)
            
            # 진행상황 로그 (1000개마다)
            if (i + 1) % 1000 == 0 or i == len(test_data) - 1:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(test_data) - (i + 1)) if (i + 1) > 0 else 0
                logger.info(f"진행률: {i+1}/{len(test_data)} ({((i+1)/len(test_data)*100):.1f}%) | 경과: {elapsed:.1f}s | 예상 남은시간: {eta:.1f}s")
                print(f"진행률: {i+1}/{len(test_data)} ({((i+1)/len(test_data)*100):.1f}%) | 경과: {elapsed:.1f}s | 예상 남은시간: {eta:.1f}s", flush=True)
                
        except Exception as e:
            logger.error(f"백테스트 중 오류 발생: {e}")
            continue
    
    # 결과 분석
    analyze_backtest_results(results, initial_capital)
    
    return results

def analyze_market_condition(row: pd.Series) -> MarketCondition:
    """시장 상황 분석"""
    
    # RSI 기반
    rsi = row.get('rsi_14', 50)
    
    # 이동평균 기반
    ma_20 = row.get('ma_20', row['close'])
    ma_50 = row.get('ma_50', row['close'])
    
    # 변동성
    volatility = row.get('volatility_20', 0.05)
    
    # 시장 상황 판단
    if rsi > 70 and ma_20 > ma_50 and volatility < 0.08:
        return MarketCondition.BULL_MARKET
    elif rsi < 30 and ma_20 < ma_50 and volatility < 0.08:
        return MarketCondition.BEAR_MARKET
    elif volatility > 0.10:
        return MarketCondition.HIGH_VOLATILITY
    elif volatility < 0.03:
        return MarketCondition.LOW_VOLATILITY
    else:
        return MarketCondition.SIDEWAYS

def generate_trading_signal(predicted_return: float, row: pd.Series, leverage: float) -> int:
    """거래 신호 생성"""
    
    # 예측 수익률 임계값
    threshold = 0.001  # 0.1%
    
    # 추가 필터링 조건
    rsi = row.get('rsi_14', 50)
    volatility = row.get('volatility_20', 0.05)
    
    # 과매수/과매도 조건
    if rsi > 80 or rsi < 20:
        return 0  # 청산
    
    # 고변동성 조건
    if volatility > 0.15:
        return 0  # 청산
    
    # 신호 생성
    if predicted_return > threshold:
        return 1  # 롱
    elif predicted_return < -threshold:
        return -1  # 숏
    else:
        return 0  # 홀드

def analyze_backtest_results(results: dict, initial_capital: float):
    """백테스트 결과 분석"""
    logger = logging.getLogger(__name__)
    
    df_results = pd.DataFrame(results)
    
    # 기본 통계
    final_capital = df_results['capital'].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # 최대 낙폭
    peak_capital = df_results['capital'].max()
    max_drawdown = (peak_capital - df_results['capital'].min()) / peak_capital * 100
    
    # 승률
    profitable_trades = len(df_results[df_results['actual_return'] > 0])
    total_trades = len(df_results[df_results['position'] != 0])
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
    
    # 평균 레버리지
    avg_leverage = df_results['leverage'].mean()
    
    # 결과 출력
    logger.info("=== ML 백테스트 결과 ===")
    logger.info(f"초기 자본: {initial_capital:,.0f}원")
    logger.info(f"최종 자본: {final_capital:,.0f}원")
    logger.info(f"총 수익률: {total_return:.2f}%")
    logger.info(f"최대 낙폭: {max_drawdown:.2f}%")
    logger.info(f"승률: {win_rate:.1f}%")
    logger.info(f"평균 레버리지: {avg_leverage:.2f}x")
    logger.info(f"총 거래 횟수: {total_trades}")
    
    # 결과 저장
    df_results.to_csv('data/backtest_data/ml_backtest_results.csv', index=False)
    logger.info("백테스트 결과가 data/backtest_data/ml_backtest_results.csv에 저장되었습니다.")

def main():
    """메인 함수"""
    logger = setup_logging()
    logger.info("ML 모델 백테스트 시스템 시작")
    
    try:
        # 3년치 데이터 생성
        df = generate_historical_data(years=3)
        
        # ML 백테스트 실행
        results = run_ml_backtest(df, initial_capital=10000000)
        
        logger.info("ML 백테스트 완료")
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 