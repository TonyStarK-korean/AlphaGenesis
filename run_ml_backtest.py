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
import re
import optuna
import json, requests

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
    """히스토리컬 데이터 생성"""
    logger = logging.getLogger(__name__)
    
    # 기본 설정
    start_date = datetime.now() - timedelta(days=years * 365)
    end_date = datetime.now()
    
    # 시간 간격 (1시간)
    time_delta = timedelta(hours=1)
    current_date = start_date
    
    data = []
    base_price = 50000  # 기본 가격 (항상 양수)
    
    while current_date <= end_date:
        # 가격 변동 (항상 양수 보장)
        price_change = np.random.normal(0, 0.02)  # 2% 표준편차
        base_price = max(base_price * (1 + price_change), 1000)  # 최소 1000원 보장
        
        # 거래량
        volume = max(int(np.random.normal(1000, 500)), 100)  # 최소 100개 보장
        
        open_p = abs(base_price * (1 + np.random.normal(0, 0.005)))
        high_p = abs(base_price * (1 + abs(np.random.normal(0, 0.01))))
        low_p = abs(base_price * (1 - abs(np.random.normal(0, 0.01))))
        close_p = abs(base_price)
        
        data.append({
            'timestamp': current_date,
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': volume,
            'symbol': 'BNB/USDT'
        })
        
        current_date += time_delta
    
    df = pd.DataFrame(data)
    
    # 데이터 검증 및 정리
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].abs()  # 절댓값으로 음수 제거
        df[col] = df[col].fillna(df[col].mean())  # NaN 값 처리
    
    df['volume'] = df['volume'].abs().fillna(1000)  # 거래량도 양수 보장
    
    logger.info(f"히스토리컬 데이터 생성 완료: {len(df)} 개 데이터")
    return df

def run_ml_backtest(df: pd.DataFrame, initial_capital: float = 10000000, model=None):
    """ML 모델 백테스트 실행"""
    logger = logging.getLogger(__name__)
    logger.info("ML 모델 백테스트 시작")

    # ML 모델은 이미 훈련된 상태로 전달됨
    ml_model = model if model is not None else PricePredictionModel()

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
    
    # 백테스트 실행
    start_time = time.time()
    for i in range(len(test_data)):
        idx, row = test_data.iloc[i].name, test_data.iloc[i]
        try:
            # 데이터 전처리 및 검증
            if pd.isna(row['close']) or row['close'] <= 0:
                logger.warning(f"[{idx}] 유효하지 않은 종가 데이터: close={row['close']}")
                continue
                
            # 현재 시장 상황 분석
            market_condition = analyze_market_condition(row)
            
            # 변동성 계산 (20일)
            volatility = row.get('volatility_20', 0.05)
            if pd.isna(volatility) or volatility < 0:
                volatility = 0.05
            
            # RSI
            rsi = row.get('rsi_14', 50)
            if pd.isna(rsi) or rsi < 0 or rsi > 100:
                rsi = 50
            
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
                    if pd.isna(predicted_return):
                        predicted_return = 0
                else:
                    predicted_return = 0
            else:
                predicted_return = 0
            
            # 거래 신호 생성
            signal = generate_trading_signal(predicted_return, row, current_leverage)
            
            # 포지션 업데이트
            if signal == 1 and position <= 0:  # 롱 진입
                position = 1
                logger.info(f"[{idx}] 롱 진입 (예측: {predicted_return:.4f}, 레버리지: {current_leverage:.2f})")
            elif signal == -1 and position >= 0:  # 숏 진입
                position = -1
                logger.info(f"[{idx}] 숏 진입 (예측: {predicted_return:.4f}, 레버리지: {current_leverage:.2f})")
            elif signal == 0 and position != 0:  # 청산
                position = 0
                logger.info(f"[{idx}] 포지션 청산")
            
            # 수익률 계산 (안전한 방식)
            if i > 0:
                prev_close = test_data.iloc[i-1]['close']
                if pd.isna(prev_close) or prev_close <= 0:
                    actual_return = 0
                    capital_change = 0
                else:
                    actual_return = (row['close'] - prev_close) / prev_close
                    if position == 1:  # 롱
                        capital_change = actual_return * current_leverage
                    elif position == -1:  # 숏
                        capital_change = -actual_return * current_leverage
                    else:
                        capital_change = 0
                
                # 자본 업데이트 (안전한 방식)
                if not pd.isna(capital_change) and abs(capital_change) < 1.0:  # 100% 이상 변동 방지
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
                current_idx = test_data.iloc[i].name if i < len(test_data) else "완료"
                logger.info(f"[{current_idx}] 진행률: {i+1}/{len(test_data)} ({((i+1)/len(test_data)*100):.1f}%) | 경과: {elapsed:.1f}s | 예상 남은시간: {eta:.1f}s")
                print(f"[{current_idx}] 진행률: {i+1}/{len(test_data)} ({((i+1)/len(test_data)*100):.1f}%) | 경과: {elapsed:.1f}s | 예상 남은시간: {eta:.1f}s", flush=True)
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"[{idx}] 백테스트 중 오류 발생: {e}")
            logger.error(f"[{idx}] 상세 오류 정보: {error_details}")
            logger.error(f"[{idx}] row 데이터: {row.to_dict() if hasattr(row, 'to_dict') else row}")
            # 에러 발생 시에도 기본값으로 결과 저장
            results['timestamp'].append(idx)
            results['capital'].append(current_capital)
            results['leverage'].append(1.0)
            results['position'].append(0)
            results['prediction'].append(0)
            results['actual_return'].append(0)
            results['cumulative_return'].append((current_capital - initial_capital) / initial_capital)
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
    if df_results.empty or len(df_results['capital']) == 0:
        logger.error("백테스트 결과 데이터가 비어 있습니다. (루프 내 예외/데이터 없음 등 원인)")
        return
    
    # 전체 성과
    final_capital = df_results['capital'].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100
    profit = final_capital - initial_capital
    peak_capital = df_results['capital'].max()
    max_drawdown = (peak_capital - df_results['capital'].min()) / peak_capital * 100
    profitable_trades = len(df_results[df_results['actual_return'] > 0])
    total_trades = len(df_results[df_results['position'] != 0])
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0

    # 백테스트 기간 정보
    start_time = df_results['timestamp'].iloc[0] if len(df_results['timestamp']) > 0 else "N/A"
    end_time = df_results['timestamp'].iloc[-1] if len(df_results['timestamp']) > 0 else "N/A"

    logger.info("=== 백테스트 성과 요약 ===")
    logger.info(f"백테스트 기간: {start_time} ~ {end_time}")
    logger.info(f"최종 자본: {final_capital:,.0f}원")
    logger.info(f"총 수익률: {total_return:.2f}%")
    logger.info(f"총 수익금(손실금): {profit:,.0f}원")
    logger.info(f"최대 낙폭: {max_drawdown:.2f}%")
    logger.info(f"총 거래 횟수: {total_trades}")
    logger.info(f"승률: {win_rate:.1f}%")

    # 종목별 성과
    if 'symbol' in df_results:
        logger.info("--- 종목별 성과 ---")
        for symbol, group in df_results.groupby('symbol'):
            sym_final = group['capital'].iloc[-1]
            sym_return = (sym_final - initial_capital) / initial_capital * 100
            sym_profit = sym_final - initial_capital
            sym_peak = group['capital'].max()
            sym_mdd = (sym_peak - group['capital'].min()) / sym_peak * 100
            sym_trades = len(group[group['position'] != 0])
            sym_win = len(group[group['actual_return'] > 0])
            sym_winrate = sym_win / sym_trades * 100 if sym_trades > 0 else 0
            sym_start = group['timestamp'].iloc[0] if len(group['timestamp']) > 0 else "N/A"
            sym_end = group['timestamp'].iloc[-1] if len(group['timestamp']) > 0 else "N/A"
            logger.info(f"[{symbol}] 기간: {sym_start} ~ {sym_end} | 최종 자본: {sym_final:,.0f}원 | 수익률: {sym_return:.2f}% | 수익금: {sym_profit:,.0f}원 | 최대 낙폭: {sym_mdd:.2f}% | 거래: {sym_trades} | 승률: {sym_winrate:.1f}%")

    # 전략(phase)별 성과
    if 'phase' in df_results:
        logger.info("--- 전략(phase)별 성과 ---")
        for phase, group in df_results.groupby('phase'):
            ph_final = group['capital'].iloc[-1]
            ph_return = (ph_final - initial_capital) / initial_capital * 100
            ph_profit = ph_final - initial_capital
            ph_peak = group['capital'].max()
            ph_mdd = (ph_peak - group['capital'].min()) / ph_peak * 100
            ph_trades = len(group[group['position'] != 0])
            ph_win = len(group[group['actual_return'] > 0])
            ph_winrate = ph_win / ph_trades * 100 if ph_trades > 0 else 0
            ph_start = group['timestamp'].iloc[0] if len(group['timestamp']) > 0 else "N/A"
            ph_end = group['timestamp'].iloc[-1] if len(group['timestamp']) > 0 else "N/A"
            logger.info(f"[{phase}] 기간: {ph_start} ~ {ph_end} | 최종 자본: {ph_final:,.0f}원 | 수익률: {ph_return:.2f}% | 수익금: {ph_profit:,.0f}원 | 최대 낙폭: {ph_mdd:.2f}% | 거래: {ph_trades} | 승률: {ph_winrate:.1f}%")

    # 결과 저장
    df_results.to_csv('data/backtest_data/ml_backtest_results.csv', index=False)
    logger.info("백테스트 결과가 data/backtest_data/ml_backtest_results.csv에 저장되었습니다.")

    # 대시보드 연동: 종목/전략별 리포트도 함께 전송
    result_json = df_results.to_dict(orient='list')
    result_json['symbol'] = results.get('symbol', 'ML백테스트')
    # 종목별/전략별 리포트도 json에 추가
    symbol_report = {}
    if 'symbol' in df_results:
        for symbol, group in df_results.groupby('symbol'):
            sym_final = group['capital'].iloc[-1]
            sym_return = (sym_final - initial_capital) / initial_capital * 100
            sym_profit = sym_final - initial_capital
            sym_peak = group['capital'].max()
            sym_mdd = (sym_peak - group['capital'].min()) / sym_peak * 100
            sym_trades = len(group[group['position'] != 0])
            sym_win = len(group[group['actual_return'] > 0])
            sym_winrate = sym_win / sym_trades * 100 if sym_trades > 0 else 0
            symbol_report[symbol] = {
                'final_capital': sym_final,
                'return': sym_return,
                'profit': sym_profit,
                'max_drawdown': sym_mdd,
                'trades': sym_trades,
                'win_rate': sym_winrate
            }
    phase_report = {}
    if 'phase' in df_results:
        for phase, group in df_results.groupby('phase'):
            ph_final = group['capital'].iloc[-1]
            ph_return = (ph_final - initial_capital) / initial_capital * 100
            ph_profit = ph_final - initial_capital
            ph_peak = group['capital'].max()
            ph_mdd = (ph_peak - group['capital'].min()) / ph_peak * 100
            ph_trades = len(group[group['position'] != 0])
            ph_win = len(group[group['actual_return'] > 0])
            ph_winrate = ph_win / ph_trades * 100 if ph_trades > 0 else 0
            phase_report[phase] = {
                'final_capital': ph_final,
                'return': ph_return,
                'profit': ph_profit,
                'max_drawdown': ph_mdd,
                'trades': ph_trades,
                'win_rate': ph_winrate
            }
    result_json['symbol_report'] = symbol_report
    result_json['phase_report'] = phase_report
    with open('data/backtest_data/ml_backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    try:
        dashboard_url = 'http://34.47.77.230:5000/api/upload_results'
        resp = requests.post(dashboard_url, json=result_json, timeout=10)
        if resp.status_code == 200:
            logger.info(f"대시보드에 결과 업로드 성공: {dashboard_url}")
        else:
            logger.warning(f"대시보드 업로드 실패: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.warning(f"대시보드 업로드 중 오류: {e}")

def main():
    """메인 함수"""
    logger = setup_logging()
    logger.info("ML 모델 백테스트 시스템 시작")
    try:
        # 3년치 데이터 생성
        logger.info("히스토리컬 데이터 생성 중...")
        df = generate_historical_data(years=3)
        logger.info(f"데이터 생성 완료: {len(df)} 개 데이터, 기간: {df.index[0]} ~ {df.index[-1]}")

        # 저장된 모델이 있으면 불러오기, 없으면 새로 훈련
        model_path = 'trained_model.pkl'
        if os.path.exists(model_path):
            from ml.models.price_prediction_model import PricePredictionModel
            ml_model = PricePredictionModel.load_model(model_path)
            logger.info(f"저장된 모델({model_path})을 불러와서 백테스트를 진행합니다.")
        else:
            logger.info("새로운 모델 훈련을 시작합니다...")
            ml_model = PricePredictionModel()
            ml_model.fit(df)
            ml_model.save_model(model_path)
            logger.info(f"모델 훈련 완료 및 저장({model_path}) 후 백테스트를 진행합니다.")

        # ML 백테스트 실행
        logger.info("ML 백테스트 실행을 시작합니다...")
        results = run_ml_backtest(df, initial_capital=10000000, model=ml_model)
        logger.info("ML 백테스트 완료")
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        import traceback
        logger.error(f"상세 오류 정보: {traceback.format_exc()}")
        raise

# Optuna 로그 한글화 함수
def translate_optuna_log(msg):
    msg = re.sub(r'Trial (\d+) finished', r'트라이얼 \1 완료', msg)
    msg = re.sub(r'parameters:', '파라미터:', msg)
    msg = re.sub(r'Best is trial (\d+) with value:', r'최고 성능 트라이얼은 \1, 값:', msg)
    msg = re.sub(r'value:', '값:', msg)
    msg = re.sub(r'Trial (\d+) failed', r'트라이얼 \1 실패', msg)
    msg = re.sub(r'A new study created in memory with name:', '새로운 스터디 생성 (메모리 내 이름):', msg)
    return msg

# Optuna 로그를 한글로 출력하도록 stdout/stderr 후킹
class KoreanOptunaLogger(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        msg = translate_optuna_log(msg)
        print(msg)

optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = []
optuna_logger.addHandler(KoreanOptunaLogger())
optuna.logging.set_verbosity(optuna.logging.INFO)

if __name__ == "__main__":
    main() 