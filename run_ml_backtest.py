#!/usr/bin/env python3
"""
ML 모델 백테스트 실행 파일
몇 년치 백테스트가 가능한 ML 모델 테스트
"""

import sys
import os
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
from tqdm import tqdm
import time
import re
import optuna
import json, requests
import calendar

# 경고 메시지 필터링
warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# scikit-learn 경고 완전 제거
os.environ['PYTHONWARNINGS'] = 'ignore'

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.models.price_prediction_model import PricePredictionModel
from core.trading_engine.dynamic_leverage_manager import DynamicLeverageManager, MarketCondition, PhaseType
from data.market_data.data_generator import MarketDataGenerator
from utils.indicators.technical_indicators import TechnicalIndicators

def setup_logging():
    """
    로그 설정 (한국시간, 초기화 시에만 __main__ 표시)
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
    
    # 초기화 시에만 __main__ 표시, 백테스트 중에는 간단한 로그
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            dt = datetime.fromtimestamp(record.created, seoul_tz)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # 초기화 관련 로그만 __main__ 표시
            if '시스템 시작' in record.getMessage() or '데이터 생성' in record.getMessage() or '모델 불러오기' in record.getMessage() or '백테스트 시작' in record.getMessage():
                return f"{time_str} - __main__ - INFO - {record.getMessage()}"
            else:
                return f"{time_str} - {record.getMessage()}"
    
    formatter = CustomFormatter()
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
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

def send_log_to_dashboard(log_msg):
    try:
        dashboard_url = 'http://34.47.77.230:5000/api/realtime_log'
        requests.post(dashboard_url, json={'log': log_msg}, timeout=2)
    except Exception as e:
        logging.getLogger(__name__).warning(f"[대시보드 전송 실패] {e} | 로그: {log_msg}")

def send_report_to_dashboard(report_dict):
    try:
        dashboard_url = 'http://34.47.77.230:5000/api/report'
        requests.post(dashboard_url, json=report_dict, timeout=2)
    except Exception as e:
        pass

def run_ml_backtest(df: pd.DataFrame, initial_capital: float = 10000000, model=None, use_dynamic_position=False):
    logger = logging.getLogger(__name__)
    logger.info("ML 모델 백테스트 시작")

    ml_model = model if model is not None else PricePredictionModel()
    leverage_manager = DynamicLeverageManager()
    indicators = TechnicalIndicators()
    df_with_indicators = indicators.add_all_indicators(df.copy())
    # 멀티타임프레임 지표 생성 (1h, 4h, 5m)
    df_with_indicators = indicators.add_multi_timeframe_indicators(df_with_indicators, timeframes=[('1h',1),('4h',4),('5m',1/12)])

    # 실전형 다중 포지션 구조
    current_capital = initial_capital  # 현금성 자본
    positions = {}  # {(symbol, direction): {...}}
    trade_history = []  # 모든 진입/청산 기록
    realized_pnl = 0  # 실현손익
    unrealized_pnl = 0  # 미실현손익
    total_capital = initial_capital

    # 테스트용: 단일 종목(BNB/USDT)만 사용, 확장 시 symbol 컬럼 활용
    symbols = df_with_indicators['symbol'].unique() if 'symbol' in df_with_indicators else ['BNB/USDT']
    train_size = int(len(df_with_indicators) * 0.7)
    train_data = df_with_indicators.iloc[:train_size]
    test_data = df_with_indicators.iloc[train_size:]

    logger.info(f"훈련 데이터: {len(train_data)} 개, 테스트 데이터: {len(test_data)} 개")

    results = {
        'timestamp': [],
        'total_capital': [],
        'current_capital': [],
        'realized_pnl': [],
        'unrealized_pnl': [],
        'open_positions': [],
        'trade_log': []
    }

    # 월별 성과 추적
    monthly_performance = {}
    last_monthly_report = None
    trade_count = 0
    winning_trades = 0
    total_profit = 0
    peak_capital = initial_capital
    max_drawdown = 0

    # 크로노스 스위칭 신호 생성 함수
    def generate_chronos_signal(row, ml_pred):
        # 상위 프레임(4H) 추세 필터
        if not (row.get('ema_20_4h',0) > row.get('ema_50_4h',0) > row.get('ema_120_4h',0) and row.get('rsi_14_4h',0) > 50 and row.get('macd_4h',0) > row.get('macd_signal_4h',0)):
            return 0, "상위 프레임 상승 신호 불일치"
        # 중간 프레임(1H) 지지/저항, VWAP, 볼린저밴드 등
        if not (row.get('close',0) > row.get('vwap_1h',0) and row.get('close',0) > row.get('bb_lower_1h',0)):
            return 0, "중간 프레임 진입 조건 불충족"
        # 하위 프레임(5m) 트리거
        if not (row.get('stoch_k_5m',100) < 20 and row.get('stoch_d_5m',100) < 20 and row.get('stoch_k_5m',0) > row.get('stoch_d_5m',0)):
            return 0, "하위 프레임 트리거 없음"
        # ML 예측수익률까지 양수(매수)일 때만 진입
        if ml_pred > 0:
            return 1, "크로노스 스위칭 매수 신호"
        elif ml_pred < 0:
            return -1, "크로노스 스위칭 매도 신호"
        else:
            return 0, "신호 없음"

    for idx, row in test_data.iterrows():
        try:
            timestamp = row.name if hasattr(row, 'name') else row.get('timestamp', idx)
            # timestamp를 적절한 형식으로 변환
            if isinstance(timestamp, (int, float)):
                # 인덱스 번호인 경우 실제 날짜로 변환
                if 'timestamp' in row:
                    timestamp = row['timestamp']
                else:
                    # 인덱스 기반 날짜 생성 (테스트 데이터용)
                    start_date = datetime(2023, 1, 1)
                    timestamp = start_date + timedelta(hours=idx)
            elif isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()
            
            # timestamp가 datetime 객체인지 확인하고 한국시간으로 변환
            if isinstance(timestamp, datetime):
                if timestamp.tzinfo is None:
                    timestamp = pytz.timezone('Asia/Seoul').localize(timestamp)
                else:
                    timestamp = timestamp.astimezone(pytz.timezone('Asia/Seoul'))
            else:
                # 문자열인 경우 기본 datetime으로 변환
                timestamp = datetime.now(pytz.timezone('Asia/Seoul'))
            
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
            current_month = timestamp.strftime("%Y-%m")
            # 시장국면 판별
            regime = detect_market_regime(row)
            strategy_name, candidate_symbols = REGIME_STRATEGY_MAP.get(regime, ('mean_reversion', ['BTC']))
            symbol = row.get('symbol', candidate_symbols[0])
            if symbol not in candidate_symbols:
                symbol = candidate_symbols[0]
            regime_desc = f"시장국면: {regime}"
            strategy_desc = f"전략: {strategy_name}"
            # === 예측수익률 계산 ===
            prediction_data = df_with_indicators.iloc[:train_size + (idx - test_data.index[0]) + 1]
            predicted_return = 0
            if ml_model is not None and prediction_data is not None:
                if len(prediction_data) > 60:
                    try:
                        pred = ml_model.predict(prediction_data)
                        logger.info(f"[{timestamp_str}] ml_model.predict() 결과: {pred[-5:] if hasattr(pred, '__getitem__') else pred}")
                        predicted_return = pred[-1]
                    except Exception as e:
                        logger.error(f"[{timestamp_str}] ml_model.predict() 예외: {e}")
                else:
                    logger.info(f"[{timestamp_str}] 예측데이터 부족, predicted_return=0")
            # 크로노스 스위칭 신호 생성
            chrono_signal, chrono_reason = generate_chronos_signal(row, predicted_return)
            # 기존 신호와 결합(AND)
            if chrono_signal != 0:
                signal = chrono_signal
                reason = chrono_reason + f" | ML예측: {predicted_return:.2%}"
            else:
                signal, signal_desc = generate_trading_signal(predicted_return, row, 1.0)
                reason = signal_desc + f" | ML예측: {predicted_return:.2%}"
            direction = 'LONG' if signal == 1 else ('SHORT' if signal == -1 else None)
            # 진단용 로그 추가
            logger.info(f"[{timestamp_str}] 신호: {signal}, 방향: {direction}, 포지션존재: {positions.get((symbol, direction))}, 예측수익률: {predicted_return:.5f}, RSI: {row.get('rsi_14', 50):.2f}, 변동성: {row.get('volatility_20', 0.05):.4f}")
            # 동적 레버리지 계산
            current_leverage = leverage_manager.update_leverage(
                phase=PhaseType.PHASE1_AGGRESSIVE,
                market_condition=analyze_market_condition(row),
                current_capital=current_capital,
                peak_capital=initial_capital,
                consecutive_wins=0,
                consecutive_losses=0,
                volatility=row.get('volatility_20', 0.05),
                rsi=row.get('rsi_14', 50)
            )
            # 비중 결정
            base_ratio = 0.1
            if use_dynamic_position:
                position_ratio = get_dynamic_position_size(base_ratio, regime, predicted_return)
            else:
                position_ratio = base_ratio
            # 진입
            if direction and (symbol, direction) not in positions:
                entry_amount = current_capital * position_ratio
                if entry_amount < 1:
                    continue
                current_capital -= entry_amount
                positions[(symbol, direction)] = {
                    'entry_price': row['close'],
                    'entry_time': timestamp_str,
                    'leverage': current_leverage,
                    'amount': entry_amount,
                    'status': 'OPEN',
                    'strategy': strategy_name,
                    'regime': regime,
                    'reason': reason,
                    'position_ratio': position_ratio
                }
                log_msg = (
                    f"[{timestamp_str}] | 시장국면: {regime} | 전략: {strategy_name} | 레버리지: {current_leverage:.2f}배 | 비중: {position_ratio*100:.1f}% | 진입: {'매수' if direction=='LONG' else '매도'} | 종목: {symbol} | 진입가: {row['close']:,.2f} | 진입금액: {entry_amount:,.0f} | 남은자본: {current_capital:,.0f} | 신호근거: {reason}"
                )
                logger.info(log_msg)
                send_log_to_dashboard(log_msg)
                results['trade_log'].append(log_msg)
            # 청산
            if direction is None:
                for pos_key in list(positions.keys()):
                    if positions[pos_key]['status'] == 'OPEN':
                        entry = positions[pos_key]
                        entry_price = entry['entry_price']
                        entry_amount = entry['amount']
                        lev = entry['leverage']
                        pos_dir = pos_key[1]
                        # 손익 계산
                        if pos_dir == 'LONG':
                            pnl_rate = (row['close'] - entry_price) / entry_price * lev
                        else:
                            pnl_rate = (entry_price - row['close']) / entry_price * lev
                        profit = entry_amount * pnl_rate
                        current_capital += entry_amount + profit
                        realized_pnl += profit
                        entry['status'] = 'CLOSED'
                        entry['exit_price'] = row['close']
                        entry['exit_time'] = timestamp_str
                        entry['profit'] = profit
                        entry['pnl_rate'] = pnl_rate
                        log_msg = (
                            f"[{timestamp_str}] | 시장국면: {regime} | 전략: {strategy_name} | 레버리지: {lev:.2f}배 | 비중: {entry['position_ratio']*100:.1f}% | 청산: {'매수' if pos_dir=='LONG' else '매도'} | 종목: {pos_key[0]} | 진입가: {entry_price:,.2f} | 청산가: {row['close']:,.2f} | 수익률: {pnl_rate*100:+.2f}% | 수익금: {profit:+,.0f} | 총자산: {current_capital:,.0f} | 신호근거: {entry['reason']}"
                        )
                        logger.info(log_msg)
                        send_log_to_dashboard(log_msg)
                        results['trade_log'].append(log_msg)
                        trade_history.append({**entry, 'symbol': pos_key[0], 'direction': pos_dir})
                        # 월별 성과 업데이트 (청산 시)
                        if current_month in monthly_performance:
                            monthly_performance[current_month]['trade_log'].append(log_msg)
                        del positions[pos_key]

            # 미실현손익 계산 (모든 오픈 포지션 평가)
            unrealized_pnl = 0
            for pos_key, entry in positions.items():
                entry_price = entry['entry_price']
                entry_amount = entry['amount']
                lev = entry['leverage']
                pos_dir = pos_key[1]
                if pos_dir == 'LONG':
                    pnl_rate = (row['close'] - entry_price) / entry_price * lev
                else:
                    pnl_rate = (entry_price - row['close']) / entry_price * lev
                unrealized_pnl += entry_amount * pnl_rate

            # 총자산 = 현금성 자본 + 미실현손익 포함 오픈포지션 평가금액
            total_capital = current_capital + sum([entry['amount'] for entry in positions.values()]) + unrealized_pnl

            # 결과 저장 (항상 모든 key에 추가)
            results['timestamp'].append(timestamp_str)
            results['total_capital'].append(total_capital)
            results['current_capital'].append(current_capital)
            results['realized_pnl'].append(realized_pnl)
            results['unrealized_pnl'].append(unrealized_pnl)
            results['open_positions'].append(len(positions))

            # 월별 성과 추적
            if current_month not in monthly_performance:
                monthly_performance[current_month] = {
                    'total_capital': total_capital,
                    'current_capital': current_capital,
                    'realized_pnl': realized_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'open_positions': len(positions),
                    'trade_log': []
                }
            monthly_performance[current_month]['total_capital'] = total_capital
            monthly_performance[current_month]['current_capital'] = current_capital
            monthly_performance[current_month]['realized_pnl'] = realized_pnl
            monthly_performance[current_month]['unrealized_pnl'] = unrealized_pnl
            monthly_performance[current_month]['open_positions'] = len(positions)
            monthly_performance[current_month]['trade_log'].append(log_msg)

            # 월별 성과 분석
            if last_monthly_report is None:
                last_monthly_report = current_month
                trade_count = 1
                winning_trades = 1 if realized_pnl > 0 else 0
                total_profit = realized_pnl
                peak_capital = total_capital
                max_drawdown = 0 if realized_pnl > 0 else (peak_capital - total_capital) / peak_capital * 100
            else:
                if current_month != last_monthly_report:
                    # 월별 성과 보고
                    report_msg = f"[월간 리포트] {last_monthly_report} | 트레이드 수: {trade_count} | 최종 자산: {total_capital:,.0f}원 | 총 수익률: {(total_capital - monthly_performance[last_monthly_report]['total_capital']) / monthly_performance[last_monthly_report]['total_capital'] * 100:+.2f}% | 총 수익금: {(total_capital - monthly_performance[last_monthly_report]['total_capital']) - (monthly_performance[last_monthly_report]['realized_pnl'] + monthly_performance[last_monthly_report]['unrealized_pnl']):+,.0f}원 | 최대 낙폭: {max_drawdown:+.2f}%"
                    logger.info(report_msg)
                    send_log_to_dashboard(report_msg)
                    results['trade_log'].append(report_msg)
                    # 월별 성과 초기화
                    last_monthly_report = current_month
                    trade_count = 1
                    winning_trades = 1 if realized_pnl > 0 else 0
                    total_profit = realized_pnl
                    peak_capital = total_capital
                    max_drawdown = 0 if realized_pnl > 0 else (peak_capital - total_capital) / peak_capital * 100
                else:
                    trade_count += 1
                    winning_trades += 1 if realized_pnl > 0 else 0
                    total_profit += realized_pnl
                    peak_capital = max(peak_capital, total_capital)
                    max_drawdown = max(max_drawdown, (peak_capital - total_capital) / peak_capital * 100)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"[{idx}] 백테스트 중 오류 발생: {e}")
            logger.error(f"[{idx}] 상세 오류 정보: {error_details}")
            # 예외 발생 시에도 각 리스트에 None 등으로 추가
            results['timestamp'].append(timestamp_str if 'timestamp_str' in locals() else None)
            results['total_capital'].append(None)
            results['current_capital'].append(None)
            results['realized_pnl'].append(None)
            results['unrealized_pnl'].append(None)
            results['open_positions'].append(None)
            continue
    # 루프 종료 후, 모든 리스트 길이 맞추기(가장 짧은 길이에 맞춰 자르기)
    min_len = min(len(v) for v in results.values())
    for k in results:
        results[k] = results[k][:min_len]

    # 결과 분석 및 리포트
    analyze_backtest_results(results, initial_capital)
    
    # 마지막 월 성과보고서 출력
    if last_monthly_report and last_monthly_report in monthly_performance:
        final_report_msg = f"=== {last_monthly_report} 최종 성과보고서 ==="
        logger.info(final_report_msg)
        send_log_to_dashboard(final_report_msg)
        results['trade_log'].append(final_report_msg)
        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
        # 최대 낙폭 부호 명확히
        max_drawdown_str = f"{max_drawdown:+.2f}%" if max_drawdown != 0 else "0.00%"
        final_report_detail = (
            f"{last_monthly_report} | 총 트레이드: {trade_count} | 승률: {win_rate:.1f}% | 최종 자산: {total_capital:,.0f}원 | 총 수익금: {total_profit:+,.0f}원 | 최대 낙폭: {max_drawdown_str}"
        )
        logger.info(final_report_detail)
        send_log_to_dashboard(final_report_detail)
        results['trade_log'].append(final_report_detail)
    
    # 최종 자본을 results에 추가
    try:
        df_results = pd.DataFrame(results)
        if not df_results.empty and 'total_capital' in df_results:
            results['final_capital'] = df_results['total_capital'].iloc[-1]
    except Exception:
        results['final_capital'] = None
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
        return MarketCondition.BULL
    elif rsi < 30 and ma_20 < ma_50 and volatility < 0.08:
        return MarketCondition.BEAR
    elif volatility > 0.10:
        return MarketCondition.HIGH_VOLATILITY
    elif volatility < 0.03:
        return MarketCondition.LOW_VOLATILITY
    else:
        return MarketCondition.SIDEWAYS

def generate_trading_signal(predicted_return: float, row: pd.Series, leverage: float):
    """거래 신호 생성 + 한글 전략 설명 반환"""
    threshold = 0.0002  # 기존 0.001에서 완화
    rsi = row.get('rsi_14', 50)
    volatility = row.get('volatility_20', 0.05)
    # 과매수/과매도 조건 완화
    if rsi > 85:
        return 0, "RSI 과매수 조건"
    if rsi < 15:
        return 0, "RSI 과매도 조건"
    # 고변동성 조건 완화
    if volatility > 0.2:
        return 0, "고변동성 조건"
    # 신호 생성
    if predicted_return > threshold:
        return 1, "예측 수익률 상승 신호"
    elif predicted_return < -threshold:
        return -1, "예측 수익률 하락 신호"
    else:
        return 0, "기본 전략 (신호 없음)"

def analyze_backtest_results(results: dict, initial_capital: float):
    """백테스트 결과 분석 (한글 실전형)"""
    logger = logging.getLogger(__name__)
    df_results = pd.DataFrame(results)
    if df_results.empty or len(df_results['total_capital']) == 0:
        logger.error("백테스트 결과 데이터가 비어 있습니다. (루프 내 예외/데이터 없음 등 원인)")
        return
    final_capital = df_results['total_capital'].dropna().iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100
    profit = final_capital - initial_capital
    peak_capital = df_results['total_capital'].max()
    min_capital = df_results['total_capital'].min()
    max_drawdown = (peak_capital - min_capital) / peak_capital * 100
    profitable_trades = len([x for x in df_results['realized_pnl'] if x is not None and x > 0])
    total_trades = len([x for x in df_results['realized_pnl'] if x is not None])
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
    start_time = df_results['timestamp'].iloc[0] if len(df_results['timestamp']) > 0 else "N/A"
    end_time = df_results['timestamp'].iloc[-1] if len(df_results['timestamp']) > 0 else "N/A"
    logger.info("\n=== 백테스트 실전형 성과 요약 ===")
    logger.info(f"기간: {start_time} ~ {end_time}")
    logger.info(f"최종 자산: {final_capital:,.0f}원 | 총 수익률: {total_return:+.2f}% | 총 수익금: {profit:+,.0f}원 | 최대 낙폭: {max_drawdown:+.2f}% | 총 거래: {total_trades}회 | 승률: {win_rate:.1f}%")
    # 종목별/전략별/국면별 등 추가 요약 필요시 여기에 한글 실전형 포맷으로 추가

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
            sym_final = group['total_capital'].iloc[-1]
            sym_return = (sym_final - initial_capital) / initial_capital * 100
            sym_profit = sym_final - initial_capital
            sym_peak = group['total_capital'].max()
            sym_mdd = (sym_peak - group['total_capital'].min()) / sym_peak * 100
            sym_trades = len(group[group['direction'] != None])
            sym_win = len(group[group['realized_pnl'] > 0])
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
            ph_final = group['total_capital'].iloc[-1]
            ph_return = (ph_final - initial_capital) / initial_capital * 100
            ph_profit = ph_final - initial_capital
            ph_peak = group['total_capital'].max()
            ph_mdd = (ph_peak - group['total_capital'].min()) / ph_peak * 100
            ph_trades = len(group[group['direction'] != None])
            ph_win = len(group[group['realized_pnl'] > 0])
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
        logger.info("3년치 과거 데이터 생성 시작")
        df = generate_historical_data(years=3)
        logger.info(f"데이터 생성 완료: {len(df)} 개 데이터 포인트")

        # 모델 로딩/학습 분기
        model_path = 'trained_model.pkl'
        if os.path.exists(model_path):
            ml_model = PricePredictionModel.load_model(model_path)
            logger.info(f"저장된 모델({model_path})을 불러와서 백테스트를 진행합니다.")
        else:
            ml_model = PricePredictionModel()
            ml_model.fit(df)
            ml_model.save_model(model_path)
            logger.info(f"모델을 새로 훈련 후 저장하고 백테스트를 진행합니다.")

        # ML 백테스트 실행
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

# === 시장국면 5단계 분류 함수 ===
def detect_market_regime(row: pd.Series) -> str:
    """가격 변화율, 변동성 등으로 시장국면 5단계(급등/상승/횡보/하락/급락) 분류"""
    pct = row.get('return_1d', 0)
    vol = row.get('volatility_20', 0.05)
    # 기준값은 실전 데이터에 맞게 조정 가능
    if pct > 0.04 and vol > 0.10:
        return '급등'
    elif pct > 0.01:
        return '상승'
    elif pct < -0.04 and vol > 0.10:
        return '급락'
    elif pct < -0.01:
        return '하락'
    else:
        return '횡보'

# === 시장국면별 전략/종목군 매핑 ===
REGIME_STRATEGY_MAP = {
    '급등':   ('momentum_breakout', ['BNB', 'SOL', 'ETH']),
    '상승':   ('trend_following',   ['BTC', 'ETH', 'SOL']),
    '횡보':   ('mean_reversion',    ['USDT', 'BTC', 'ETH']),
    '하락':   ('short_momentum',    ['BTC', 'XRP', 'ADA']),
    '급락':   ('btc_short_only',    ['BTC']),
}

# === 동적 비중 함수 ===
def get_dynamic_position_size(base_ratio, market_condition, predicted_return):
    # 시장국면별 조정
    if market_condition in ['급등', '상승']:
        base_ratio *= 1.5
    elif market_condition in ['하락', '급락']:
        base_ratio *= 0.5
    # 예측수익률에 따라 추가 조정
    if abs(predicted_return) > 0.01:
        base_ratio *= 1.3
    elif abs(predicted_return) < 0.002:
        base_ratio *= 0.7
    # 최대/최소 비중 제한
    base_ratio = min(max(base_ratio, 0.03), 0.3)  # 3%~30%
    return base_ratio

def print_summary(result, label):
    """실전형 한글 요약 출력"""
    print(f"[요약] {label} | 최종 자산: {result['final_capital']:,.0f}원 | 총 수익률: {result['total_return']:+.2f}% | 총 수익금: {result['final_capital']-result['initial_capital']:+,.0f}원 | 최대 낙폭: {result['max_drawdown']:+.2f}% | 거래: {result['total_trades']}회 | 승률: {result['win_rate']:.1f}%")

if __name__ == "__main__":
    import pandas as pd
    # 데이터 로드 (예시)
    df = pd.read_csv('data/market_data/BNB_USDT_1h.csv')
    initial_capital = 10000000
    print("[고정비중 백테스트]")
    result_fixed = run_ml_backtest(df, initial_capital=initial_capital, use_dynamic_position=False)
    print("[동적비중 백테스트]")
    result_dynamic = run_ml_backtest(df, initial_capital=initial_capital, use_dynamic_position=True)
    # 결과 비교 출력 (예시)
    print_summary(result_fixed, '고정비중')
    print_summary(result_dynamic, '동적비중') 