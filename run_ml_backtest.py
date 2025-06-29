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
    except Exception:
        pass  # 에러 무시, 아무 메시지도 출력하지 않음

def send_report_to_dashboard(report_dict):
    try:
        dashboard_url = 'http://34.47.77.230:5000/api/report'
        requests.post(dashboard_url, json=report_dict, timeout=2)
    except Exception as e:
        pass

def run_ml_backtest(df: pd.DataFrame, initial_capital: float = 10000000, model=None, use_dynamic_position=False):
    logger = logging.getLogger(__name__)
    logger.info("ML 모델 백테스트 시작")

    # ML 모델 초기화 및 검증
    ml_model = model if model is not None else PricePredictionModel()
    if not hasattr(ml_model, 'models') or not ml_model.models:
        logger.info("ML 모델 초기화 중...")
        ml_model = PricePredictionModel()
    
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
    
    # 초기 ML 모델 훈련 (충분한 데이터가 있는 경우)
    if len(train_data) >= 50:  # 최소 요구사항을 낮춤
        logger.info("초기 ML 모델 훈련 시작...")
        initial_training_success = ml_model.fit(train_data)
        if initial_training_success:
            logger.info("초기 ML 모델 훈련 완료")
        else:
            logger.warning("초기 ML 모델 훈련 실패 - 데이터 부족")
    else:
        logger.warning(f"초기 훈련 데이터 부족 ({len(train_data)}개) - 백테스트 중 훈련 예정")

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

    # 리스크 추적 변수
    daily_pnl = 0
    weekly_pnl = 0
    monthly_pnl = 0
    last_daily_reset = None
    last_weekly_reset = None
    last_monthly_reset = None

    # 크로노스 스위칭 신호 생성 함수 (통합 고수익 전략)
    def generate_chronos_signal(row, ml_pred):
        # 상위 프레임(4H) 추세 필터 - 더 엄격한 조건
        ema_trend = (row.get('ema_20_4h',0) > row.get('ema_50_4h',0) > row.get('ema_120_4h',0))
        rsi_bull = row.get('rsi_14_4h',50) > 50 and row.get('rsi_14_4h',100) < 80
        macd_bull = row.get('macd_4h',0) > row.get('macd_signal_4h',0) and row.get('macd_4h',0) > 0
        
        if not (ema_trend and rsi_bull and macd_bull):
            return 0, "상위 프레임 상승 신호 불일치"
        
        # 중간 프레임(1H) 지지/저항, VWAP, 볼린저밴드 등 - 더 정교한 조건
        vwap_support = row.get('close',0) > row.get('vwap_1h',0) * 1.001  # VWAP 0.1% 이상 상승
        bb_support = row.get('close',0) > row.get('bb_lower_1h',0) * 1.002  # 볼린저 하단 0.2% 이상
        volume_support = row.get('volume',0) > row.get('volume_ma_5',0) * 1.2  # 거래량 20% 이상 증가
        
        if not (vwap_support and bb_support and volume_support):
            return 0, "중간 프레임 진입 조건 불충족"
        
        # 하위 프레임(5m) 트리거 - 더 민감한 조건
        stoch_oversold = row.get('stoch_k_5m',100) < 25 and row.get('stoch_d_5m',100) < 25
        stoch_bullish = row.get('stoch_k_5m',0) > row.get('stoch_d_5m',0) and row.get('stoch_k_5m',0) > 20
        rsi_5m_bull = row.get('rsi_14_5m',50) > 40 and row.get('rsi_14_5m',100) < 70
        
        if not (stoch_oversold and stoch_bullish and rsi_5m_bull):
            return 0, "하위 프레임 트리거 없음"
        
        # ML 예측수익률 기반 신호 강도 판단
        if ml_pred > 0.01:  # 강한 매수 신호
            return 2, "크로노스 스위칭 강한 매수 신호"
        elif ml_pred > 0.005:  # 중간 매수 신호
            return 1, "크로노스 스위칭 매수 신호"
        elif ml_pred < -0.01:  # 강한 매도 신호
            return -2, "크로노스 스위칭 강한 매도 신호"
        elif ml_pred < -0.005:  # 중간 매도 신호
            return -1, "크로노스 스위칭 매도 신호"
        else:
            return 0, "신호 없음"

    for idx, row in test_data.iterrows():
        try:
            # timestamp를 적절한 형식으로 변환
            if 'timestamp' in row and pd.notnull(row['timestamp']):
                try:
                    timestamp = pd.to_datetime(row['timestamp'])
                except Exception:
                    timestamp = row['timestamp']
            elif isinstance(row.name, (pd.Timestamp, datetime)):
                timestamp = row.name
            else:
                # 인덱스 기반 날짜 생성 (테스트 데이터용)
                start_date = datetime(2023, 1, 1)
                timestamp = start_date + timedelta(hours=idx)
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
                if len(prediction_data) > 50:  # 최소 데이터 요구사항을 낮춤
                    try:
                        # 모델이 훈련되지 않은 경우 훈련
                        if not hasattr(ml_model, 'feature_names') or ml_model.feature_names is None:
                            logger.info(f"[{timestamp_str}] ML 모델 훈련 중...")
                            training_success = ml_model.fit(prediction_data)
                            if training_success:
                                logger.info(f"[{timestamp_str}] ML 모델 훈련 완료")
                            else:
                                predicted_return = 0
                                continue
                        
                        # 모델 훈련 상태 재확인
                        if hasattr(ml_model, 'feature_names') and ml_model.feature_names is not None:
                            pred = ml_model.predict(prediction_data)
                            if pred is not None and len(pred) > 0:
                                predicted_return = pred[-1]
                            else:
                                predicted_return = 0
                        else:
                            predicted_return = 0
                    except Exception as e:
                        predicted_return = 0
                else:
                    predicted_return = 0
            # 크로노스 스위칭 신호 생성
            chrono_signal, chrono_reason = generate_chronos_signal(row, predicted_return)
            # 기존 신호와 결합(AND)
            if chrono_signal != 0:
                signal = chrono_signal
                reason = chrono_reason + f" | ML예측: {predicted_return*100:.2f}%"
            else:
                signal, signal_desc = generate_trading_signal(predicted_return, row, 1.0)
                reason = signal_desc + f" | ML예측: {predicted_return*100:.2f}%"
            direction = 'LONG' if signal == 1 else ('SHORT' if signal == -1 else None)
            
            # 매매 현황 로그 (매 100번째마다 출력)
            if idx % 100 == 0:
                open_positions_count = len([p for p in positions.values() if p['status'] == 'OPEN'])
                total_pnl = realized_pnl + unrealized_pnl
                pnl_rate = (total_pnl / initial_capital) * 100
                logger.info(f"[{timestamp_str}] === 매매 현황 === | 총자산: {current_capital:,.0f} | 실현손익: {realized_pnl:+,.0f} | 미실현손익: {unrealized_pnl:+,.0f} | 수익률: {pnl_rate:+.2f}% | 보유포지션: {open_positions_count}개")
                if positions:
                    logger.info("┌────────┬─────┬────────┬────────┬────────┬────────┬────────┬────────┐")
                    logger.info("│  종목  │ 방향 │ 진입가 │ 현재가 │ 평가손익 │ 수익률 │ 레버리지 │ 진입시각 │")
                    logger.info("├────────┼─────┼────────┼────────┼────────┼────────┼────────┼────────┤")
                    for pos_key, entry in positions.items():
                        profit = (row['close'] - entry['entry_price']) * entry['amount'] if pos_key[1] == 'LONG' else (entry['entry_price'] - row['close']) * entry['amount']
                        pnl_rate = (row['close'] - entry['entry_price']) / entry['entry_price'] * 100 if pos_key[1] == 'LONG' else (entry['entry_price'] - row['close']) / entry['entry_price'] * 100
                        logger.info(f"│ {pos_key[0]:^6} │ {pos_key[1]:^4} │ {entry['entry_price']:>8.2f} │ {row['close']:>8.2f} │ {profit:>8,.0f} │ {pnl_rate:>6.2f}% │ {entry['leverage']:>6.2f} │ {entry['entry_time']} │")
                    logger.info("└────────┴─────┴────────┴────────┴────────┴────────┴────────┴────────┘")
            
            # 동적 레버리지 계산 (시장국면별)
            current_leverage = get_dynamic_leverage(regime, predicted_return, row.get('volatility_20', 0.05))
            # 비중 결정
            base_ratio = 0.1
            if use_dynamic_position:
                position_ratio = get_dynamic_position_size(predicted_return, signal)
            else:
                position_ratio = base_ratio
            # 진입
            if direction and (symbol, direction) not in positions:
                # 리스크 한도 체크
                risk_ok, risk_msg = check_risk_limits(current_capital, initial_capital, daily_pnl, weekly_pnl, monthly_pnl)
                if not risk_ok:
                    logger.info(f"[{timestamp_str}] | 리스크 한도 초과: {risk_msg} | 거래 중단")
                    continue
                
                # 실전형 손절/익절 계산
                stop_loss, take_profit = get_risk_management(current_leverage, predicted_return)
                
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
                    'position_ratio': position_ratio,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'peak_price': row['close'],  # 트레일링 스탑용
                    'pyramiding_count': 0,  # 피라미딩 횟수
                    'direction': direction
                }
                log_msg = (
                    f"[{timestamp_str}] | {'진입':^4} | {regime:^4} | {STRATEGY_KOR_MAP.get(strategy_name, strategy_name):^10} | {'매수' if direction=='LONG' else '매도':^4} | {symbol:^6} | "
                    f"{row['close']:>8,.2f} | {'-':>8} | {'-':>7} | {'-':>8} | {current_capital:>10,.0f} | {position_ratio*100:>5.1f}% | {current_leverage:>4.2f}배 | {reason} | {predicted_return*100:.2f}%"
                )
                logger.info(log_msg)
                send_log_to_dashboard(log_msg)
                results['trade_log'].append(log_msg)
            
            # 피라미딩 체크 (기존 포지션에 추가 진입)
            for pos_key in list(positions.keys()):
                if positions[pos_key]['status'] == 'OPEN':
                    entry = positions[pos_key]
                    entry_price = entry['entry_price']
                    entry_amount = entry['amount']
                    current_price = row['close']
                    
                    # 수익률 계산
                    if pos_key[1] == 'LONG':
                        profit_rate = (current_price - entry_price) / entry_price
                    else:
                        profit_rate = (entry_price - current_price) / entry_price
                    
                    # 피라미딩 조건 체크
                    should_pyramid, additional_amount = check_pyramiding(positions, pos_key[0], pos_key[1], profit_rate)
                    if should_pyramid and additional_amount > 0 and current_capital >= additional_amount:
                        current_capital -= additional_amount
                        entry['amount'] += additional_amount
                        entry['pyramiding_count'] += 1
                        entry['peak_price'] = max(entry['peak_price'], current_price)
                        
                        # 피라미딩 로그 표 형식 통일
                        pyramid_log = (
                            f"[{timestamp_str}] | {'피라':^4} | {regime:^4} | {STRATEGY_KOR_MAP.get(strategy_name, strategy_name):^10} | {'매수' if pos_key[1]=='LONG' else '매도':^4} | {pos_key[0]:^6} | "
                            f"{entry_price:>8,.2f} | {'-':>8} | {profit_rate*100:+.2f}% | {additional_amount:>8,.0f} | {current_capital:>10,.0f} | {entry['position_ratio']*100:>5.1f}% | {entry['leverage']:>4.2f}배 | 피라미딩 조건충족 | - | {entry['pyramiding_count']}회"
                        )
                        logger.info(pyramid_log)
                        send_log_to_dashboard(pyramid_log)
                        results['trade_log'].append(pyramid_log)
            
            # 청산 조건 체크 (신호 없음, 손절, 익절, 트레일링 스탑)
            if direction is None:
                for pos_key in list(positions.keys()):
                    if positions[pos_key]['status'] == 'OPEN':
                        entry = positions[pos_key]
                        entry_price = entry['entry_price']
                        entry_amount = entry['amount']
                        lev = entry['leverage']
                        pos_dir = entry['direction']
                        current_price = row['close']
                        
                        # 손익 계산
                        if pos_dir == 'LONG':
                            pnl_rate = (current_price - entry_price) / entry_price * lev
                        else:
                            pnl_rate = (entry_price - current_price) / entry_price * lev
                        
                        # 청산 조건 체크
                        should_close = False
                        close_reason = ""
                        
                        # 손절 체크
                        if pnl_rate <= -entry['stop_loss']:
                            should_close = True
                            close_reason = "손절"
                        
                        # 익절 체크
                        elif pnl_rate >= entry['take_profit']:
                            should_close = True
                            close_reason = "익절"
                        
                        # 트레일링 스탑 체크
                        elif check_trailing_stop(positions, pos_key[0], pos_dir, current_price):
                            should_close = True
                            close_reason = "트레일링 스탑"
                        
                        if should_close:
                            profit = entry_amount * pnl_rate
                            current_capital += entry_amount + profit
                            realized_pnl += profit
                            
                            # 리스크 추적 업데이트
                            daily_pnl += profit
                            weekly_pnl += profit
                            monthly_pnl += profit
                            
                            entry['status'] = 'CLOSED'
                            entry['exit_price'] = current_price
                            entry['exit_time'] = timestamp_str
                            entry['profit'] = profit
                            entry['pnl_rate'] = pnl_rate
                            entry['close_reason'] = close_reason
                            
                            log_msg = (
                                f"[{timestamp_str}] | {'청산':^4} | {regime:^4} | {STRATEGY_KOR_MAP.get(strategy_name, strategy_name):^10} | {'매수' if pos_dir=='LONG' else '매도':^4} | {pos_key[0]:^6} | "
                                f"{entry_price:>8,.2f} | {current_price:>8,.2f} | {pnl_rate*100:+.2f}% | {profit:+,.0f} | {current_capital:>10,.0f} | {entry['position_ratio']*100:>5.1f}% | {lev:>4.2f}배 | {close_reason} | {predicted_return*100:.2f}%"
                            )
                            logger.info(log_msg)
                            send_log_to_dashboard(log_msg)
                            results['trade_log'].append(log_msg)
                            trade_history.append({**entry, 'symbol': pos_key[0], 'direction': pos_dir})
                            
                            # 거래 통계 업데이트 (청산 시에만)
                            trade_count += 1
                            if profit > 0:
                                winning_trades += 1
                            total_profit += profit
                            peak_capital = max(peak_capital, total_capital)
                            max_drawdown = max(max_drawdown, (peak_capital - total_capital) / peak_capital * 100) if peak_capital > 0 else 0
            
            # 리스크 추적 리셋 (일/주/월)
            current_date = timestamp.date()
            if last_daily_reset != current_date:
                daily_pnl = 0
                last_daily_reset = current_date
            
            if last_weekly_reset is None or (current_date - last_weekly_reset).days >= 7:
                weekly_pnl = 0
                last_weekly_reset = current_date
            
            if last_monthly_reset is None or (current_date - last_monthly_reset).days >= 30:
                monthly_pnl = 0
                last_monthly_reset = current_date

            # 미실현손익 계산 (모든 오픈 포지션 평가)
            unrealized_pnl = 0
            for pos_key, entry in positions.items():
                entry_price = entry['entry_price']
                entry_amount = entry['amount']
                lev = entry['leverage']
                pos_dir = entry['direction']
                if pos_dir == 'LONG':
                    pnl_rate = (row['close'] - entry_price) / entry_price * lev
                else:
                    pnl_rate = (entry_price - row['close']) / entry_price * lev
                unrealized_pnl += entry_amount * pnl_rate

            # 총자산 = 현금성 자본 + 미실현손익 포함 오픈포지션 평가금액
            current_position_value = sum([
                (row['close'] - entry['entry_price']) * entry['amount'] if entry.get('status')=='OPEN' and entry.get('direction')=='LONG' else
                (entry['entry_price'] - row['close']) * entry['amount'] if entry.get('status')=='OPEN' and entry.get('direction')=='SHORT' else 0
                for entry in positions.values()
            ])
            total_capital = current_capital + current_position_value + unrealized_pnl

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
                    'trade_count': 0,
                    'winning_trades': 0,
                    'trade_log': []
                }
            monthly_performance[current_month]['total_capital'] = total_capital
            monthly_performance[current_month]['current_capital'] = current_capital
            monthly_performance[current_month]['realized_pnl'] = realized_pnl
            monthly_performance[current_month]['unrealized_pnl'] = unrealized_pnl
            monthly_performance[current_month]['open_positions'] = len(positions)
            if 'log_msg' in locals():
                monthly_performance[current_month]['trade_log'].append(log_msg)

            # 월별 성과 분석
            if last_monthly_report is None:
                last_monthly_report = current_month
                trade_count = 0
                winning_trades = 0
                total_profit = 0
                peak_capital = total_capital
                max_drawdown = 0
            else:
                if current_month != last_monthly_report:
                    # 월별 성과 보고 (승률 포함)
                    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
                    monthly_return = ((total_capital - monthly_performance[last_monthly_report]['total_capital']) / monthly_performance[last_monthly_report]['total_capital'] * 100) if monthly_performance[last_monthly_report]['total_capital'] > 0 else 0
                    monthly_profit = (total_capital - monthly_performance[last_monthly_report]['total_capital']) - (monthly_performance[last_monthly_report]['realized_pnl'] + monthly_performance[last_monthly_report]['unrealized_pnl'])
                    
                    report_msg = f"[월간 리포트] {last_monthly_report} | 거래수: {trade_count} | 승률: {win_rate:.1f}% | 최종자산: {total_capital:,.0f}원 | 수익률: {monthly_return:+.2f}% | 수익금: {monthly_profit:+,.0f}원 | 최대낙폭: {max_drawdown:+.2f}%"
                    logger.info(report_msg)
                    send_log_to_dashboard(report_msg)
                    results['trade_log'].append(report_msg)
                    
                    # 월별 성과 초기화
                    last_monthly_report = current_month
                    trade_count = 0
                    winning_trades = 0
                    total_profit = 0
                    peak_capital = total_capital
                    max_drawdown = 0
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
    """백테스트 결과 분석 (통합 고수익 전략)"""
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
    
    # 월별 수익률 계산
    monthly_returns = []
    if 'timestamp' in df_results and len(df_results['timestamp']) > 0:
        df_results['date'] = pd.to_datetime(df_results['timestamp'])
        df_results['month'] = df_results['date'].dt.to_period('M')
        monthly_data = df_results.groupby('month')['total_capital'].agg(['first', 'last'])
        monthly_returns = ((monthly_data['last'] - monthly_data['first']) / monthly_data['first'] * 100).tolist()
    
    avg_monthly_return = np.mean(monthly_returns) if monthly_returns else 0
    max_monthly_return = max(monthly_returns) if monthly_returns else 0
    min_monthly_return = min(monthly_returns) if monthly_returns else 0
    
    # 샤프 비율 계산 (간단한 버전)
    returns = df_results['total_capital'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
    
    start_time = df_results['timestamp'].iloc[0] if len(df_results['timestamp']) > 0 else "N/A"
    end_time = df_results['timestamp'].iloc[-1] if len(df_results['timestamp']) > 0 else "N/A"
    
    logger.info("\n" + "="*80)
    logger.info("🚀 통합 고수익 전략 백테스트 결과 (목표: 월 수익률 25~35%)")
    logger.info("="*80)
    logger.info(f"📅 백테스트 기간: {start_time} ~ {end_time}")
    logger.info(f"💰 최종 자산: {final_capital:,.0f}원 | 총 수익률: {total_return:+.2f}% | 총 수익금: {profit:+,.0f}원")
    logger.info(f"📊 월 평균 수익률: {avg_monthly_return:+.2f}% | 최고 월: {max_monthly_return:+.2f}% | 최저 월: {min_monthly_return:+.2f}%")
    logger.info(f"⚠️  최대 낙폭: {max_drawdown:+.2f}% | 샤프 비율: {sharpe_ratio:.2f}")
    logger.info(f"🎯 총 거래: {total_trades}회 | 승률: {win_rate:.1f}% | 수익거래: {profitable_trades}회")
    
    # 목표 달성도 평가
    target_achieved = "✅ 달성" if avg_monthly_return >= 25 else "❌ 미달성"
    logger.info(f"🎯 월 수익률 25% 목표: {target_achieved} (현재: {avg_monthly_return:.1f}%)")
    
    # 리스크 평가
    if max_drawdown <= 10:
        risk_level = "🟢 낮음"
    elif max_drawdown <= 15:
        risk_level = "🟡 보통"
    else:
        risk_level = "🔴 높음"
    logger.info(f"⚠️  리스크 수준: {risk_level} (최대 낙폭: {max_drawdown:.1f}%)")
    
    # 전략 효과 분석
    logger.info("\n" + "-"*60)
    logger.info("📈 전략 효과 분석")
    logger.info("-"*60)
    
    if avg_monthly_return >= 25:
        logger.info("✅ 크로노스 스위칭 + 동적 레버리지 전략 효과 우수")
        logger.info("✅ 피라미딩 + 트레일링 스탑으로 수익 극대화")
        logger.info("✅ 실전형 리스크 관리로 안정성 확보")
    elif avg_monthly_return >= 15:
        logger.info("⚠️  전략 효과 보통 - 파라미터 최적화 필요")
        logger.info("💡 제안: 레버리지 범위 확대 또는 신호 민감도 조정")
    else:
        logger.info("❌ 전략 효과 부족 - 전면 재검토 필요")
        logger.info("💡 제안: 시장국면별 전략 분리 또는 ML 모델 재훈련")
    
    # 개선 제안
    logger.info("\n" + "-"*60)
    logger.info("💡 성과 개선 제안")
    logger.info("-"*60)
    
    if win_rate < 70:
        logger.info("🎯 승률 개선: 신호 필터링 강화, 다중시간 조건 엄격화")
    
    if max_drawdown > 12:
        logger.info("🛡️ 리스크 관리: 손절폭 축소, 레버리지 범위 축소")
    
    if avg_monthly_return < 25:
        logger.info("📈 수익률 개선: 피라미딩 조건 완화, 익절폭 확대")
    
    if sharpe_ratio < 2.0:
        logger.info("⚖️ 샤프 비율 개선: 변동성 대비 수익률 최적화")
    
    logger.info("\n" + "="*80)
    logger.info("🎯 상위 0.01% 통합 고수익 전략 분석 완료!")
    logger.info("="*80)
    
    # 결과 저장
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'avg_monthly_return': avg_monthly_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'target_achieved': avg_monthly_return >= 25
    }

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

# === 동적 레버리지 계산 (시장국면별)
def get_dynamic_leverage(regime, ml_pred, volatility):
    base_leverage = 1.0
    if regime == "급등":
        base_leverage = 2.5
    elif regime == "상승":
        base_leverage = 2.0
    elif regime == "횡보":
        base_leverage = 1.5
    elif regime == "하락":
        base_leverage = 1.0
    elif regime == "급락":
        base_leverage = 0.8
    
    # ML 예측수익률에 따른 조정
    if abs(ml_pred) > 0.01:
        base_leverage *= 1.2
    elif abs(ml_pred) < 0.002:
        base_leverage *= 0.8
    
    # 변동성에 따른 조정
    if volatility > 0.15:
        base_leverage *= 0.7
    elif volatility < 0.05:
        base_leverage *= 1.1
    
    return min(max(base_leverage, 0.5), 3.0)  # 0.5~3.0배 범위

# === 동적 비중 계산 (ML 예측수익률 기반)
def get_dynamic_position_size(ml_pred, signal_strength):
    base_size = 0.05  # 기본 5%
    
    # ML 예측 신호 강도에 따른 비중 조절
    if abs(ml_pred) > 0.02:  # 강한 신호 (2% 이상)
        if signal_strength == 2:  # 강한 신호
            return 0.20  # 20%
        elif signal_strength == 1:  # 중간 신호
            return 0.15  # 15%
        else:
            return 0.12  # 12%
    elif abs(ml_pred) > 0.01:  # 중간 신호 (1-2%)
        if signal_strength == 2:
            return 0.15  # 15%
        elif signal_strength == 1:
            return 0.12  # 12%
        else:
            return 0.10  # 10%
    elif abs(ml_pred) > 0.005:  # 약한 신호 (0.5-1%)
        if signal_strength == 2:
            return 0.12  # 12%
        elif signal_strength == 1:
            return 0.10  # 10%
        else:
            return 0.08  # 8%
    else:  # 매우 약한 신호
        if signal_strength == 2:
            return 0.08  # 8%
        elif signal_strength == 1:
            return 0.06  # 6%
        else:
            return base_size  # 5%

# 실전형 손절/익절 계산 (레버리지 반영)
def get_risk_management(leverage, ml_pred):
    # 레버리지 반영 손절/익절
    stop_loss = 0.02 / leverage  # 레버리지가 높을수록 손절폭 좁아짐
    take_profit = 0.05 * leverage  # 레버리지가 높을수록 익절폭 넓어짐
    
    # ML 예측수익률에 따른 조정
    if abs(ml_pred) > 0.01:
        take_profit *= 1.3  # 강한 신호 시 익절폭 확대
    elif abs(ml_pred) < 0.002:
        stop_loss *= 0.8  # 약한 신호 시 손절폭 축소
    
    return stop_loss, take_profit

# 피라미딩 전략 (최대 5회, 조건 완화)
def check_pyramiding(positions, symbol, direction, current_profit_rate):
    if (symbol, direction) not in positions:
        return False, 0
    position = positions[(symbol, direction)]
    entry_amount = position['amount']
    pyramiding_count = position.get('pyramiding_count', 0)
    # 피라미딩 조건: 2% 이상 수익, 최대 5회
    if pyramiding_count < 5 and current_profit_rate >= 0.02:
        return True, entry_amount * 0.5  # 50% 추가
    return False, 0

# 트레일링 스탑
def check_trailing_stop(positions, symbol, direction, current_price, trailing_distance=0.015):
    if (symbol, direction) not in positions:
        return False
    
    position = positions[(symbol, direction)]
    if 'peak_price' not in position:
        position['peak_price'] = position['entry_price']
    
    # 고점 업데이트
    if current_price > position['peak_price']:
        position['peak_price'] = current_price
    
    # 트레일링 스탑 조건 (3% 이상 수익 시 활성화)
    profit_rate = (current_price - position['entry_price']) / position['entry_price']
    if profit_rate >= 0.03:
        # 고점 대비 1.5% 하락 시 청산
        if current_price < position['peak_price'] * (1 - trailing_distance):
            return True
    
    return False

# 실전형 리스크 관리
def check_risk_limits(current_capital, initial_capital, daily_loss=0, weekly_loss=0, monthly_loss=0):
    total_return = (current_capital - initial_capital) / initial_capital
    
    # 일일 손실 한도: 3%
    if daily_loss < -0.03:
        return False, "일일 손실 한도 초과"
    
    # 주간 손실 한도: 8%
    if weekly_loss < -0.08:
        return False, "주간 손실 한도 초과"
    
    # 월간 손실 한도: 15%
    if monthly_loss < -0.15:
        return False, "월간 손실 한도 초과"
    
    return True, "리스크 한도 내"

def print_summary(result, label):
    """실전형 한글 요약 출력"""
    print(f"[요약] {label} | 최종 자산: {result['final_capital']:,.0f}원 | 총 수익률: {result['total_return']:+.2f}% | 총 수익금: {result['final_capital']-result['initial_capital']:+,.0f}원 | 최대 낙폭: {result['max_drawdown']:+.2f}% | 거래: {result['total_trades']}회 | 승률: {result['win_rate']:.1f}%")

# 전략 한글 변환 맵
STRATEGY_KOR_MAP = {
    'mean_reversion': '역추세',
    'trend_following': '추세추종',
    'momentum_breakout': '모멘텀돌파',
    'short_momentum': '숏모멘텀',
    'btc_short_only': '비트코인숏전략'
}

if __name__ == "__main__":
    import pandas as pd
    # 데이터 로드 (예시)
    df = pd.read_csv('data/market_data/BNB_USDT_1h.csv')
    initial_capital = 10000000
    print("[동적비중 백테스트]")
    result_dynamic = run_ml_backtest(df, initial_capital=initial_capital, use_dynamic_position=True)
    # 결과 출력
    print_summary(result_dynamic, '동적비중') 