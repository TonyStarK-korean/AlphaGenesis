#!/usr/bin/env python3
"""
파라미터 최적화 시스템
ML 기반 전략 파라미터 자동 최적화
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib
from pathlib import Path

# 프로젝트 모듈
from ml.models.price_prediction_model import PricePredictionModel
from exchange.binance_futures_api import BinanceFuturesAPI
from config.unified_config import config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """최적화 결과 데이터 클래스"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict]
    evaluation_metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ParameterOptimizer:
    """파라미터 최적화 클래스"""
    
    def __init__(self, config_path: str = None):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        self.optimization_dir = self.project_root / "ml" / "optimization"
        self.results_dir = self.optimization_dir / "results"
        
        # 디렉토리 생성
        self.optimization_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 최적화 설정
        self.optimization_config = {
            'n_trials': 100,
            'n_jobs': 4,
            'timeout': 3600,  # 1시간
            'sampler': 'TPE',
            'pruner': 'MedianPruner',
            'direction': 'maximize',
            'metric': 'sharpe_ratio'
        }
        
        # 백테스트 설정
        self.backtest_config = {
            'initial_capital': 1000000,
            'commission': 0.001,
            'slippage': 0.0001,
            'lookback_days': 90,
            'validation_days': 30
        }
        
        # 데이터 캐시
        self.data_cache = {}
        
        # 실행기
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("파라미터 최적화 시스템 초기화 완료")
    
    def define_parameter_space(self, strategy_type: str = 'triple_combo') -> Dict[str, Any]:
        """
        최적화할 파라미터 공간 정의
        
        Args:
            strategy_type: 전략 타입
            
        Returns:
            파라미터 공간 정의
        """
        try:
            if strategy_type == 'triple_combo':
                return {
                    # ML 모델 파라미터
                    'ml_lookback_window': (50, 200),
                    'ml_prediction_horizon': (1, 24),
                    'ml_feature_importance_threshold': (0.01, 0.1),
                    
                    # 전략 파라미터
                    'confidence_threshold': (0.3, 0.8),
                    'trend_threshold': (0.001, 0.02),
                    'volatility_threshold': (0.01, 0.1),
                    'volume_threshold': (0.8, 2.0),
                    
                    # 위험 관리 파라미터
                    'position_size_ratio': (0.05, 0.3),
                    'stop_loss_ratio': (0.01, 0.05),
                    'take_profit_ratio': (0.02, 0.1),
                    'max_drawdown_threshold': (0.1, 0.3),
                    
                    # 기술적 지표 파라미터
                    'sma_short_period': (5, 20),
                    'sma_long_period': (20, 100),
                    'rsi_period': (10, 30),
                    'rsi_overbought': (70, 90),
                    'rsi_oversold': (10, 30),
                    'macd_fast': (8, 16),
                    'macd_slow': (20, 30),
                    'bollinger_period': (15, 30),
                    'bollinger_std': (1.5, 2.5),
                    
                    # 시장 조건 파라미터
                    'trend_detection_sensitivity': (0.1, 1.0),
                    'volatility_adjustment_factor': (0.5, 2.0),
                    'market_regime_threshold': (0.05, 0.2)
                }
            
            elif strategy_type == 'scalping':
                return {
                    # 스캘핑 전용 파라미터
                    'scalp_confidence_threshold': (0.5, 0.9),
                    'scalp_profit_target': (0.005, 0.02),
                    'scalp_stop_loss': (0.003, 0.01),
                    'scalp_time_limit': (5, 60),  # 분
                    'scalp_volume_filter': (1.2, 3.0),
                    
                    # 기본 파라미터
                    'ml_lookback_window': (20, 100),
                    'position_size_ratio': (0.1, 0.5),
                    'rsi_period': (5, 15),
                    'bollinger_period': (10, 20)
                }
            
            elif strategy_type == 'trend_following':
                return {
                    # 트렌드 추종 전용 파라미터
                    'trend_confirmation_period': (10, 50),
                    'trend_strength_threshold': (0.02, 0.1),
                    'trend_reversal_threshold': (0.01, 0.05),
                    'trend_momentum_factor': (0.5, 2.0),
                    
                    # 기본 파라미터
                    'ml_lookback_window': (100, 300),
                    'confidence_threshold': (0.4, 0.7),
                    'position_size_ratio': (0.05, 0.2),
                    'stop_loss_ratio': (0.02, 0.08),
                    'take_profit_ratio': (0.05, 0.15)
                }
            
            else:
                raise ValueError(f"지원되지 않는 전략 타입: {strategy_type}")
                
        except Exception as e:
            logger.error(f"파라미터 공간 정의 실패: {e}")
            return {}
    
    def create_objective_function(self, symbols: List[str], strategy_type: str) -> callable:
        """
        Optuna 목적 함수 생성
        
        Args:
            symbols: 최적화할 심볼 리스트
            strategy_type: 전략 타입
            
        Returns:
            목적 함수
        """
        def objective(trial):
            try:
                # 파라미터 공간에서 파라미터 샘플링
                param_space = self.define_parameter_space(strategy_type)
                params = {}
                
                for param_name, param_range in param_space.items():
                    if isinstance(param_range, tuple):
                        if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                
                # 백테스트 실행
                backtest_results = self.run_backtest_with_params(symbols, params, strategy_type)
                
                # 목적 함수 값 계산
                metric = self.optimization_config['metric']
                score = backtest_results.get(metric, 0.0)
                
                # 추가 제약 조건 확인
                if self.check_constraints(backtest_results):
                    return score
                else:
                    return -999.0  # 제약 조건 위반 시 낮은 점수
                    
            except Exception as e:
                logger.error(f"목적 함수 실행 실패: {e}")
                return -999.0
        
        return objective
    
    def run_backtest_with_params(self, symbols: List[str], params: Dict[str, Any], 
                                strategy_type: str) -> Dict[str, Any]:
        """
        파라미터를 적용한 백테스트 실행
        
        Args:
            symbols: 심볼 리스트
            params: 파라미터
            strategy_type: 전략 타입
            
        Returns:
            백테스트 결과
        """
        try:
            # 데이터 준비
            data_dict = self.prepare_backtest_data(symbols)
            
            # 전략 실행
            if strategy_type == 'triple_combo':
                results = self.run_triple_combo_backtest(data_dict, params)
            elif strategy_type == 'scalping':
                results = self.run_scalping_backtest(data_dict, params)
            elif strategy_type == 'trend_following':
                results = self.run_trend_following_backtest(data_dict, params)
            else:
                raise ValueError(f"지원되지 않는 전략 타입: {strategy_type}")
            
            # 성능 지표 계산
            performance_metrics = self.calculate_performance_metrics(results)
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")
            return {}
    
    def run_triple_combo_backtest(self, data_dict: Dict[str, pd.DataFrame], 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """
        트리플 콤보 전략 백테스트 실행
        
        Args:
            data_dict: 데이터 딕셔너리
            params: 파라미터
            
        Returns:
            백테스트 결과
        """
        try:
            # 전략 설정
            initial_capital = self.backtest_config['initial_capital']
            capital = initial_capital
            positions = {}
            trades = []
            
            # 각 심볼에 대해 백테스트 실행
            for symbol, df in data_dict.items():
                if df.empty:
                    continue
                
                # 기술적 지표 계산 (파라미터 적용)
                df = self.calculate_technical_indicators_with_params(df, params)
                
                # ML 예측 (파라미터 적용)
                predictions = self.generate_ml_predictions_with_params(df, params)
                
                # 거래 신호 생성
                signals = self.generate_triple_combo_signals(df, predictions, params)
                
                # 거래 실행 시뮬레이션
                symbol_results = self.simulate_trades(df, signals, params)
                trades.extend(symbol_results['trades'])
                
                # 자본 업데이트
                capital += symbol_results['total_pnl']
            
            return {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': (capital - initial_capital) / initial_capital,
                'trades': trades,
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t['pnl'] > 0]),
                'losing_trades': len([t for t in trades if t['pnl'] <= 0])
            }
            
        except Exception as e:
            logger.error(f"트리플 콤보 백테스트 실행 실패: {e}")
            return {}
    
    def calculate_technical_indicators_with_params(self, df: pd.DataFrame, 
                                                  params: Dict[str, Any]) -> pd.DataFrame:
        """
        파라미터를 적용한 기술적 지표 계산
        
        Args:
            df: 가격 데이터
            params: 파라미터
            
        Returns:
            기술적 지표가 추가된 데이터프레임
        """
        try:
            # 이동평균
            sma_short = params.get('sma_short_period', 10)
            sma_long = params.get('sma_long_period', 50)
            df[f'sma_{sma_short}'] = df['close'].rolling(window=sma_short).mean()
            df[f'sma_{sma_long}'] = df['close'].rolling(window=sma_long).mean()
            
            # RSI
            rsi_period = params.get('rsi_period', 14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            ema_fast = df['close'].ewm(span=macd_fast).mean()
            ema_slow = df['close'].ewm(span=macd_slow).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # 볼린저 밴드
            bb_period = params.get('bollinger_period', 20)
            bb_std = params.get('bollinger_std', 2.0)
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            
            # 변동성
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # 거래량 지표
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}")
            return df
    
    def generate_ml_predictions_with_params(self, df: pd.DataFrame, 
                                           params: Dict[str, Any]) -> np.ndarray:
        """
        파라미터를 적용한 ML 예측 생성
        
        Args:
            df: 가격 데이터
            params: 파라미터
            
        Returns:
            예측 배열
        """
        try:
            # 간단한 ML 예측 (실제로는 더 복잡한 모델 사용)
            lookback_window = params.get('ml_lookback_window', 100)
            prediction_horizon = params.get('ml_prediction_horizon', 1)
            
            predictions = []
            
            for i in range(lookback_window, len(df)):
                # 과거 데이터 추출
                historical_data = df.iloc[i-lookback_window:i]
                
                # 간단한 모멘텀 기반 예측
                price_momentum = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0]
                volume_momentum = historical_data['volume_ratio'].iloc[-5:].mean()
                volatility = historical_data['volatility'].iloc[-1]
                
                # 예측 계산
                prediction = price_momentum * 0.4 + (volume_momentum - 1) * 0.3 + volatility * 0.3
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"ML 예측 생성 실패: {e}")
            return np.array([])
    
    def generate_triple_combo_signals(self, df: pd.DataFrame, predictions: np.ndarray,
                                     params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        트리플 콤보 신호 생성
        
        Args:
            df: 가격 데이터
            predictions: ML 예측
            params: 파라미터
            
        Returns:
            거래 신호 리스트
        """
        try:
            signals = []
            confidence_threshold = params.get('confidence_threshold', 0.6)
            trend_threshold = params.get('trend_threshold', 0.01)
            volume_threshold = params.get('volume_threshold', 1.2)
            
            # 예측 데이터와 인덱스 맞추기
            start_idx = len(df) - len(predictions)
            
            for i, prediction in enumerate(predictions):
                idx = start_idx + i
                
                if idx >= len(df):
                    break
                
                row = df.iloc[idx]
                
                # 신호 생성 조건
                confidence = 0.0
                action = 'HOLD'
                
                # 상승 신호 조건
                if (prediction > trend_threshold and 
                    row['rsi'] < 70 and 
                    row['volume_ratio'] > volume_threshold and
                    row['macd'] > row['macd_signal']):
                    confidence = 0.7
                    action = 'BUY'
                
                # 하락 신호 조건
                elif (prediction < -trend_threshold and 
                      row['rsi'] > 30 and 
                      row['volume_ratio'] > volume_threshold and
                      row['macd'] < row['macd_signal']):
                    confidence = 0.7
                    action = 'SELL'
                
                # 신뢰도 임계값 확인
                if confidence >= confidence_threshold:
                    signals.append({
                        'timestamp': row.name,
                        'action': action,
                        'confidence': confidence,
                        'price': row['close'],
                        'prediction': prediction
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"거래 신호 생성 실패: {e}")
            return []
    
    def simulate_trades(self, df: pd.DataFrame, signals: List[Dict[str, Any]],
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """
        거래 시뮬레이션
        
        Args:
            df: 가격 데이터
            signals: 거래 신호 리스트
            params: 파라미터
            
        Returns:
            거래 결과
        """
        try:
            position_size_ratio = params.get('position_size_ratio', 0.1)
            stop_loss_ratio = params.get('stop_loss_ratio', 0.02)
            take_profit_ratio = params.get('take_profit_ratio', 0.05)
            commission = self.backtest_config['commission']
            
            trades = []
            current_position = None
            
            for signal in signals:
                signal_time = signal['timestamp']
                signal_price = signal['price']
                
                # 기존 포지션 있는 경우 청산
                if current_position:
                    # 청산 조건 확인 (스탑로스, 테이크프로핏, 반대 신호)
                    should_close = False
                    close_reason = ""
                    
                    if current_position['side'] == 'BUY':
                        if signal_price <= current_position['stop_loss']:
                            should_close = True
                            close_reason = "stop_loss"
                        elif signal_price >= current_position['take_profit']:
                            should_close = True
                            close_reason = "take_profit"
                        elif signal['action'] == 'SELL':
                            should_close = True
                            close_reason = "reverse_signal"
                    else:  # SHORT
                        if signal_price >= current_position['stop_loss']:
                            should_close = True
                            close_reason = "stop_loss"
                        elif signal_price <= current_position['take_profit']:
                            should_close = True
                            close_reason = "take_profit"
                        elif signal['action'] == 'BUY':
                            should_close = True
                            close_reason = "reverse_signal"
                    
                    if should_close:
                        # 포지션 청산
                        if current_position['side'] == 'BUY':
                            pnl = (signal_price - current_position['entry_price']) * current_position['quantity']
                        else:
                            pnl = (current_position['entry_price'] - signal_price) * current_position['quantity']
                        
                        pnl -= commission * current_position['quantity'] * signal_price  # 청산 수수료
                        
                        trades.append({
                            'entry_time': current_position['entry_time'],
                            'exit_time': signal_time,
                            'side': current_position['side'],
                            'entry_price': current_position['entry_price'],
                            'exit_price': signal_price,
                            'quantity': current_position['quantity'],
                            'pnl': pnl,
                            'close_reason': close_reason
                        })
                        
                        current_position = None
                
                # 새로운 포지션 진입
                if signal['action'] in ['BUY', 'SELL'] and current_position is None:
                    quantity = position_size_ratio  # 비율로 계산
                    entry_fee = commission * quantity * signal_price
                    
                    if signal['action'] == 'BUY':
                        stop_loss = signal_price * (1 - stop_loss_ratio)
                        take_profit = signal_price * (1 + take_profit_ratio)
                    else:
                        stop_loss = signal_price * (1 + stop_loss_ratio)
                        take_profit = signal_price * (1 - take_profit_ratio)
                    
                    current_position = {
                        'side': signal['action'],
                        'entry_time': signal_time,
                        'entry_price': signal_price,
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_fee': entry_fee
                    }
            
            # 결과 계산
            total_pnl = sum(trade['pnl'] for trade in trades)
            
            return {
                'trades': trades,
                'total_pnl': total_pnl,
                'trade_count': len(trades)
            }
            
        except Exception as e:
            logger.error(f"거래 시뮬레이션 실패: {e}")
            return {'trades': [], 'total_pnl': 0.0, 'trade_count': 0}
    
    def calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        성능 지표 계산
        
        Args:
            results: 백테스트 결과
            
        Returns:
            성능 지표
        """
        try:
            trades = results.get('trades', [])
            total_return = results.get('total_return', 0.0)
            
            if not trades:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0
                }
            
            # 기본 지표
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0.0
            
            # 수익률 계산
            returns = [t['pnl'] / self.backtest_config['initial_capital'] for t in trades]
            
            # 샤프 비율
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # 최대 낙폭
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
            
            # 수익 인수
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'average_win': gross_profit / len(winning_trades) if winning_trades else 0.0,
                'average_loss': gross_loss / len(losing_trades) if losing_trades else 0.0
            }
            
        except Exception as e:
            logger.error(f"성능 지표 계산 실패: {e}")
            return {}
    
    def check_constraints(self, results: Dict[str, Any]) -> bool:
        """
        제약 조건 확인
        
        Args:
            results: 백테스트 결과
            
        Returns:
            제약 조건 만족 여부
        """
        try:
            # 최소 거래 수
            if results.get('total_trades', 0) < 10:
                return False
            
            # 최대 낙폭 제한
            if results.get('max_drawdown', 0) > 0.3:
                return False
            
            # 승률 제한
            if results.get('win_rate', 0) < 0.3:
                return False
            
            # 수익률 제한
            if results.get('total_return', 0) < -0.5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"제약 조건 확인 실패: {e}")
            return False
    
    def prepare_backtest_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        백테스트 데이터 준비
        
        Args:
            symbols: 심볼 리스트
            
        Returns:
            심볼별 데이터 딕셔너리
        """
        try:
            # 캐시에서 데이터 확인
            cache_key = '_'.join(sorted(symbols))
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # 데이터 로드
            data_dict = {}
            api = BinanceFuturesAPI()
            
            for symbol in symbols:
                try:
                    # 더미 데이터 생성 (실제로는 API에서 데이터 조회)
                    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
                    np.random.seed(42)
                    
                    # 비트코인 시뮬레이션
                    base_price = 50000 if 'BTC' in symbol else 3000
                    prices = [base_price]
                    
                    for i in range(1, len(dates)):
                        change = np.random.normal(0, 0.02)
                        new_price = prices[-1] * (1 + change)
                        prices.append(new_price)
                    
                    df = pd.DataFrame({
                        'timestamp': dates,
                        'open': prices,
                        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                        'close': prices,
                        'volume': np.random.uniform(100, 1000, len(dates))
                    })
                    
                    df.set_index('timestamp', inplace=True)
                    data_dict[symbol] = df
                    
                except Exception as e:
                    logger.error(f"{symbol} 데이터 로드 실패: {e}")
            
            # 캐시에 저장
            self.data_cache[cache_key] = data_dict
            
            return data_dict
            
        except Exception as e:
            logger.error(f"백테스트 데이터 준비 실패: {e}")
            return {}
    
    async def optimize_parameters(self, symbols: List[str], strategy_type: str = 'triple_combo',
                                 n_trials: int = 100) -> OptimizationResult:
        """
        파라미터 최적화 실행
        
        Args:
            symbols: 심볼 리스트
            strategy_type: 전략 타입
            n_trials: 시도 횟수
            
        Returns:
            최적화 결과
        """
        try:
            logger.info(f"파라미터 최적화 시작: {strategy_type}")
            
            # Optuna 스터디 생성
            study = optuna.create_study(
                direction=self.optimization_config['direction'],
                sampler=TPESampler(n_startup_trials=10),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            
            # 목적 함수 생성
            objective = self.create_objective_function(symbols, strategy_type)
            
            # 최적화 실행
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=self.optimization_config['timeout'],
                n_jobs=1  # 멀티프로세싱 비활성화 (안정성)
            )
            
            # 최적 파라미터로 최종 백테스트 실행
            best_params = study.best_params
            backtest_results = self.run_backtest_with_params(symbols, best_params, strategy_type)
            
            # 결과 생성
            result = OptimizationResult(
                best_params=best_params,
                best_score=study.best_value,
                optimization_history=[
                    {
                        'trial': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'state': trial.state.name
                    }
                    for trial in study.trials
                ],
                evaluation_metrics=backtest_results,
                backtest_results=backtest_results
            )
            
            # 결과 저장
            await self.save_optimization_result(result, symbols, strategy_type)
            
            logger.info(f"파라미터 최적화 완료: 최고 점수 {study.best_value:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"파라미터 최적화 실패: {e}")
            raise
    
    async def save_optimization_result(self, result: OptimizationResult, 
                                      symbols: List[str], strategy_type: str):
        """
        최적화 결과 저장
        
        Args:
            result: 최적화 결과
            symbols: 심볼 리스트
            strategy_type: 전략 타입
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{strategy_type}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # 결과 데이터 준비
            result_data = {
                'symbols': symbols,
                'strategy_type': strategy_type,
                'timestamp': result.timestamp.isoformat(),
                'best_params': result.best_params,
                'best_score': result.best_score,
                'optimization_history': result.optimization_history,
                'evaluation_metrics': result.evaluation_metrics,
                'backtest_results': result.backtest_results
            }
            
            # JSON 파일로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"최적화 결과 저장: {filepath}")
            
        except Exception as e:
            logger.error(f"최적화 결과 저장 실패: {e}")
    
    async def load_optimization_result(self, filename: str) -> OptimizationResult:
        """
        최적화 결과 로드
        
        Args:
            filename: 파일명
            
        Returns:
            최적화 결과
        """
        try:
            filepath = self.results_dir / filename
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = OptimizationResult(
                best_params=data['best_params'],
                best_score=data['best_score'],
                optimization_history=data['optimization_history'],
                evaluation_metrics=data['evaluation_metrics'],
                backtest_results=data['backtest_results'],
                timestamp=datetime.fromisoformat(data['timestamp'])
            )
            
            return result
            
        except Exception as e:
            logger.error(f"최적화 결과 로드 실패: {e}")
            raise

# 테스트 코드
async def test_parameter_optimization():
    """파라미터 최적화 테스트"""
    try:
        print("🚀 파라미터 최적화 테스트 시작")
        
        # 최적화 객체 생성
        optimizer = ParameterOptimizer()
        
        # 테스트 심볼
        test_symbols = ['BTC/USDT', 'ETH/USDT']
        
        # 최적화 실행
        result = await optimizer.optimize_parameters(test_symbols, 'triple_combo', n_trials=20)
        
        print(f"✅ 최적화 완료")
        print(f"최고 점수: {result.best_score:.4f}")
        print(f"최적 파라미터: {result.best_params}")
        
        print("🎉 파라미터 최적화 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    # 비동기 테스트 실행
    asyncio.run(test_parameter_optimization())