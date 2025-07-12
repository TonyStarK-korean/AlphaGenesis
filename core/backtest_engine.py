"""
실제 백테스트 엔진 - 실제 데이터 기반 백테스트 시스템
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import json
import time

from .data_manager import DataManager
from .ml_optimizer import MLOptimizer
from .dynamic_leverage import DynamicLeverageManager

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """백테스트 결과 데이터 클래스"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_leverage: float
    max_leverage: float
    min_leverage: float
    split_trades: Dict[str, Any]
    trade_log: List[Dict[str, Any]]
    created_at: str
    ml_optimized: bool = False
    ml_params: Dict[str, Any] = None

class RealBacktestEngine:
    """실제 백테스트 엔진"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_optimizer = MLOptimizer()
        self.leverage_manager = DynamicLeverageManager()
        self.results = []
        
        # 지원하는 전략 목록
        self.strategies = {
            'triple_combo': {
                'name': '트리플 콤보 전략',
                'description': 'RSI, MACD, 볼린저 밴드 조합',
                'timeframe': '1h',
                'params': {
                    'rsi_period': (10, 20),
                    'rsi_oversold': (25, 35),
                    'rsi_overbought': (65, 75),
                    'macd_fast': (10, 15),
                    'macd_slow': (20, 30),
                    'bb_period': (15, 25)
                }
            },
            'rsi_strategy': {
                'name': 'RSI 전략',
                'description': 'RSI 지표 기반 역추세 전략',
                'timeframe': '15m',
                'params': {
                    'rsi_period': (10, 20),
                    'rsi_oversold': (20, 35),
                    'rsi_overbought': (65, 80)
                }
            },
            'macd_strategy': {
                'name': 'MACD 전략',
                'description': 'MACD 크로스오버 전략',
                'timeframe': '30m',
                'params': {
                    'macd_fast': (8, 15),
                    'macd_slow': (18, 30),
                    'macd_signal': (7, 12)
                }
            },
            'momentum_strategy': {
                'name': '모멘텀 전략',
                'description': '가격 모멘텀 추세 추종',
                'timeframe': '4h',
                'params': {
                    'momentum_period': (15, 25),
                    'threshold': (0.03, 0.08)
                }
            },
            'ml_ensemble': {
                'name': 'ML 앙상블 전략',
                'description': '머신러닝 앙상블 예측',
                'timeframe': '1h',
                'params': {
                    'confidence_threshold': (0.6, 0.8),
                    'ensemble_models': ['XGBoost', 'RandomForest', 'LSTM']
                }
            },
            'simple_triple_combo': {
                'name': '심플 트리플 콤보',
                'description': '간단한 트리플 콤보 전략',
                'timeframe': '1h',
                'params': {
                    'rsi_period': (10, 20),
                    'rsi_oversold': (25, 35),
                    'rsi_overbought': (65, 75),
                    'macd_fast': (10, 15),
                    'macd_slow': (20, 30),
                    'bb_period': (15, 25)
                }
            }
        }
        
        # 주요 USDT 선물 심볼
        self.major_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT',
            'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT',
            'LTC/USDT', 'BCH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT'
        ]
    
    async def run_backtest(
        self,
        config: Dict[str, Any],
        log_callback: Optional[callable] = None
    ) -> BacktestResult:
        """
        실제 백테스트 실행
        
        Args:
            config: 백테스트 설정
            log_callback: 로그 콜백 함수
            
        Returns:
            BacktestResult: 백테스트 결과
        """
        try:
            if log_callback:
                log_callback("🚀 실제 백테스트 시작", "system", 0)
            
            # 설정 추출
            strategy_id = config.get('strategy')
            
            # 전략 존재 확인
            if strategy_id not in self.strategies:
                supported_strategies = ', '.join(self.strategies.keys())
                error_msg = f"지원하지 않는 전략: {strategy_id}. 지원하는 전략: {supported_strategies}"
                logger.error(error_msg)
                if log_callback:
                    log_callback(f"❌ {error_msg}", "error", 0)
                raise ValueError(error_msg)
            
            symbol = config.get('symbol')
            symbol_type = config.get('symbol_type', 'individual')
            start_date = datetime.strptime(config.get('start_date'), '%Y-%m-%d')
            end_date = datetime.strptime(config.get('end_date'), '%Y-%m-%d')
            timeframe = config.get('timeframe', '1h')
            initial_capital = float(config.get('initial_capital', 10000000))
            ml_optimization = config.get('ml_optimization', False)
            
            
            if log_callback:
                log_callback(f"📊 설정 검증 완료", "system", 5)
                log_callback(f"  └─ 전략: {self.strategies[strategy_id]['name']}", "data", 5)
                log_callback(f"  └─ 심볼: {symbol if symbol_type == 'individual' else '전체 시장'}", "data", 6)
                log_callback(f"  └─ 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}", "data", 7)
                log_callback(f"  └─ 초기자본: {initial_capital:,.0f}원", "data", 8)
            
            # 데이터 다운로드
            if symbol_type == 'individual':
                data = await self.download_symbol_data(symbol, timeframe, start_date, end_date, log_callback)
                symbols_to_test = [symbol]
                
                # 데이터가 없는 경우 처리
                if data.empty:
                    if log_callback:
                        log_callback(f"⚠️ {symbol} 데이터 없음, 기본 심볼로 대체", "data", 15)
                    # 기본 심볼로 대체
                    data = await self.download_symbol_data('BTC/USDT', timeframe, start_date, end_date, log_callback)
                    symbols_to_test = ['BTC/USDT']
            else:
                data = await self.download_market_data(timeframe, start_date, end_date, log_callback)
                symbols_to_test = list(data.keys())
                
                # 유효한 데이터가 있는 심볼만 선택
                valid_symbols = [s for s in symbols_to_test if s in data and not data[s].empty]
                if valid_symbols:
                    symbols_to_test = valid_symbols
                else:
                    if log_callback:
                        log_callback(f"⚠️ 유효한 시장 데이터 없음, 기본 심볼 사용", "data", 15)
                    # 기본 심볼 사용
                    data = await self.download_symbol_data('BTC/USDT', timeframe, start_date, end_date, log_callback)
                    symbols_to_test = ['BTC/USDT']
            
            if log_callback:
                log_callback(f"✅ 데이터 다운로드 완료", "data", 20)
            
            # ML 최적화
            if ml_optimization:
                optimized_params = await self.optimize_strategy(strategy_id, data, log_callback)
            else:
                optimized_params = self.get_default_params(strategy_id)
            
            if log_callback:
                log_callback(f"⚙️ 전략 파라미터 최적화 완료", "system", 30)
            
            # 백테스트 실행
            result = await self.execute_backtest(
                strategy_id, symbols_to_test, data, optimized_params, 
                initial_capital, start_date, end_date, log_callback
            )
            
            if log_callback:
                log_callback(f"📊 백테스트 완료!", "system", 100)
                log_callback(f"🏆 최종 수익률: {result.total_return:.2f}%", "result", 100)
                log_callback(f"📈 총 거래 횟수: {result.total_trades}회", "result", 100)
                log_callback(f"🎯 승률: {result.win_rate:.1f}%", "result", 100)
            
            return result
            
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")
            if log_callback:
                log_callback(f"❌ 백테스트 실패: {str(e)}", "error", 0)
            raise e
    
    async def download_symbol_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime,
        log_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """개별 심볼 데이터 다운로드"""
        try:
            if log_callback:
                log_callback(f"📊 {symbol} 데이터 다운로드 중...", "data", 10)
            
            # 실제 데이터 다운로드 시도
            try:
                data = await self.data_manager.download_historical_data(
                    symbol, timeframe, start_date, end_date
                )
            except Exception as download_error:
                if log_callback:
                    log_callback(f"⚠️ 실시간 데이터 실패, 로컬 데이터 사용: {str(download_error)}", "warning", 12)
                
                # 로컬 데이터 시도
                try:
                    data = self.data_manager.load_market_data(symbol, timeframe)
                    if not data.empty:
                        # 날짜 범위 필터링
                        if 'timestamp' in data.columns:
                            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
                        if log_callback:
                            log_callback(f"✅ {symbol} 로컬 데이터 사용 ({len(data)} 캔들)", "data", 14)
                except Exception as local_error:
                    if log_callback:
                        log_callback(f"❌ 로컬 데이터도 실패: {str(local_error)}", "error", 15)
                    data = pd.DataFrame()
            
            if data.empty:
                raise ValueError(f"데이터 다운로드 실패: {symbol}")
            
            # 기술적 지표 추가
            data = self.data_manager.add_technical_indicators(data)
            
            if log_callback:
                log_callback(f"✅ {symbol} 데이터 준비 완료 ({len(data)} 캔들)", "data", 15)
            
            return data
            
        except Exception as e:
            logger.error(f"심볼 데이터 다운로드 실패: {e}")
            if log_callback:
                log_callback(f"❌ {symbol} 데이터 로드 실패: {str(e)}", "error", 0)
            raise e
    
    async def download_market_data(
        self, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime,
        log_callback: Optional[callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """시장 전체 데이터 다운로드 (실제 전체 USDT 선물 포함)"""
        try:
            if log_callback:
                log_callback(f"🔍 시장 전체 심볼 조회 중...", "data", 5)
            
            # 실제 바이낸스 USDT 선물 심볼 조회
            all_symbols = await self.data_manager.get_all_usdt_futures_symbols()
            
            if log_callback:
                log_callback(f"📊 총 {len(all_symbols)}개 심볼 데이터 다운로드 시작...", "data", 10)
            
            # 상위 50개 심볼만 선택 (성능 고려)
            selected_symbols = all_symbols[:50]
            
            # 심볼별 데이터 다운로드
            market_data = {}
            for i, symbol in enumerate(selected_symbols):
                try:
                    progress = 10 + (i / len(selected_symbols)) * 70
                    if log_callback and i % 5 == 0:
                        log_callback(f"  └─ {symbol} 다운로드 중... ({i+1}/{len(selected_symbols)})", "data", progress)
                    
                    data = await self.data_manager.download_historical_data(
                        symbol, timeframe, start_date, end_date, limit=1000
                    )
                    
                    if not data.empty:
                        # 기술적 지표 추가
                        processed_data = self.data_manager.add_technical_indicators(data)
                        market_data[symbol] = processed_data
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"심볼 {symbol} 다운로드 실패: {e}")
                    continue
            
            if log_callback:
                log_callback(f"✅ 시장 데이터 준비 완료 ({len(market_data)} 심볼)", "data", 80)
            
            return market_data
            
        except Exception as e:
            logger.error(f"시장 데이터 다운로드 실패: {e}")
            if log_callback:
                log_callback(f"❌ 시장 데이터 다운로드 실패: {str(e)}", "error", 0)
            raise e
    
    async def optimize_strategy(
        self, 
        strategy_id: str, 
        data: Any, 
        log_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """전략 파라미터 최적화"""
        try:
            if log_callback:
                log_callback(f"🤖 ML 최적화 시작...", "system", 25)
            
            strategy_info = self.strategies[strategy_id]
            param_ranges = strategy_info['params']
            
            # 개별 심볼 vs 시장 전체 처리
            if isinstance(data, pd.DataFrame):
                # 개별 심볼
                train_data = data.iloc[:int(len(data) * 0.8)]  # 80% 훈련용
                result = self.ml_optimizer.optimize_strategy_parameters(
                    train_data, strategy_id, param_ranges, n_trials=50
                )
            else:
                # 시장 전체 - 대표 심볼 사용
                main_symbol = 'BTC/USDT'
                if main_symbol in data:
                    train_data = data[main_symbol].iloc[:int(len(data[main_symbol]) * 0.8)]
                    result = self.ml_optimizer.optimize_strategy_parameters(
                        train_data, strategy_id, param_ranges, n_trials=50
                    )
                else:
                    result = {'best_params': self.get_default_params(strategy_id)}
            
            if log_callback:
                log_callback(f"✅ ML 최적화 완료", "system", 28)
            
            return result.get('best_params', self.get_default_params(strategy_id))
            
        except Exception as e:
            logger.error(f"전략 최적화 실패: {e}")
            return self.get_default_params(strategy_id)
    
    def get_default_params(self, strategy_id: str) -> Dict[str, Any]:
        """전략 기본 파라미터 반환"""
        defaults = {
            'triple_combo': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_fast': 12,
                'macd_slow': 26,
                'bb_period': 20
            },
            'rsi_strategy': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            },
            'macd_strategy': {
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            'momentum_strategy': {
                'momentum_period': 20,
                'threshold': 0.05
            },
            'ml_ensemble': {
                'confidence_threshold': 0.7,
                'ensemble_models': ['XGBoost', 'RandomForest']
            },
            'simple_triple_combo': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_fast': 12,
                'macd_slow': 26,
                'bb_period': 20
            }
        }
        return defaults.get(strategy_id, {})
    
    async def execute_backtest(
        self,
        strategy_id: str,
        symbols: List[str],
        data: Any,
        params: Dict[str, Any],
        initial_capital: float,
        start_date: datetime,
        end_date: datetime,
        log_callback: Optional[callable] = None
    ) -> BacktestResult:
        """백테스트 실행"""
        try:
            if log_callback:
                log_callback(f"🎯 백테스트 실행 시작", "system", 40)
            
            # 포트폴리오 초기화
            portfolio = {
                'capital': initial_capital,
                'positions': {},
                'trade_log': [],
                'equity_curve': [],
                'leverage_history': []
            }
            
            total_trades = 0
            winning_trades = 0
            split_trades = {'total_splits': 0, 'split_success_rate': 0, 'avg_split_count': 0}
            
            # 개별 심볼 vs 시장 전체 처리
            if len(symbols) == 1:
                # 개별 심볼 백테스트
                result = await self.backtest_single_symbol(
                    symbols[0], data, strategy_id, params, portfolio, log_callback
                )
            else:
                # 시장 전체 백테스트
                result = await self.backtest_multiple_symbols(
                    symbols, data, strategy_id, params, portfolio, log_callback
                )
            
            # 결과 계산
            final_value = portfolio['capital']
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            # 통계 계산
            equity_curve = pd.Series(portfolio['equity_curve'])
            returns = equity_curve.pct_change().dropna()
            
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(equity_curve)
            
            total_trades = len(portfolio['trade_log'])
            winning_trades = sum(1 for trade in portfolio['trade_log'] if trade['pnl'] > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 레버리지 통계
            leverage_stats = self.calculate_leverage_stats(portfolio['leverage_history'])
            
            if log_callback:
                log_callback(f"📊 결과 계산 완료", "system", 90)
            
            # 결과 객체 생성
            result = BacktestResult(
                strategy_name=self.strategies[strategy_id]['name'],
                symbol=symbols[0] if len(symbols) == 1 else 'MARKET_WIDE',
                timeframe=self.strategies[strategy_id]['timeframe'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=initial_capital,
                final_value=final_value,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=total_trades - winning_trades,
                avg_leverage=leverage_stats['avg'],
                max_leverage=leverage_stats['max'],
                min_leverage=leverage_stats['min'],
                split_trades=split_trades,
                trade_log=portfolio['trade_log'],
                created_at=datetime.now().isoformat(),
                ml_optimized=True,
                ml_params=params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")
            raise e
    
    async def backtest_single_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        strategy_id: str,
        params: Dict[str, Any],
        portfolio: Dict[str, Any],
        log_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """개별 심볼 백테스트"""
        try:
            if log_callback:
                log_callback(f"📈 {symbol} 백테스트 실행 중...", "analysis", 50)
            
            # 전략별 시그널 생성
            signals = self.generate_signals(data, strategy_id, params)
            
            # 거래 실행
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if i < len(signals):
                    signal = signals[i]
                    
                    # 동적 레버리지 계산
                    try:
                        recent_data = data.iloc[max(0, i-20):i+1]
                        leverage_result = self.leverage_manager.calculate_optimal_leverage(
                            market_data=recent_data,
                            strategy=strategy_id,
                            current_position=portfolio.get('capital', 0),
                            portfolio_value=portfolio.get('capital', 100000),
                            risk_metrics=None
                        )
                        leverage = leverage_result.get('optimal_leverage', 1.0) if isinstance(leverage_result, dict) else 1.0
                    except Exception as e:
                        logger.error(f"레버리지 계산 실패: {e}")
                        leverage = 1.0  # 기본값
                    
                    portfolio['leverage_history'].append(leverage)
                    
                    # 매매 실행
                    if signal == 1:  # 매수
                        await self.execute_trade(
                            'BUY', symbol, row, portfolio, leverage, log_callback
                        )
                    elif signal == -1:  # 매도
                        await self.execute_trade(
                            'SELL', symbol, row, portfolio, leverage, log_callback
                        )
                
                # 포트폴리오 가치 업데이트
                portfolio['equity_curve'].append(self.calculate_portfolio_value(portfolio, row))
                
                # 진행률 업데이트
                progress = 50 + (i / len(data)) * 40
                if log_callback and i % 100 == 0:
                    log_callback(f"  └─ 진행률: {progress:.1f}%", "analysis", progress)
            
            if log_callback:
                log_callback(f"✅ {symbol} 백테스트 완료", "analysis", 90)
            
            return {'status': 'success'}
            
        except Exception as e:
            logger.error(f"개별 심볼 백테스트 실패: {e}")
            raise e
    
    async def backtest_multiple_symbols(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        strategy_id: str,
        params: Dict[str, Any],
        portfolio: Dict[str, Any],
        log_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """시장 전체 백테스트 (매매 기회 스캔 포함)"""
        try:
            if log_callback:
                log_callback(f"🔍 시장 전체 매매 기회 스캔 시작...", "analysis", 30)
            
            # 시장 전체 스캔으로 매매 기회 탐색
            opportunities = await self.data_manager.scan_market_opportunities(
                strategy_id, 
                timeframe=self.strategies[strategy_id]['timeframe'],
                top_n=len(symbols),
                log_callback=log_callback
            )
            
            if log_callback:
                log_callback(f"📈 발견된 기회: {len(opportunities)}개", "analysis", 40)
            
            # 발견된 기회를 기반으로 백테스트 실행
            executed_trades = 0
            for i, opportunity in enumerate(opportunities):
                symbol = opportunity['symbol']
                
                if symbol in data and not data[symbol].empty:
                    if log_callback:
                        log_callback(f"  └─ {symbol} 매매 실행 중... (점수: {opportunity['score']:.2f})", "analysis", 50 + (i/len(opportunities)) * 30)
                    
                    # 기회 점수가 높은 심볼에 대해 백테스트 실행
                    if opportunity['score'] > 0.7:
                        await self.backtest_single_symbol(
                            symbol, data[symbol], strategy_id, params, portfolio, None
                        )
                        executed_trades += 1
                        
                        # 매매 기회 로그
                        if log_callback:
                            log_callback(f"    💰 {symbol} {opportunity['signal']} 신호 실행", opportunity['signal'].lower(), None)
            
            # 나머지 심볼들도 백테스트 (기회가 없더라도)
            remaining_symbols = [s for s in symbols if s not in [opp['symbol'] for opp in opportunities]]
            for i, symbol in enumerate(remaining_symbols):
                if symbol in data and not data[symbol].empty:
                    if log_callback and i % 10 == 0:
                        log_callback(f"  └─ {symbol} 일반 분석 중...", "analysis", 80 + (i/len(remaining_symbols)) * 10)
                    
                    await self.backtest_single_symbol(
                        symbol, data[symbol], strategy_id, params, portfolio, None
                    )
            
            if log_callback:
                log_callback(f"✅ 시장 전체 백테스트 완료 (매매 기회: {executed_trades}개)", "analysis", 90)
            
            return {
                'status': 'success',
                'opportunities_found': len(opportunities),
                'trades_executed': executed_trades,
                'opportunities': opportunities
            }
            
        except Exception as e:
            logger.error(f"시장 전체 백테스트 실패: {e}")
            raise e
    
    def generate_signals(self, data: pd.DataFrame, strategy_id: str, params: Dict[str, Any]) -> List[int]:
        """전략별 매매 시그널 생성"""
        try:
            signals = []
            
            if strategy_id == 'triple_combo':
                # 트리플 콤보 전략
                rsi_oversold = params.get('rsi_oversold', 30)
                rsi_overbought = params.get('rsi_overbought', 70)
                
                for i in range(len(data)):
                    if i < 50:  # 충분한 데이터가 있을 때까지 대기
                        signals.append(0)
                        continue
                    
                    rsi = data['RSI'].iloc[i]
                    macd = data['MACD'].iloc[i]
                    macd_signal = data['MACD_Signal'].iloc[i]
                    bb_upper = data['BB_Upper'].iloc[i]
                    bb_lower = data['BB_Lower'].iloc[i]
                    close = data['close'].iloc[i]
                    
                    # 매수 신호 - 조건 완화 (3개 중 2개 충족 시 매수)
                    buy_conditions = 0
                    if rsi < rsi_oversold:
                        buy_conditions += 1
                    if macd > macd_signal:
                        buy_conditions += 1
                    if close <= bb_lower * 1.01:  # 볼린저 밴드 하단 1% 여유
                        buy_conditions += 1
                    
                    # 매도 신호 - 조건 완화 (3개 중 2개 충족 시 매도)
                    sell_conditions = 0
                    if rsi > rsi_overbought:
                        sell_conditions += 1
                    if macd < macd_signal:
                        sell_conditions += 1
                    if close >= bb_upper * 0.99:  # 볼린저 밴드 상단 1% 여유
                        sell_conditions += 1
                    
                    if buy_conditions >= 2:
                        signals.append(1)
                    elif sell_conditions >= 2:
                        signals.append(-1)
                    else:
                        signals.append(0)
            
            elif strategy_id == 'simple_triple_combo':
                # 심플 트리플 콤보 전략 (triple_combo와 같은 로직)
                rsi_oversold = params.get('rsi_oversold', 30)
                rsi_overbought = params.get('rsi_overbought', 70)
                
                for i in range(len(data)):
                    if i < 50:  # 충분한 데이터가 있을 때까지 대기
                        signals.append(0)
                        continue
                    
                    rsi = data['RSI'].iloc[i]
                    macd = data['MACD'].iloc[i]
                    macd_signal = data['MACD_Signal'].iloc[i]
                    bb_upper = data['BB_Upper'].iloc[i]
                    bb_lower = data['BB_Lower'].iloc[i]
                    close = data['close'].iloc[i]
                    
                    # 매수 신호 - 조건 완화 (3개 중 2개 충족 시 매수)
                    buy_conditions = 0
                    if rsi < rsi_oversold:
                        buy_conditions += 1
                    if macd > macd_signal:
                        buy_conditions += 1
                    if close <= bb_lower * 1.01:  # 볼린저 밴드 하단 1% 여유
                        buy_conditions += 1
                    
                    # 매도 신호 - 조건 완화 (3개 중 2개 충족 시 매도)
                    sell_conditions = 0
                    if rsi > rsi_overbought:
                        sell_conditions += 1
                    if macd < macd_signal:
                        sell_conditions += 1
                    if close >= bb_upper * 0.99:  # 볼린저 밴드 상단 1% 여유
                        sell_conditions += 1
                    
                    if buy_conditions >= 2:
                        signals.append(1)
                    elif sell_conditions >= 2:
                        signals.append(-1)
                    else:
                        signals.append(0)
            
            elif strategy_id == 'rsi_strategy':
                # RSI 전략
                rsi_oversold = params.get('rsi_oversold', 30)
                rsi_overbought = params.get('rsi_overbought', 70)
                
                for i in range(len(data)):
                    if i < 20:
                        signals.append(0)
                        continue
                    
                    rsi = data['RSI'].iloc[i]
                    
                    if rsi < rsi_oversold:
                        signals.append(1)
                    elif rsi > rsi_overbought:
                        signals.append(-1)
                    else:
                        signals.append(0)
            
            # 다른 전략들도 추가...
            else:
                # 기본 전략
                signals = [0] * len(data)
            
            return signals
            
        except Exception as e:
            logger.error(f"시그널 생성 실패: {e}")
            return [0] * len(data)
    
    async def execute_trade(
        self,
        action: str,
        symbol: str,
        market_data: pd.Series,
        portfolio: Dict[str, Any],
        leverage: float,
        log_callback: Optional[callable] = None
    ):
        """거래 실행"""
        try:
            price = market_data['close']
            timestamp = market_data.name
            
            # 포지션 관리
            if action == 'BUY':
                if symbol not in portfolio['positions']:
                    # 새 포지션 생성
                    position_size = portfolio['capital'] * 0.02  # 2% 리스크
                    
                    portfolio['positions'][symbol] = {
                        'size': position_size / price * leverage,
                        'entry_price': price,
                        'leverage': leverage,
                        'timestamp': timestamp
                    }
                    
                    # 거래 로그
                    portfolio['trade_log'].append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': price,
                        'size': position_size / price * leverage,
                        'leverage': leverage,
                        'pnl': 0
                    })
                    
                    if log_callback:
                        log_callback(f"💰 [진입] {symbol} 매수 실행", "buy", None)
                        log_callback(f"  └─ 가격: ${price:.2f} | 레버리지: {leverage:.1f}x", "buy", None)
            
            elif action == 'SELL':
                if symbol in portfolio['positions']:
                    position = portfolio['positions'][symbol]
                    
                    # 수익 계산
                    pnl = (price - position['entry_price']) * position['size']
                    portfolio['capital'] += pnl
                    
                    # 거래 로그
                    portfolio['trade_log'].append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': price,
                        'size': position['size'],
                        'leverage': position['leverage'],
                        'pnl': pnl
                    })
                    
                    if log_callback:
                        log_callback(f"🎯 [청산] {symbol} 매도 실행", "sell", None)
                        log_callback(f"  └─ 수익: ${pnl:.2f} ({pnl/position['entry_price']*100:.2f}%)", "sell", None)
                    
                    # 포지션 제거
                    del portfolio['positions'][symbol]
            
        except Exception as e:
            logger.error(f"거래 실행 실패: {e}")
    
    def calculate_portfolio_value(self, portfolio: Dict[str, Any], market_data: pd.Series) -> float:
        """포트폴리오 가치 계산"""
        try:
            total_value = portfolio['capital']
            
            for symbol, position in portfolio['positions'].items():
                if symbol in market_data:
                    current_price = market_data['close']
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    total_value += unrealized_pnl
            
            return total_value
            
        except Exception as e:
            logger.error(f"포트폴리오 가치 계산 실패: {e}")
            return portfolio['capital']
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """샤프 비율 계산"""
        try:
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            return annual_return / annual_volatility
            
        except Exception as e:
            logger.error(f"샤프 비율 계산 실패: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """최대 낙폭 계산"""
        try:
            if len(equity_curve) == 0:
                return 0.0
            
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100
            
            return abs(drawdown.min())
            
        except Exception as e:
            logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0
    
    def calculate_leverage_stats(self, leverage_history: List[float]) -> Dict[str, float]:
        """레버리지 통계 계산"""
        try:
            if not leverage_history:
                return {'avg': 1.0, 'max': 1.0, 'min': 1.0}
            
            # 레버리지 값들을 숫자로 변환
            numeric_leverages = []
            for lev in leverage_history:
                if isinstance(lev, (int, float)):
                    numeric_leverages.append(float(lev))
                elif isinstance(lev, dict):
                    # 딕셔너리인 경우 optimal_leverage 값 사용
                    numeric_leverages.append(float(lev.get('optimal_leverage', 1.0)))
                else:
                    numeric_leverages.append(1.0)
            
            if not numeric_leverages:
                return {'avg': 1.0, 'max': 1.0, 'min': 1.0}
            
            return {
                'avg': float(np.mean(numeric_leverages)),
                'max': float(max(numeric_leverages)),
                'min': float(min(numeric_leverages))
            }
            
        except Exception as e:
            logger.error(f"레버리지 통계 계산 실패: {e}")
            return {'avg': 1.0, 'max': 1.0, 'min': 1.0}