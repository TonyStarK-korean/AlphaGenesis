import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import time
from typing import Dict, List, Optional, Callable

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TripleComboStrategy 선택적 임포트
try:
    from triple_combo_strategy import TripleComboStrategy
    TRIPLE_COMBO_AVAILABLE = True
except ImportError:
    print("Warning: TripleComboStrategy not available. Using mock class.")
    TRIPLE_COMBO_AVAILABLE = False
    class TripleComboStrategy(bt.Strategy):
        params = ()
        def __init__(self):
            pass
        def next(self):
            pass

class SmaCross(bt.Strategy):
    """이동평균 교차 전략"""
    params = (('pfast', 10), ('pslow', 30),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.pfast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.pslow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        elif self.crossover < 0:
            self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

class RsiStrategy(bt.Strategy):
    """RSI 전략"""
    params = (('period', 14), ('upper', 70), ('lower', 30),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.rsi < self.p.lower:
                self.order = self.buy()
        else:
            if self.rsi > self.p.upper:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

class BollingerBandsStrategy(bt.Strategy):
    """볼린저 밴드 전략"""
    params = (('period', 20), ('devfactor', 2),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.devfactor)

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.dataclose < self.bb.lines.bot:
                self.order = self.buy()
        else:
            if self.dataclose > self.bb.lines.top:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

class MacdStrategy(bt.Strategy):
    """MACD 전략"""
    params = (('fast_period', 12), ('slow_period', 26), ('signal_period', 9),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.macd = bt.indicators.MACD(
            self.data.close, 
            period_me1=self.p.fast_period, 
            period_me2=self.p.slow_period, 
            period_signal=self.p.signal_period
        )

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.macd.macd > self.macd.signal:
                self.order = self.buy()
        else:
            if self.macd.macd < self.macd.signal:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

# 전략 레지스트리
STRATEGY_REGISTRY = {
    'SmaCross': SmaCross,
    'RSI': RsiStrategy,
    'BollingerBands': BollingerBandsStrategy,
    'MACD': MacdStrategy,
}

if TRIPLE_COMBO_AVAILABLE:
    STRATEGY_REGISTRY['TripleCombo'] = TripleComboStrategy

# 자산 변화 곡선을 위한 커스텀 분석기
class EquityCurveAnalyzer(bt.Analyzer):
    def __init__(self):
        self.equity_curve = []
        self.dates = []
        
    def next(self):
        self.equity_curve.append(self.strategy.broker.getvalue())
        self.dates.append(self.strategy.data.datetime.date())
        
    def get_analysis(self):
        return {
            'equity_curve': self.equity_curve,
            'dates': self.dates
        }

# 매매 로그 분석기
class TradeLogAnalyzer(bt.Analyzer):
    def __init__(self):
        self.trades = []
        
    def notify_order(self, order):
        if order.status in [order.Completed]:
            trade_info = {
                'date': self.strategy.data.datetime.date().isoformat(),
                'time': self.strategy.data.datetime.time().isoformat(),
                'symbol': order.data._name,
                'action': 'BUY' if order.isbuy() else 'SELL',
                'size': order.size,
                'price': order.executed.price,
                'value': order.executed.value,
                'commission': order.executed.comm,
                'pnl': 0  # 나중에 계산
            }
            self.trades.append(trade_info)
    
    def get_analysis(self):
        return self.trades

class BacktestEngine:
    """향상된 백테스트 엔진"""
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'market_data')
        self.results_cache = {}
        
    def get_available_symbols(self):
        """사용 가능한 심볼 목록 반환"""
        symbols = []
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv'):
                    symbol = file.replace('.csv', '')
                    symbols.append(symbol)
        return symbols
    
    def get_data_info(self, symbol):
        """데이터 정보 반환"""
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            return None
            
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return {
                'start_date': df['Date'].min().strftime('%Y-%m-%d'),
                'end_date': df['Date'].max().strftime('%Y-%m-%d'),
                'total_rows': len(df)
            }
        except Exception as e:
            print(f"데이터 정보 조회 오류: {e}")
            return None
    
    def download_data(self, symbol, start_date, end_date, save_csv=True):
        """지정된 기간의 데이터 다운로드 및 CSV 저장"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}.csv")
            if not os.path.exists(file_path):
                return None
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            # 날짜 필터링
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            filtered_df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
            if len(filtered_df) == 0:
                return None
            # OHLCV 데이터로 변환
            ohlcv_data = []
            for _, row in filtered_df.iterrows():
                ohlcv_data.append({
                    'time': row['Date'].strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                })
            # CSV 저장
            saved_csv_path = None
            if save_csv:
                save_dir = os.path.join(self.data_dir, 'downloaded')
                os.makedirs(save_dir, exist_ok=True)
                saved_csv_path = os.path.join(save_dir, f"{symbol}_{start_date}_{end_date}.csv")
                filtered_df.to_csv(saved_csv_path, index=False)
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'data_points': len(ohlcv_data),
                'ohlcv': ohlcv_data,
                'csv_path': saved_csv_path
            }
        except Exception as e:
            print(f"데이터 다운로드 오류: {e}")
            return None
    
    def run_backtest(self, config, progress_callback=None, stop_flag=None):
        """백테스트 실행 (다운로드된 CSV 우선 사용, 실시간 진행상황 콜백 지원)"""
        try:
            # 설정 추출
            start_date_str = config.get('start_date')
            end_date_str = config.get('end_date')
            symbol = config.get('symbol', 'BTC_USDT')
            initial_capital = config.get('initial_capital', 10000)
            strategy_name = config.get('strategy', 'SmaCross')
            strategy_params = config.get('params', {})
            leverage = config.get('leverage', 1)
            position_pct = config.get('position_pct', 1.0)

            if progress_callback:
                progress_callback({'type': 'log', 'message': f"[시작] 백테스트 실행: {symbol} ({start_date_str} ~ {end_date_str})"})
                progress_callback({'type': 'log', 'message': f"전략: {strategy_name}, 파라미터: {strategy_params}"})

            # 중지 플래그: threading.Event 또는 [False] 리스트 등 사용 가능
            if stop_flag is None:
                stop_flag = [False]

            # 1. Cerebro 엔진 설정
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(initial_capital)
            cerebro.broker.setcommission(commission=0.001)  # 0.1% 수수료

            # 2. 전략 클래스 확인
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if not strategy_class:
                raise ValueError(f"전략 '{strategy_name}'을(를) 찾을 수 없습니다.")

            # 3. 데이터 로드: 다운로드된 CSV 우선 사용
            downloaded_path = os.path.join(self.data_dir, 'downloaded', f"{symbol}_{start_date_str}_{end_date_str}.csv")
            if os.path.exists(downloaded_path):
                file_path = downloaded_path
            else:
                file_path = os.path.join(self.data_dir, f"{symbol}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # 날짜 필터링(다운로드된 파일이면 이미 필터됨)
            if not os.path.exists(downloaded_path):
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            else:
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)

            if len(df) == 0:
                raise ValueError("선택한 기간에 데이터가 없습니다.")

            # 4. 데이터 피드 생성
            data_feed = bt.feeds.PandasData(
                dataname=df,
                datetime=None,  # 인덱스가 이미 날짜
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=-1
            )
            cerebro.adddata(data_feed)

            # 5. 분석기 추가
            cerebro.addanalyzer(EquityCurveAnalyzer, _name='equity')
            cerebro.addanalyzer(TradeLogAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analysis')

            # 6. 전략 실행
            if progress_callback:
                progress_callback({'type': 'log', 'message': '[실행] 전략 시뮬레이션 시작'})
            results = cerebro.run(strategy=strategy_class, **strategy_params)
            strategy = results[0]

            # 7. 결과 분석
            equity_analyzer = strategy.analyzers.equity.get_analysis()
            trade_analyzer = strategy.analyzers.trades.get_analysis()
            sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
            drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
            trade_analysis = strategy.analyzers.trade_analysis.get_analysis()

            # 8. 자산 곡선 데이터 생성
            equity_curve = []
            total_steps = len(equity_analyzer['equity_curve'])
            # 진행률/ETA 계산용 시작 시간
            start_time = time.time()
            for i, (date, equity) in enumerate(zip(equity_analyzer['dates'], equity_analyzer['equity_curve'])):
                # 중지 요청 시 즉시 종료
                if stop_flag and (hasattr(stop_flag, 'is_set') and stop_flag.is_set() or isinstance(stop_flag, list) and stop_flag[0]):
                    if progress_callback:
                        progress_callback({'type': 'log', 'message': '[중지] 백테스트가 사용자에 의해 중단되었습니다.'})
                        progress_callback({'type': 'stopped'})
                    return {
                        'success': False,
                        'error': '백테스트가 중지되었습니다.'
                    }
                equity_curve.append({
                    'time': date.isoformat(),
                    'value': float(equity)
                })

                # 진행상황 콜백 (실시간 업데이트용)
                if progress_callback and total_steps > 0 and i % max(1, total_steps // 100) == 0:
                    progress = (i / total_steps) * 100
                    eta = None
                    if i > 0:
                        elapsed = time.time() - start_time
                        eta = elapsed / (i / total_steps) - elapsed if i > 0 else None
                    progress_callback({
                        'type': 'progress',
                        'progress': progress,
                        'current_equity': float(equity),
                        'total_return': ((equity - initial_capital) / initial_capital) * 100,
                        'step': i,
                        'total_steps': total_steps,
                        'eta': eta
                    })

            # 9. 최종 결과 정리
            final_capital = strategy.broker.getvalue()
            total_return = ((final_capital - initial_capital) / initial_capital) * 100

            # 승률 계산
            total_trades = len(trade_analysis.get('total', {}).get('total', 0))
            won_trades = len(trade_analysis.get('won', {}).get('total', 0))
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

            # 최대 낙폭
            max_drawdown = drawdown_analyzer.get('max', {}).get('drawdown', 0) * 100

            # 샤프 지수
            sharpe_ratio = sharpe_analyzer.get('sharperatio', 0)

            # 매매 로그 포맷팅
            formatted_trades = []
            for trade in trade_analyzer:
                formatted_trades.append({
                    'date': trade['date'],
                    'time': trade['time'],
                    'symbol': trade['symbol'],
                    'action': trade['action'],
                    'size': trade['size'],
                    'price': f"{trade['price']:.2f}",
                    'value': f"{trade['value']:.2f}",
                    'commission': f"{trade['commission']:.2f}",
                    'pnl': f"{trade.get('pnl', 0):.2f}"
                })

            if progress_callback:
                progress_callback({'type': 'log', 'message': '[완료] 백테스트 종료'})
                progress_callback({'type': 'result', 'final_capital': final_capital, 'total_return': total_return, 'win_rate': win_rate, 'max_drawdown': max_drawdown, 'sharpe_ratio': sharpe_ratio, 'total_trades': total_trades})

            result = {
                'success': True,
                'symbol': symbol,
                'strategy': strategy_name,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': round(total_return, 2),
                'win_rate': round(win_rate, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'total_trades': total_trades,
                'equity_curve': equity_curve,
                'trades': formatted_trades,
                'performance_metrics': {
                    'total_return_pct': round(total_return, 2),
                    'annualized_return': round(total_return * (365 / (end_date - start_date).days), 2),
                    'volatility': round(np.std([p['value'] for p in equity_curve]) / initial_capital * 100, 2),
                    'max_drawdown_pct': round(max_drawdown, 2),
                    'sharpe_ratio': round(sharpe_ratio, 2),
                    'win_rate_pct': round(win_rate, 2),
                    'profit_factor': round(
                        sum([t['pnl'] for t in formatted_trades if float(t['pnl']) > 0]) / 
                        abs(sum([t['pnl'] for t in formatted_trades if float(t['pnl']) < 0])) if 
                        sum([t['pnl'] for t in formatted_trades if float(t['pnl']) < 0]) != 0 else 0, 2
                    )
                }
            }

            return result

        except Exception as e:
            print(f"백테스트 실행 오류: {e}")
            if progress_callback:
                progress_callback({'type': 'error', 'message': f"[에러] {str(e)}"})
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_strategies(self):
        """사용 가능한 전략 목록 반환"""
        strategies = []
        for name, strategy_class in STRATEGY_REGISTRY.items():
            if name == 'TripleCombo' and not TRIPLE_COMBO_AVAILABLE:
                continue
                
            # 전략별 기본 파라미터
            if name == 'SmaCross':
                params = {'pfast': 10, 'pslow': 30}
            elif name == 'RSI':
                params = {'period': 14, 'upper': 70, 'lower': 30}
            elif name == 'BollingerBands':
                params = {'period': 20, 'devfactor': 2}
            elif name == 'MACD':
                params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            elif name == 'TripleCombo':
                params = {}
            else:
                params = {}
                
            strategies.append({
                'name': name,
                'display_name': self._get_strategy_display_name(name),
                'description': self._get_strategy_description(name),
                'default_params': params
            })
        
        return strategies
    
    def _get_strategy_display_name(self, strategy_name):
        """전략 표시 이름"""
        names = {
            'SmaCross': '이동평균 교차 (SMA Cross)',
            'RSI': '상대강도지수 (RSI)',
            'BollingerBands': '볼린저 밴드 (Bollinger Bands)',
            'MACD': 'MACD (Moving Average Convergence Divergence)',
            'TripleCombo': '트리플 콤보 (Triple Combo)'
        }
        return names.get(strategy_name, strategy_name)
    
    def _get_strategy_description(self, strategy_name):
        """전략 설명"""
        descriptions = {
            'SmaCross': '단기와 장기 이동평균의 교차를 이용한 매매 전략',
            'RSI': '과매수/과매도 구간을 이용한 반전 매매 전략',
            'BollingerBands': '볼린저 밴드의 상단/하단 터치를 이용한 매매 전략',
            'MACD': 'MACD와 시그널 라인의 교차를 이용한 추세 추종 전략',
            'TripleCombo': '다중 지표를 조합한 고급 매매 전략'
        }
        return descriptions.get(strategy_name, '설명 없음')

def create_dummy_data_if_not_exists():
    """더미 데이터 생성 (기존 함수 유지)"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'market_data')
    os.makedirs(data_dir, exist_ok=True)

    # 날짜 범위는 함수 상단에서 한 번만 정의
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # BTC_USDT 더미 데이터 생성
    btc_file = os.path.join(data_dir, 'BTC_USDT.csv')
    if not os.path.exists(btc_file):
        print("BTC_USDT 더미 데이터 생성 중...")
        np.random.seed(42)  # 재현 가능한 랜덤 데이터
        base_price = 30000
        prices = []
        for i in range(len(dates)):
            if i == 0:
                price = base_price
            else:
                change = np.random.normal(0, 0.02)  # 2% 표준편차
                price = prices[-1] * (1 + change)
            prices.append(price)
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            daily_volatility = np.random.uniform(0.01, 0.03)
            open_price = price
            high_price = price * (1 + np.random.uniform(0, daily_volatility))
            low_price = price * (1 - np.random.uniform(0, daily_volatility))
            close_price = price * (1 + np.random.uniform(-daily_volatility/2, daily_volatility/2))
            volume = np.random.uniform(1000, 10000)
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': round(volume, 2)
            })
        df = pd.DataFrame(data)
        df.to_csv(btc_file, index=False)
        print(f"BTC_USDT 더미 데이터 생성 완료: {len(data)}개 데이터 포인트")

    # 다른 심볼들도 생성
    symbols = ['ETH_USDT', 'ADA_USDT', 'BNB_USDT', 'DOT_USDT']
    for symbol in symbols:
        symbol_file = os.path.join(data_dir, f'{symbol}.csv')
        if not os.path.exists(symbol_file):
            print(f"{symbol} 더미 데이터 생성 중...")
            base_prices = {
                'ETH_USDT': 2000,
                'ADA_USDT': 0.5,
                'BNB_USDT': 300,
                'DOT_USDT': 10
            }
            base_price = base_prices.get(symbol, 100)
            prices = []
            for i in range(len(dates)):
                if i == 0:
                    price = base_price
                else:
                    change = np.random.normal(0, 0.025)  # 2.5% 표준편차
                    price = prices[-1] * (1 + change)
                prices.append(price)
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                daily_volatility = np.random.uniform(0.015, 0.035)
                open_price = price
                high_price = price * (1 + np.random.uniform(0, daily_volatility))
                low_price = price * (1 - np.random.uniform(0, daily_volatility))
                close_price = price * (1 + np.random.uniform(-daily_volatility/2, daily_volatility/2))
                volume = np.random.uniform(5000, 50000)
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Open': round(open_price, 4),
                    'High': round(high_price, 4),
                    'Low': round(low_price, 4),
                    'Close': round(close_price, 4),
                    'Volume': round(volume, 2)
                })
            df = pd.DataFrame(data)
            df.to_csv(symbol_file, index=False)
            print(f"{symbol} 더미 데이터 생성 완료: {len(data)}개 데이터 포인트")

if __name__ == '__main__':
    create_dummy_data_if_not_exists() 