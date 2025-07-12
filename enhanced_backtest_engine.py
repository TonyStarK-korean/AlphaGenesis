#!/usr/bin/env python3
"""
🚀 강화된 백테스트 엔진
4가지 전략 비교 및 전체 USDT.P 선물 백테스트 지원
"""

import pandas as pd
import numpy as np
import requests
import ccxt
from datetime import datetime, timedelta
import warnings
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json

warnings.filterwarnings('ignore')

from enhanced_strategies import (
    ComprehensiveStrategySystem, 
    EnhancedStrategy1, 
    EnhancedStrategy2,
    HourlyTradingStrategy
)

class EnhancedBacktestEngine:
    """
    강화된 백테스트 엔진
    - 4가지 전략 비교 (기본 vs 알파)
    - 전체 USDT.P 선물 백테스트
    - 실시간 진행 상황 모니터링
    """
    
    def __init__(self, initial_capital=10000, commission=0.0004):
        self.initial_capital = initial_capital
        self.commission = commission
        self.exchange = ccxt.binance({
            'apiKey': None,  # API 키 없이도 공개 데이터 사용 가능
            'secret': None,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # 4가지 전략 초기화
        self.strategies = {
            "strategy1_basic": HourlyTradingStrategy(),
            "strategy1_alpha": EnhancedStrategy1(),
            "strategy2_basic": HourlyTradingStrategy(),
            "strategy2_alpha": EnhancedStrategy2()
        }
        
        # 백테스트 결과 저장
        self.results = {}
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """진행 상황 콜백 함수 설정"""
        self.progress_callback = callback
    
    def _update_progress(self, message, percentage=None):
        """진행 상황 업데이트"""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    async def get_all_usdt_futures(self):
        """바이낸스에서 모든 USDT 선물 심볼 가져오기"""
        try:
            self._update_progress("바이낸스 USDT 선물 심볼 목록 가져오는 중...")
            
            # 바이낸스 선물 마켓 정보 가져오기
            markets = self.exchange.load_markets()
            
            # USDT 선물만 필터링
            usdt_futures = []
            for symbol, market in markets.items():
                if (market['type'] == 'future' and 
                    market['quote'] == 'USDT' and 
                    market['active'] and
                    symbol.endswith('/USDT')):
                    usdt_futures.append(symbol)
            
            self._update_progress(f"총 {len(usdt_futures)}개 USDT 선물 심볼 발견")
            return sorted(usdt_futures)
            
        except Exception as e:
            self._update_progress(f"심볼 목록 가져오기 실패: {e}")
            return []
    
    async def download_ohlcv_data(self, symbol, timeframe='1h', start_date=None, end_date=None):
        """OHLCV 데이터 다운로드"""
        try:
            # 날짜 처리
            if isinstance(start_date, str):
                start_timestamp = self.exchange.parse8601(start_date + 'T00:00:00Z')
            else:
                start_timestamp = self.exchange.parse8601(start_date.isoformat() + 'Z')
            
            if isinstance(end_date, str):
                end_timestamp = self.exchange.parse8601(end_date + 'T23:59:59Z')
            else:
                end_timestamp = self.exchange.parse8601(end_date.isoformat() + 'Z')
            
            # 데이터 다운로드
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=start_timestamp,
                limit=1000
            )
            
            if not ohlcv:
                return None
            
            # DataFrame으로 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 종료 날짜까지 필터링
            df = df[df.index <= pd.to_datetime(end_timestamp, unit='ms')]
            
            return df
            
        except Exception as e:
            self._update_progress(f"{symbol} 데이터 다운로드 실패: {e}")
            return None
    
    def run_strategy_backtest(self, strategy, df, symbol):
        """개별 전략 백테스트 실행"""
        try:
            if df is None or len(df) < 200:  # 최소 200개 데이터 포인트 필요
                return None
            
            # 전략별 신호 생성
            if hasattr(strategy, 'strategy1_early_surge'):
                signals = strategy.strategy1_early_surge(df)
            elif hasattr(strategy, 'strategy2_pullback_surge'):
                signals = strategy.strategy2_pullback_surge(df)
            else:
                signals = strategy.generate_signals(df)
            
            if signals is None or len(signals) == 0:
                return None
            
            # 백테스트 실행
            capital = self.initial_capital
            position = 0
            trades = []
            max_capital = capital
            min_capital = capital
            
            for i in range(len(df)):
                current_price = df['close'].iloc[i]
                
                # 매수 신호 확인
                if (i < len(signals) and 
                    signals['signal'].iloc[i] == 1 and 
                    position == 0):
                    
                    # 매수
                    position_size = capital * 0.95 / current_price
                    position = position_size
                    capital -= position_size * current_price * (1 + self.commission)
                    
                    entry_price = current_price
                    entry_index = i
                    
                    # 매도 조건 확인 (간단한 예시)
                    for j in range(i + 1, min(i + 48, len(df))):  # 최대 48시간 보유
                        exit_price = df['close'].iloc[j]
                        profit_pct = ((exit_price - entry_price) / entry_price) * 100
                        
                        # 5% 손절 또는 10% 익절 또는 시간 만료
                        if (profit_pct <= -5 or profit_pct >= 10 or 
                            j == min(i + 47, len(df) - 1)):
                            
                            capital += position * exit_price * (1 - self.commission)
                            
                            trades.append({
                                'entry_time': df.index[i],
                                'exit_time': df.index[j],
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit_pct': profit_pct,
                                'duration_hours': j - i,
                                'symbol': symbol
                            })
                            
                            position = 0
                            max_capital = max(max_capital, capital)
                            min_capital = min(min_capital, capital)
                            break
            
            # 결과 계산
            total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
            max_drawdown = ((min_capital - max_capital) / max_capital) * 100 if max_capital > 0 else 0
            
            winning_trades = [t for t in trades if t['profit_pct'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            
            avg_profit = np.mean([t['profit_pct'] for t in trades]) if trades else 0
            avg_duration = np.mean([t['duration_hours'] for t in trades]) if trades else 0
            
            return {
                'symbol': symbol,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_duration_hours': avg_duration,
                'final_capital': capital,
                'trades': trades[:10]  # 최근 10개 거래만 저장
            }
            
        except Exception as e:
            self._update_progress(f"{symbol} 백테스트 실패: {e}")
            return None
    
    async def run_four_strategy_comparison(self, symbol, start_date, end_date):
        """4가지 전략 비교 백테스트"""
        try:
            self._update_progress(f"{symbol} 데이터 다운로드 중...")
            
            # 데이터 다운로드
            df = await self.download_ohlcv_data(symbol, '1h', start_date, end_date)
            if df is None:
                return None
            
            self._update_progress(f"{symbol} 4가지 전략 백테스트 실행 중...")
            
            # 4가지 전략 백테스트
            strategy_results = {}
            
            for strategy_name, strategy in self.strategies.items():
                self._update_progress(f"{symbol} - {strategy_name} 전략 테스트 중...")
                result = self.run_strategy_backtest(strategy, df, symbol)
                if result:
                    strategy_results[strategy_name] = result
            
            return strategy_results
            
        except Exception as e:
            self._update_progress(f"{symbol} 4가지 전략 비교 실패: {e}")
            return None
    
    async def run_all_symbols_backtest(self, start_date, end_date, max_symbols=50):
        """전체 USDT.P 선물에 대한 백테스트"""
        try:
            # 모든 USDT 선물 심볼 가져오기
            all_symbols = await self.get_all_usdt_futures()
            
            if not all_symbols:
                self._update_progress("사용 가능한 심볼이 없습니다.")
                return {}
            
            # 심볼 수 제한 (API 제한 고려)
            symbols_to_test = all_symbols[:max_symbols]
            self._update_progress(f"상위 {len(symbols_to_test)}개 심볼에 대해 백테스트 실행")
            
            all_results = {}
            total_symbols = len(symbols_to_test)
            
            # 병렬 처리를 위한 세마포어 설정 (동시 요청 수 제한)
            semaphore = asyncio.Semaphore(5)
            
            async def process_symbol(symbol, index):
                async with semaphore:
                    try:
                        progress = (index + 1) / total_symbols * 100
                        self._update_progress(f"[{index+1}/{total_symbols}] {symbol} 처리 중...", progress)
                        
                        result = await self.run_four_strategy_comparison(symbol, start_date, end_date)
                        if result:
                            all_results[symbol] = result
                        
                        # API 제한 고려하여 잠시 대기
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        self._update_progress(f"{symbol} 처리 중 오류: {e}")
            
            # 모든 심볼 병렬 처리
            tasks = [process_symbol(symbol, i) for i, symbol in enumerate(symbols_to_test)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self._update_progress(f"백테스트 완료: {len(all_results)}개 심볼 처리됨")
            return all_results
            
        except Exception as e:
            self._update_progress(f"전체 심볼 백테스트 실패: {e}")
            return {}
    
    def analyze_results(self, results):
        """백테스트 결과 분석"""
        try:
            analysis = {
                'summary': {},
                'best_performers': {},
                'strategy_rankings': {},
                'recommendations': []
            }
            
            if not results:
                return analysis
            
            # 전략별 성과 집계
            strategy_performance = {
                'strategy1_basic': [],
                'strategy1_alpha': [],
                'strategy2_basic': [],
                'strategy2_alpha': []
            }
            
            for symbol, symbol_results in results.items():
                for strategy_name, strategy_result in symbol_results.items():
                    if strategy_result and 'total_return' in strategy_result:
                        strategy_performance[strategy_name].append({
                            'symbol': symbol,
                            'return': strategy_result['total_return'],
                            'win_rate': strategy_result['win_rate'],
                            'trades': strategy_result['total_trades']
                        })
            
            # 전략별 평균 성과 계산
            for strategy_name, performances in strategy_performance.items():
                if performances:
                    avg_return = np.mean([p['return'] for p in performances])
                    avg_win_rate = np.mean([p['win_rate'] for p in performances])
                    total_trades = sum([p['trades'] for p in performances])
                    
                    analysis['strategy_rankings'][strategy_name] = {
                        'average_return': avg_return,
                        'average_win_rate': avg_win_rate,
                        'total_trades': total_trades,
                        'tested_symbols': len(performances),
                        'best_symbol': max(performances, key=lambda x: x['return'])['symbol'],
                        'best_return': max(performances, key=lambda x: x['return'])['return']
                    }
            
            # 추천사항 생성
            if analysis['strategy_rankings']:
                best_strategy = max(analysis['strategy_rankings'].items(), 
                                  key=lambda x: x[1]['average_return'])
                
                analysis['recommendations'].append({
                    'type': 'best_strategy',
                    'strategy': best_strategy[0],
                    'reason': f"평균 수익률 {best_strategy[1]['average_return']:.2f}%로 최고 성과",
                    'details': best_strategy[1]
                })
                
                # 알파 전략 vs 기본 전략 비교
                alpha_vs_basic = self._compare_alpha_strategies(analysis['strategy_rankings'])
                if alpha_vs_basic:
                    analysis['recommendations'].extend(alpha_vs_basic)
            
            return analysis
            
        except Exception as e:
            self._update_progress(f"결과 분석 실패: {e}")
            return {'summary': {}, 'best_performers': {}, 'strategy_rankings': {}, 'recommendations': []}
    
    def _compare_alpha_strategies(self, strategy_rankings):
        """알파 전략과 기본 전략 비교"""
        recommendations = []
        
        try:
            # 전략 1 비교
            if ('strategy1_basic' in strategy_rankings and 
                'strategy1_alpha' in strategy_rankings):
                basic = strategy_rankings['strategy1_basic']['average_return']
                alpha = strategy_rankings['strategy1_alpha']['average_return']
                improvement = alpha - basic
                
                if improvement > 2:  # 2% 이상 개선
                    recommendations.append({
                        'type': 'alpha_improvement',
                        'strategy': 'strategy1_alpha',
                        'reason': f"급등 초입 전략에서 알파 지표로 {improvement:.2f}% 성과 개선",
                        'improvement': improvement
                    })
                elif improvement < -2:  # 2% 이상 악화
                    recommendations.append({
                        'type': 'alpha_degradation',
                        'strategy': 'strategy1_basic',
                        'reason': f"급등 초입 전략에서 알파 지표가 {abs(improvement):.2f}% 성과 악화",
                        'degradation': abs(improvement)
                    })
            
            # 전략 2 비교
            if ('strategy2_basic' in strategy_rankings and 
                'strategy2_alpha' in strategy_rankings):
                basic = strategy_rankings['strategy2_basic']['average_return']
                alpha = strategy_rankings['strategy2_alpha']['average_return']
                improvement = alpha - basic
                
                if improvement > 2:  # 2% 이상 개선
                    recommendations.append({
                        'type': 'alpha_improvement',
                        'strategy': 'strategy2_alpha',
                        'reason': f"눌림목 후 급등 전략에서 알파 지표로 {improvement:.2f}% 성과 개선",
                        'improvement': improvement
                    })
                elif improvement < -2:  # 2% 이상 악화
                    recommendations.append({
                        'type': 'alpha_degradation',
                        'strategy': 'strategy2_basic',
                        'reason': f"눌림목 후 급등 전략에서 알파 지표가 {abs(improvement):.2f}% 성과 악화",
                        'degradation': abs(improvement)
                    })
                    
        except Exception as e:
            self._update_progress(f"알파 전략 비교 실패: {e}")
        
        return recommendations

# 백테스트 엔진 인스턴스
enhanced_backtest_engine = EnhancedBacktestEngine()

if __name__ == "__main__":
    print("🚀 강화된 백테스트 엔진 시작")
    
    # 테스트 실행
    import asyncio
    
    async def test_backtest():
        engine = EnhancedBacktestEngine(initial_capital=10000)
        
        # 테스트용 진행 상황 콜백
        def progress_callback(message, percentage=None):
            if percentage:
                print(f"[{percentage:.1f}%] {message}")
            else:
                print(f"[INFO] {message}")
        
        engine.set_progress_callback(progress_callback)
        
        # 4가지 전략 비교 테스트
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        
        results = await engine.run_four_strategy_comparison("BTC/USDT", start_date, end_date)
        if results:
            print("=== 4가지 전략 비교 결과 ===")
            for strategy_name, result in results.items():
                print(f"{strategy_name}: {result['total_return']:.2f}% 수익률")
    
    # asyncio.run(test_backtest())