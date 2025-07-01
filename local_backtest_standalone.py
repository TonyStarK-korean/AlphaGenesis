#!/usr/bin/env python3
"""
로컬 데이터 백테스트 (독립 실행 버전)
웹대시보드 없이도 실행 가능
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

class LocalBacktestStandalone:
    """웹대시보드 없이 로컬에서만 실행되는 백테스트"""
    
    def __init__(self):
        self.data_path = Path("data/market_data")
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🏠 로컬 백테스트 시스템 초기화 (독립 실행)")
    
    def log_message(self, message):
        """로그 메시지 출력"""
        self.logger.info(message)
    
    def check_data_files(self):
        """로컬 데이터 파일 확인"""
        self.log_message("📊 로컬 데이터 파일 확인 중...")
        
        data_files = []
        if self.data_path.exists():
            for file in self.data_path.glob("*.csv"):
                if "data_generator" not in file.name:
                    size_mb = file.stat().st_size / (1024*1024)
                    data_files.append({
                        'name': file.name,
                        'size_mb': round(size_mb, 1)
                    })
        
        self.log_message(f"✅ 총 {len(data_files)}개 데이터 파일 발견")
        for file in data_files[:5]:  # 처음 5개만 표시
            self.log_message(f"   📁 {file['name']} ({file['size_mb']}MB)")
        
        return len(data_files) > 0, data_files
    
    def load_data(self, symbol="BTC_USDT", timeframe="1h"):
        """데이터 로드"""
        try:
            filename = f"{symbol}_{timeframe}.csv"
            file_path = self.data_path / filename
            
            if not file_path.exists():
                self.log_message(f"❌ {filename} 파일을 찾을 수 없습니다!")
                return None
            
            df = pd.read_csv(file_path)
            
            # timestamp 컬럼이 있는지 확인
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            else:
                # 첫 번째 컬럼을 timestamp로 사용
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
            
            df = df.sort_index()
            
            self.log_message(f"✅ {filename} 데이터 로드 완료: {len(df)}개 레코드")
            self.log_message(f"📅 데이터 범위: {df.index[0]} ~ {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.log_message(f"❌ 데이터 로드 오류: {e}")
            return None
    
    def simple_trading_strategy(self, df):
        """간단한 매매 전략 (이동평균 크로스오버)"""
        self.log_message("🤖 간단한 이동평균 전략 적용 중...")
        
        # 기술적 지표 계산
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        
        # 매매 신호 생성
        df['signal'] = 0
        df.loc[(df['ma_20'] > df['ma_50']) & (df['rsi'] < 70), 'signal'] = 1  # 매수
        df.loc[(df['ma_20'] < df['ma_50']) | (df['rsi'] > 80), 'signal'] = -1  # 매도
        
        # 신호 통계
        buy_signals = len(df[df['signal'] == 1])
        sell_signals = len(df[df['signal'] == -1])
        
        self.log_message(f"✅ 매매 신호 생성 완료: 매수 {buy_signals}개, 매도 {sell_signals}개")
        return df
    
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def run_backtest(self, df, initial_capital=10000000):
        """백테스트 실행"""
        self.log_message(f"🚀 백테스트 시작 - 초기자본: ₩{initial_capital:,}")
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        capital_history = []
        
        # 수수료
        commission = 0.001  # 0.1%
        
        total_rows = len(df)
        progress_step = max(1, total_rows // 20)  # 5%씩 업데이트
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i < 50:  # 지표 계산을 위해 처음 50개는 스킵
                continue
            
            current_price = row['close']
            signal = row['signal']
            
            # 진행률 업데이트
            if i % progress_step == 0:
                progress = (i / total_rows) * 100
                self.log_message(f"⏱️ 진행률: {progress:.1f}% ({i}/{total_rows})")
            
            # 매수 신호
            if position == 0 and signal == 1:
                position = (capital * 0.95) / current_price  # 95% 투자 (수수료 고려)
                entry_price = current_price
                capital -= position * current_price * (1 + commission)
                
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'time': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'amount': position,
                    'capital_after': capital + position * current_price
                })
                
                self.log_message(f"📈 매수: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')}")
            
            # 매도 신호
            elif position > 0 and signal == -1:
                sell_value = position * current_price * (1 - commission)
                capital += sell_value
                
                pnl = (current_price - entry_price) / entry_price * 100
                
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'time': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'amount': position,
                    'pnl': pnl,
                    'capital_after': capital
                })
                
                pnl_symbol = "💰" if pnl > 0 else "💸"
                self.log_message(f"📉 매도: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')} {pnl_symbol} {pnl:+.2f}%")
                
                position = 0
            
            # 자본 기록
            current_value = capital + (position * current_price if position > 0 else 0)
            capital_history.append({
                'time': timestamp.strftime('%Y-%m-%d %H:%M'),
                'capital': current_value
            })
        
        # 마지막에 포지션이 남아있으면 청산
        if position > 0:
            final_value = position * df['close'].iloc[-1]
            capital += final_value * (1 - commission)
            
            self.log_message(f"🔄 최종 청산: ${df['close'].iloc[-1]:,.2f}")
        
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # 최대 낙폭 계산
        peak = initial_capital
        max_drawdown = 0
        for point in capital_history:
            if point['capital'] > peak:
                peak = point['capital']
            drawdown = (peak - point['capital']) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 승률 계산
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        total_trades_with_pnl = [t for t in trades if 'pnl' in t]
        win_rate = (len(winning_trades) / len(total_trades_with_pnl)) * 100 if total_trades_with_pnl else 0
        
        results = {
            'final_capital': final_capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'capital_history': capital_history,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades)
        }
        
        return results
    
    def save_results(self, results, filename="backtest_results.json"):
        """결과를 JSON 파일로 저장"""
        try:
            import json
            
            results_path = Path("results")
            results_path.mkdir(exist_ok=True)
            
            file_path = results_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            self.log_message(f"💾 결과 저장 완료: {file_path}")
            
        except Exception as e:
            self.log_message(f"❌ 결과 저장 실패: {e}")
    
    def run(self):
        """메인 실행"""
        try:
            self.log_message("🏠 로컬 백테스트 시작")
            
            # 데이터 파일 확인
            has_data, data_files = self.check_data_files()
            if not has_data:
                self.log_message("❌ 데이터 파일을 찾을 수 없습니다!")
                return
            
            # 사용할 데이터 선택
            print("\n사용 가능한 데이터:")
            symbols = set()
            for file in data_files:
                symbol = file['name'].split('_')[0] + '_' + file['name'].split('_')[1]
                symbols.add(symbol)
            
            symbols = list(symbols)
            for i, symbol in enumerate(symbols):
                print(f"{i+1}. {symbol}")
            
            try:
                choice = int(input(f"\n선택 (1-{len(symbols)}, 기본값: 1): ") or "1") - 1
                selected_symbol = symbols[choice]
            except:
                selected_symbol = symbols[0]
            
            # 데이터 로드
            df = self.load_data(selected_symbol, "1h")
            if df is None:
                return
            
            # 매매 전략 적용
            df = self.simple_trading_strategy(df)
            
            # 백테스트 실행
            results = self.run_backtest(df)
            
            # 결과 요약
            print("\n" + "="*60)
            self.log_message("🎉 백테스트 완료!")
            self.log_message(f"💰 최종 자본: ₩{results['final_capital']:,.0f}")
            self.log_message(f"📈 총 수익률: {results['total_return']:+.2f}%")
            self.log_message(f"📉 최대 낙폭: {results['max_drawdown']:.2f}%")
            self.log_message(f"🎯 승률: {results['win_rate']:.1f}%")
            self.log_message(f"🔄 총 거래: {results['total_trades']}회")
            self.log_message(f"✅ 수익 거래: {results['winning_trades']}회")
            print("="*60)
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{selected_symbol}_{timestamp}.json"
            self.save_results(results, filename)
            
        except KeyboardInterrupt:
            self.log_message("🛑 사용자가 백테스트를 중단했습니다")
        except Exception as e:
            self.log_message(f"❌ 전체 오류: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🏠 로컬 데이터 백테스트 (독립 실행)")
    print("=" * 60)
    print("📊 웹대시보드 없이 로컬에서만 실행됩니다")
    print("🛑 중단하려면 Ctrl+C를 누르세요")
    print("=" * 60)
    
    backtest = LocalBacktestStandalone()
    backtest.run() 