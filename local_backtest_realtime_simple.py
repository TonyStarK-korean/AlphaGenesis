#!/usr/bin/env python3
"""
로컬 데이터로 백테스트 실행 → 웹대시보드 실시간 반영 (간단 버전)
run_ml_backtest.py 없이 독립적으로 실행
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import requests
import json
from pathlib import Path

class LocalBacktestRealtime:
    """로컬 데이터 백테스트 → 웹대시보드 실시간 전송"""
    
    def __init__(self, dashboard_url="http://localhost:5001"):
        self.dashboard_url = dashboard_url
        self.data_path = Path("data/market_data")
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🏠 로컬 백테스트 → 웹대시보드 실시간 전송 시스템 초기화")
        self.logger.info(f"🔗 대시보드 URL: {self.dashboard_url}")
    
    def send_log(self, message):
        """실시간 로그를 웹대시보드로 전송"""
        try:
            data = {
                'log': message,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            response = requests.post(
                f"{self.dashboard_url}/api/realtime_log",
                json=data,
                timeout=5
            )
            
            # 로컬 로그도 출력
            self.logger.info(message)
            
        except Exception as e:
            self.logger.error(f"❌ 로그 전송 실패: {e}")
    
    def send_report(self, results):
        """백테스트 결과를 웹대시보드로 전송"""
        try:
            response = requests.post(
                f"{self.dashboard_url}/api/report",
                json=results,
                timeout=10
            )
            
            self.logger.info("📊 백테스트 결과 웹대시보드로 전송 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 결과 전송 실패: {e}")
    
    def check_data_files(self):
        """로컬 데이터 파일 확인"""
        self.send_log("📊 로컬 데이터 파일 확인 중...")
        
        data_files = []
        if self.data_path.exists():
            for file in self.data_path.glob("*.csv"):
                if "data_generator" not in file.name:
                    size_mb = file.stat().st_size / (1024*1024)
                    data_files.append({
                        'name': file.name,
                        'size_mb': round(size_mb, 1)
                    })
        
        self.send_log(f"✅ 총 {len(data_files)}개 데이터 파일 발견")
        for file in data_files[:5]:  # 처음 5개만 표시
            self.send_log(f"   📁 {file['name']} ({file['size_mb']}MB)")
        
        return len(data_files) > 0
    
    def load_data(self, symbol="BTC_USDT", timeframe="1h"):
        """데이터 로드"""
        try:
            filename = f"{symbol}_{timeframe}.csv"
            file_path = self.data_path / filename
            
            if not file_path.exists():
                self.send_log(f"❌ {filename} 파일을 찾을 수 없습니다!")
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
            
            self.send_log(f"✅ {filename} 데이터 로드 완료: {len(df)}개 레코드")
            self.send_log(f"📅 데이터 범위: {df.index[0]} ~ {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.send_log(f"❌ 데이터 로드 오류: {e}")
            return None
    
    def simple_trading_strategy(self, df):
        """간단한 매매 전략 (이동평균 크로스오버)"""
        self.send_log("🤖 간단한 이동평균 전략 적용 중...")
        
        # 기술적 지표 계산
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        
        # 매매 신호 생성
        df['signal'] = 0
        df.loc[(df['ma_20'] > df['ma_50']) & (df['rsi'] < 70), 'signal'] = 1  # 매수
        df.loc[(df['ma_20'] < df['ma_50']) | (df['rsi'] > 80), 'signal'] = -1  # 매도
        
        self.send_log("✅ 매매 신호 생성 완료")
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
        self.send_log(f"🚀 백테스트 시작 - 초기자본: ₩{initial_capital:,}")
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        capital_history = []
        
        # 수수료
        commission = 0.001  # 0.1%
        
        total_rows = len(df)
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i < 50:  # 지표 계산을 위해 처음 50개는 스킵
                continue
            
            current_price = row['close']
            signal = row['signal']
            
            # 진행률 업데이트 (5%마다)
            if i % max(1, total_rows // 20) == 0:
                progress = (i / total_rows) * 100
                self.send_log(f"⏱️ 진행률: {progress:.1f}% ({i}/{total_rows})")
            
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
                
                self.send_log(f"📈 매수: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')}")
                time.sleep(0.1)  # 실시간 느낌
            
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
                self.send_log(f"📉 매도: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')} {pnl_symbol} {pnl:+.2f}%")
                
                position = 0
                time.sleep(0.1)  # 실시간 느낌
            
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
            
            self.send_log(f"🔄 최종 청산: ${df['close'].iloc[-1]:,.2f}")
        
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
    
    def run(self):
        """메인 실행"""
        try:
            # 대시보드 연결 테스트
            self.send_log("🔗 웹대시보드 연결 테스트 중...")
            
            try:
                response = requests.get(f"{self.dashboard_url}/api/backtest/status", timeout=5)
                self.send_log("✅ 웹대시보드 연결 성공!")
                dashboard_connected = True
            except:
                self.send_log("⚠️ 웹대시보드 연결 실패 - 로컬에서만 실행")
                dashboard_connected = False
            
            # 데이터 파일 확인
            if not self.check_data_files():
                self.send_log("❌ 데이터 파일을 찾을 수 없습니다!")
                return
            
            # 데이터 로드
            df = self.load_data("BTC_USDT", "1h")
            if df is None:
                return
            
            # 매매 전략 적용
            df = self.simple_trading_strategy(df)
            
            # 백테스트 실행
            results = self.run_backtest(df)
            
            # 결과 요약
            self.send_log("🎉 백테스트 완료!")
            self.send_log(f"💰 최종 자본: ₩{results['final_capital']:,.0f}")
            self.send_log(f"📈 총 수익률: {results['total_return']:+.2f}%")
            self.send_log(f"📉 최대 낙폭: {results['max_drawdown']:.2f}%")
            self.send_log(f"🎯 승률: {results['win_rate']:.1f}%")
            self.send_log(f"🔄 총 거래: {results['total_trades']}회")
            
            # 웹대시보드로 결과 전송
            if dashboard_connected:
                self.send_report(results)
                self.send_log("📊 결과가 웹대시보드에 전송되었습니다!")
            
        except KeyboardInterrupt:
            self.send_log("🛑 사용자가 백테스트를 중단했습니다")
        except Exception as e:
            self.send_log(f"❌ 전체 오류: {e}")
            self.logger.error(f"전체 오류: {e}")

if __name__ == "__main__":
    print("🏠 로컬 데이터 백테스트 → 웹대시보드 실시간 반영")
    print("=" * 60)
    print("🔗 웹대시보드와 연결하여 실시간으로 결과를 전송합니다")
    print("🛑 중단하려면 Ctrl+C를 누르세요")
    print("=" * 60)
    
    # 대시보드 URL 입력받기
    dashboard_url = input("\n웹대시보드 URL (기본값: http://localhost:5001): ").strip()
    if not dashboard_url:
        dashboard_url = "http://localhost:5001"
    
    print(f"\n🔗 연결할 대시보드: {dashboard_url}")
    print("🚀 백테스트를 시작합니다...\n")
    
    backtest = LocalBacktestRealtime(dashboard_url)
    backtest.run()  