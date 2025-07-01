#!/usr/bin/env python3
"""
고급 백테스트 → 웹대시보드 실시간 반영
전체 매매전략 (다중시간프레임 + ML + CVD + 피쳐상승) 반영
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import requests
import json
from pathlib import Path

def safe_float(val):
    try:
        return float(val)
    except Exception:
        if hasattr(val, 'values'):
            return float(val.values[0])
        return float(val)

class AdvancedBacktestRealtime:
    """고급 백테스트 → 웹대시보드 실시간 전송"""
    
    def __init__(self, dashboard_url="http://localhost:5001"):
        self.dashboard_url = dashboard_url
        self.data_path = Path("data/market_data")
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 고급 백테스트 → 웹대시보드 실시간 전송 시스템 초기화")
        self.logger.info(f"🔗 대시보드 URL: {self.dashboard_url}")
        
        # 전략 설정
        self.strategies = {
            'multi_timeframe': True,
            'ml_prediction': True,
            'cvd_analysis': True,
            'feature_momentum': True,
            'dynamic_leverage': True
        }
    
    def send_log(self, message):
        """실시간 로그를 웹대시보드로 전송"""
        try:
            data = {
                'log': message,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            requests.post(
                f"{self.dashboard_url}/api/realtime_log",
                json=data,
                timeout=5
            )
            self.logger.info(message)
        except Exception as e:
            self.logger.error(f"❌ 로그 전송 실패: {e}")
    
    def send_report(self, results):
        """백테스트 결과를 웹대시보드로 전송"""
        try:
            requests.post(
                f"{self.dashboard_url}/api/report",
                json=results,
                timeout=10
            )
            self.logger.info("📊 백테스트 결과 웹대시보드로 전송 완료")
        except Exception as e:
            self.logger.error(f"❌ 결과 전송 실패: {e}")
    
    def load_multi_timeframe_data(self, symbol="BTC_USDT"):
        """다중시간프레임 데이터 로드"""
        self.send_log("📊 다중시간프레임 데이터 로딩 중...")
        
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        data = {}
        
        for tf in timeframes:
            filename = f"{symbol}_{tf}.csv"
            file_path = self.data_path / filename
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # timestamp 컬럼 처리
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                else:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df = df.set_index(df.columns[0])
                
                df = df.sort_index()
                data[tf] = df
                
                self.send_log(f"✅ {tf} 데이터 로드: {len(df)}개 레코드")
            else:
                self.send_log(f"⚠️ {filename} 파일 없음 - {tf} 전략 비활성화")
                self.strategies['multi_timeframe'] = False
        
        return data
    
    def calculate_technical_indicators(self, df):
        """기술적 지표 계산"""
        # 기본 지표
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['ma_200'] = df['close'].rolling(200).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 볼린저 밴드
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def simulate_ml_prediction(self, df):
        """ML 예측 시뮬레이션"""
        self.send_log("🤖 ML 예측 모델 시뮬레이션 중...")
        
        # 가상의 ML 예측 생성 (실제로는 학습된 모델 사용)
        np.random.seed(42)  # 재현 가능한 결과
        
        # 가격 변화율 기반 예측
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(20).std()
        
        # ML 예측 시뮬레이션 (가격 변화율 + 노이즈)
        df['ml_prediction'] = (
            df['price_change'].shift(1) * 0.3 +  # 이전 변화율
            df['rsi'].map(lambda x: (x - 50) / 100) * 0.2 +  # RSI 기반
            df['macd_histogram'] * 0.1 +  # MACD 기반
            np.random.normal(0, 0.01, len(df))  # 노이즈
        )
        
        # 예측 신호 생성
        df['ml_signal'] = 0
        df.loc[df['ml_prediction'] > 0.005, 'ml_signal'] = 1  # 강한 상승
        df.loc[df['ml_prediction'] > 0.002, 'ml_signal'] = 0.5  # 약한 상승
        df.loc[df['ml_prediction'] < -0.005, 'ml_signal'] = -1  # 강한 하락
        df.loc[df['ml_prediction'] < -0.002, 'ml_signal'] = -0.5  # 약한 하락
        
        self.send_log(f"✅ ML 예측 완료: 상승신호 {len(df[df['ml_signal'] > 0])}개, 하락신호 {len(df[df['ml_signal'] < 0])}개")
        return df
    
    def simulate_cvd_analysis(self, df):
        """CVD (Cumulative Volume Delta) 분석 시뮬레이션"""
        self.send_log("📊 CVD 분석 시뮬레이션 중...")
        
        # 가상의 거래량 델타 생성
        df['volume_delta'] = np.random.normal(0, df['volume'].mean() * 0.1, len(df))
        df['cvd'] = df['volume_delta'].cumsum()
        df['cvd_ma'] = df['cvd'].rolling(20).mean()
        
        # CVD 신호 생성
        df['cvd_signal'] = 0
        df.loc[df['cvd'] > df['cvd_ma'] * 1.2, 'cvd_signal'] = 1  # 매수 압력
        df.loc[df['cvd'] < df['cvd_ma'] * 0.8, 'cvd_signal'] = -1  # 매도 압력
        
        self.send_log(f"✅ CVD 분석 완료: 매수압력 {len(df[df['cvd_signal'] > 0])}개, 매도압력 {len(df[df['cvd_signal'] < 0])}개")
        return df
    
    def calculate_feature_momentum(self, df):
        """피쳐 상승 모멘텀 계산"""
        self.send_log("📈 피쳐 상승 모멘텀 계산 중...")
        
        # 여러 피쳐의 모멘텀 계산
        features = ['rsi', 'macd_histogram', 'close']
        
        for feature in features:
            if feature in df.columns:
                # 모멘텀 계산 (현재값 - 과거값)
                df[f'{feature}_momentum'] = df[feature] - df[feature].shift(5)
                df[f'{feature}_momentum_ma'] = df[f'{feature}_momentum'].rolling(10).mean()
        
        # 종합 모멘텀 신호
        momentum_signals = []
        for feature in features:
            if f'{feature}_momentum' in df.columns:
                momentum_signals.append(df[f'{feature}_momentum'])
        
        if momentum_signals:
            df['total_momentum'] = pd.concat(momentum_signals, axis=1).mean(axis=1)
            df['momentum_signal'] = 0
            df.loc[df['total_momentum'] > df['total_momentum'].rolling(20).std(), 'momentum_signal'] = 1
            df.loc[df['total_momentum'] < -df['total_momentum'].rolling(20).std(), 'momentum_signal'] = -1
        
        self.send_log("✅ 피쳐 모멘텀 계산 완료")
        return df
    
    def generate_composite_signal(self, df):
        """복합 신호 생성"""
        self.send_log("🎯 복합 매매 신호 생성 중...")
        
        # 각 전략별 신호 가중치
        signals = []
        weights = []
        
        # 기본 기술적 신호
        if 'ma_20' in df.columns and 'ma_50' in df.columns:
            tech_signal = np.where(df['ma_20'] > df['ma_50'], 1, -1)
            signals.append(tech_signal)
            weights.append(0.2)
        
        # RSI 신호
        if 'rsi' in df.columns:
            rsi_signal = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
            signals.append(rsi_signal)
            weights.append(0.15)
        
        # ML 신호
        if 'ml_signal' in df.columns:
            signals.append(df['ml_signal'].values)
            weights.append(0.3)
        
        # CVD 신호
        if 'cvd_signal' in df.columns:
            signals.append(df['cvd_signal'].values)
            weights.append(0.2)
        
        # 모멘텀 신호
        if 'momentum_signal' in df.columns:
            signals.append(df['momentum_signal'].values)
            weights.append(0.15)
        
        # 가중 평균으로 최종 신호 생성
        if signals:
            composite_signal = np.zeros(len(df))
            total_weight = sum(weights)
            
            for signal, weight in zip(signals, weights):
                composite_signal += signal * (weight / total_weight)
            
            df['composite_signal'] = composite_signal
            
            # 신호 강도에 따른 매매 결정
            df['final_signal'] = 0
            df.loc[df['composite_signal'] > 0.3, 'final_signal'] = 1  # 강한 매수
            df.loc[df['composite_signal'] < -0.3, 'final_signal'] = -1  # 강한 매도
        
        buy_signals = len(df[df['final_signal'] == 1])
        sell_signals = len(df[df['final_signal'] == -1])
        
        self.send_log(f"✅ 복합 신호 생성 완료: 매수 {buy_signals}개, 매도 {sell_signals}개")
        return df
    
    def calculate_dynamic_leverage(self, df, current_capital, initial_capital):
        """동적 레버리지 계산"""
        # 자본 변화에 따른 레버리지 조정
        capital_ratio = current_capital / initial_capital
        
        if capital_ratio > 1.1:  # 10% 이상 수익
            base_leverage = 1.5
        elif capital_ratio > 1.05:  # 5% 이상 수익
            base_leverage = 1.2
        elif capital_ratio < 0.95:  # 5% 이상 손실
            base_leverage = 0.5
        else:
            base_leverage = 1.0
        
        # 변동성에 따른 조정
        if 'atr' in df.columns:
            volatility_factor = df['atr'].rolling(20).mean() / df['close'].rolling(20).mean()
            leverage = base_leverage * (1 - volatility_factor * 10)  # 변동성 높으면 레버리지 감소
            leverage = np.clip(leverage, 0.1, 2.0)  # 0.1~2.0 범위로 제한
        else:
            leverage = base_leverage
        
        return leverage
    
    def run_advanced_backtest(self, df, initial_capital=10000000):
        self.send_log(f"🚀 고급 백테스트 시작 - 초기자본: ₩{initial_capital:,}")
        self.send_log("🎯 적용 전략: 다중시간프레임 + ML + CVD + 피쳐상승 + 동적레버리지")
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        capital_history = []
        commission = 0.001  # 0.1%
        total_rows = len(df)
        progress_step = max(1, total_rows // 20)
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i < 200:
                continue
            current_price = safe_float(row['close'])
            signal = row.get('final_signal', 0)
            if i % progress_step == 0:
                progress = (i / total_rows) * 100
                self.send_log(f"⏱️ 진행률: {progress:.1f}% ({i}/{total_rows})")
            leverage = self.calculate_dynamic_leverage(df.iloc[:i+1], capital, initial_capital)
            if position == 0 and signal == 1:
                position_size = (capital * 0.95 * leverage) / current_price
                position = position_size
                entry_price = current_price
                capital -= position * current_price * (1 + commission)
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'time': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'amount': position,
                    'leverage': leverage,
                    'capital_after': capital + position * current_price
                })
                self.send_log(f"📈 매수: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')} (레버리지: {leverage:.2f}x)")
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
                    'leverage': leverage,
                    'capital_after': capital
                })
                pnl_symbol = "💰" if pnl > 0 else "💸"
                self.send_log(f"📉 매도: ${current_price:,.2f} at {timestamp.strftime('%m-%d %H:%M')} {pnl_symbol} {pnl:+.2f}% (레버리지: {leverage:.2f}x)")
                position = 0
            current_value = capital + (position * current_price if position > 0 else 0)
            capital_history.append({
                'time': timestamp.strftime('%Y-%m-%d %H:%M'),
                'capital': current_value,
                'leverage': leverage
            })
        if position > 0:
            final_value = position * safe_float(df['close'].iloc[-1])
            capital += final_value * (1 - commission)
            self.send_log(f"🔄 최종 청산: ${safe_float(df['close'].iloc[-1]):,.2f}")
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100
        peak = initial_capital
        max_drawdown = 0
        for point in capital_history:
            if point['capital'] > peak:
                peak = point['capital']
            drawdown = (peak - point['capital']) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        total_trades_with_pnl = [t for t in trades if 'pnl' in t]
        win_rate = (len(winning_trades) / len(total_trades_with_pnl)) * 100 if total_trades_with_pnl else 0
        avg_leverage = np.mean([t.get('leverage', 1.0) for t in trades])
        results = {
            'final_capital': final_capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'capital_history': capital_history,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'avg_leverage': avg_leverage,
            'strategies_used': list(self.strategies.keys())
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
            
            # 데이터 로드
            data = self.load_multi_timeframe_data("BTC_USDT")
            if not data:
                self.send_log("❌ 데이터 로드 실패!")
                return
            
            # 1시간 데이터로 메인 백테스트
            df = data['1h'].copy()
            
            # 기술적 지표 계산
            df = self.calculate_technical_indicators(df)
            
            # ML 예측
            if self.strategies['ml_prediction']:
                df = self.simulate_ml_prediction(df)
            
            # CVD 분석
            if self.strategies['cvd_analysis']:
                df = self.simulate_cvd_analysis(df)
            
            # 피쳐 모멘텀
            if self.strategies['feature_momentum']:
                df = self.calculate_feature_momentum(df)
            
            # 복합 신호 생성
            df = self.generate_composite_signal(df)
            
            # 고급 백테스트 실행
            results = self.run_advanced_backtest(df)
            
            # 결과 요약
            self.send_log("🎉 고급 백테스트 완료!")
            self.send_log(f"💰 최종 자본: ₩{results['final_capital']:,.0f}")
            self.send_log(f"📈 총 수익률: {results['total_return']:+.2f}%")
            self.send_log(f"📉 최대 낙폭: {results['max_drawdown']:.2f}%")
            self.send_log(f"🎯 승률: {results['win_rate']:.1f}%")
            self.send_log(f"🔄 총 거래: {results['total_trades']}회")
            self.send_log(f"⚡ 평균 레버리지: {results['avg_leverage']:.2f}x")
            self.send_log(f"🎯 사용 전략: {', '.join(results['strategies_used'])}")
            
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
    print("🚀 고급 백테스트 → 웹대시보드 실시간 반영")
    print("=" * 60)
    print("🎯 전체 매매전략: 다중시간프레임 + ML + CVD + 피쳐상승 + 동적레버리지")
    print("🛑 중단하려면 Ctrl+C를 누르세요")
    print("=" * 60)
    
    dashboard_url = input("\n웹대시보드 URL (기본값: http://localhost:5001): ").strip()
    if not dashboard_url:
        dashboard_url = "http://localhost:5001"
    
    print(f"\n🔗 연결할 대시보드: {dashboard_url}")
    print("🚀 고급 백테스트를 시작합니다...\n")
    
    backtest = AdvancedBacktestRealtime(dashboard_url)
    backtest.run()