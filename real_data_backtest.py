#!/usr/bin/env python3
"""
실제 데이터 백테스트 시스템
- API 연동 (Binance, CoinGecko)
- CSV 파일 로드
- 실제 비트코인 데이터 기반 백테스트
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import time
import json

# API 라이브러리들 (선택적 임포트)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests 라이브러리가 없습니다. pip install requests")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ ccxt 라이브러리가 없습니다. pip install ccxt")

class RealDataLoader:
    """실제 데이터 로더 클래스"""
    
    def __init__(self):
        self.data_cache = {}
    
    def load_from_binance_api(self, symbol='BTCUSDT', interval='1h', limit=1000):
        """Binance API에서 실제 데이터 로드"""
        print(f"📡 Binance API에서 {symbol} 데이터 로드 중...")
        
        if not REQUESTS_AVAILABLE:
            print("❌ requests 라이브러리가 필요합니다.")
            return None
        
        try:
            # Binance API URL
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            print(f"   요청: {symbol}, {interval}, 최근 {limit}개")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # 데이터 변환
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # 필요한 컬럼만 선택하고 형변환
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 타임스탬프 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['datetime'] = df['timestamp']
                
                print(f"✅ Binance 데이터 로드 성공: {len(df)}개 캔들")
                print(f"   기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
                print(f"   가격 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
                
                return df
                
            else:
                print(f"❌ Binance API 오류: {response.status_code}")
                print(f"   응답: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ Binance API 연결 실패: {e}")
            return None
    
    def load_from_coingecko_api(self, coin_id='bitcoin', vs_currency='usd', days=30):
        """CoinGecko API에서 실제 데이터 로드"""
        print(f"📡 CoinGecko API에서 {coin_id} 데이터 로드 중...")
        
        if not REQUESTS_AVAILABLE:
            print("❌ requests 라이브러리가 필요합니다.")
            return None
        
        try:
            # CoinGecko API URL
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            print(f"   요청: {coin_id}, {days}일간 데이터")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # 가격과 거래량 데이터 추출
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                
                if not prices:
                    print("❌ 가격 데이터가 없습니다.")
                    return None
                
                # DataFrame 생성
                df_data = []
                for i, (timestamp, price) in enumerate(prices):
                    volume = volumes[i][1] if i < len(volumes) else 0
                    
                    df_data.append({
                        'timestamp': pd.to_datetime(timestamp, unit='ms'),
                        'datetime': pd.to_datetime(timestamp, unit='ms'),
                        'open': price,  # CoinGecko는 OHLC 제공 안함
                        'high': price * 1.02,  # 근사치
                        'low': price * 0.98,   # 근사치
                        'close': price,
                        'volume': volume
                    })
                
                df = pd.DataFrame(df_data)
                
                print(f"✅ CoinGecko 데이터 로드 성공: {len(df)}개 포인트")
                print(f"   기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
                print(f"   가격 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
                
                return df
                
            else:
                print(f"❌ CoinGecko API 오류: {response.status_code}")
                if response.status_code == 429:
                    print("   API 호출 한도 초과. 잠시 후 다시 시도하세요.")
                return None
                
        except Exception as e:
            print(f"❌ CoinGecko API 연결 실패: {e}")
            return None
    
    def load_from_ccxt(self, exchange_name='binance', symbol='BTC/USDT', timeframe='1h', limit=1000):
        """CCXT 라이브러리로 거래소 데이터 로드"""
        print(f"📡 CCXT로 {exchange_name}에서 {symbol} 데이터 로드 중...")
        
        if not CCXT_AVAILABLE:
            print("❌ ccxt 라이브러리가 필요합니다. pip install ccxt")
            return None
        
        try:
            # 거래소 초기화
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'rateLimit': 1200,
                'enableRateLimit': True,
            })
            
            print(f"   요청: {symbol}, {timeframe}, 최근 {limit}개")
            
            # OHLCV 데이터 가져오기
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                print("❌ 데이터가 없습니다.")
                return None
            
            # DataFrame 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['timestamp'] = df['datetime']
            
            print(f"✅ CCXT 데이터 로드 성공: {len(df)}개 캔들")
            print(f"   기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
            print(f"   가격 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ CCXT 데이터 로드 실패: {e}")
            return None
    
    def load_from_csv(self, file_path, datetime_col='datetime', price_cols=None):
        """CSV 파일에서 데이터 로드"""
        print(f"📂 CSV 파일에서 데이터 로드 중: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                print(f"❌ 파일이 존재하지 않습니다: {file_path}")
                return None
            
            # CSV 로드
            df = pd.read_csv(file_path)
            print(f"   원본 데이터: {len(df)}행, {len(df.columns)}열")
            print(f"   컬럼: {list(df.columns)}")
            
            # 기본 컬럼 매핑
            column_mapping = {
                'time': 'datetime',
                'timestamp': 'datetime',
                'date': 'datetime',
                'price': 'close',
                'close_price': 'close',
                'vol': 'volume',
                'volume_24h': 'volume'
            }
            
            # 컬럼명 정규화
            df.columns = df.columns.str.lower().str.strip()
            
            # 매핑 적용
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # 필수 컬럼 확인
            required_cols = ['datetime', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ 필수 컬럼이 없습니다: {missing_cols}")
                print(f"   사용 가능한 컬럼: {list(df.columns)}")
                return None
            
            # 날짜 변환
            if datetime_col in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(df[datetime_col])
                except:
                    print(f"❌ 날짜 컬럼 변환 실패: {datetime_col}")
                    return None
            
            # 숫자 컬럼 변환
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # OHLC 컬럼이 없으면 close 가격으로 채우기
            if 'open' not in df.columns:
                df['open'] = df['close']
            if 'high' not in df.columns:
                df['high'] = df['close'] * 1.01
            if 'low' not in df.columns:
                df['low'] = df['close'] * 0.99
            if 'volume' not in df.columns:
                df['volume'] = 1000  # 기본값
            
            # 정렬
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # timestamp 컬럼 추가
            df['timestamp'] = df['datetime']
            
            print(f"✅ CSV 데이터 로드 성공: {len(df)}개 행")
            print(f"   기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
            print(f"   가격 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ CSV 파일 로드 실패: {e}")
            return None
    
    def save_to_csv(self, df, file_path):
        """데이터를 CSV 파일로 저장"""
        try:
            df.to_csv(file_path, index=False)
            print(f"✅ 데이터 저장 완료: {file_path}")
            return True
        except Exception as e:
            print(f"❌ 데이터 저장 실패: {e}")
            return False

def calculate_real_indicators(df):
    """실제 데이터용 기술적 지표 계산"""
    print("📈 실제 데이터 기반 기술적 지표 계산 중...")
    
    try:
        # 이동평균
        df['ma_5'] = df['close'].rolling(5, min_periods=1).mean()
        df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['ma_50'] = df['close'].rolling(50, min_periods=1).mean()
        df['ma_200'] = df['close'].rolling(200, min_periods=1).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 볼린저 밴드
        bb_window = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_window, min_periods=1).mean()
        bb_rolling_std = df['close'].rolling(bb_window, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR (Average True Range)
        df['prev_close'] = df['close'].shift(1)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['prev_close'])
        df['low_close'] = np.abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14, min_periods=1).mean()
        
        # 변동성
        df['volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std()
        df['returns'] = df['close'].pct_change()
        
        # 거래량 분석
        df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        
        # Price Action 지표
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_20'] = df['close'].pct_change(20)
        
        # 지지저항 레벨 (간단한 버전)
        rolling_max = df['close'].rolling(20, min_periods=1).max()
        rolling_min = df['close'].rolling(20, min_periods=1).min()
        df['resistance_level'] = rolling_max
        df['support_level'] = rolling_min
        df['price_position'] = (df['close'] - df['support_level']) / (df['resistance_level'] - df['support_level'] + 1e-10)
        
        # NaN 처리
        df = df.ffill().fillna(0)
        
        # 불필요한 컬럼 제거
        cols_to_drop = ['prev_close', 'high_low', 'high_close', 'low_close', 'true_range']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        print(f"✅ 기술적 지표 계산 완료: {len(df.columns)}개 컬럼")
        
        return df
        
    except Exception as e:
        print(f"❌ 지표 계산 오류: {e}")
        return df

def real_ml_prediction(df):
    """실제 데이터 기반 ML 예측"""
    print("🤖 실제 데이터 기반 ML 예측 중...")
    
    try:
        predictions = []
        
        for i in range(len(df)):
            if i < 50:
                predictions.append(0.0)
                continue
            
            # 다양한 시간 프레임의 특징 추출
            current_row = df.iloc[i]
            
            # 1. 기술적 지표 신호
            rsi_signal = (current_row['rsi'] - 50) / 50
            macd_signal = 1 if current_row['macd'] > current_row['macd_signal'] else -1
            bb_signal = (current_row['bb_position'] - 0.5) * 2  # -1 ~ 1
            
            # 2. 이동평균 신호
            ma_short_signal = (current_row['ma_5'] - current_row['ma_20']) / (current_row['ma_20'] + 1e-10)
            ma_long_signal = (current_row['ma_20'] - current_row['ma_50']) / (current_row['ma_50'] + 1e-10)
            
            # 3. 모멘텀 신호
            momentum_short = current_row['price_change_5']
            momentum_long = current_row['price_change_20']
            
            # 4. 변동성 신호
            vol_signal = current_row['volatility']
            atr_signal = current_row['atr'] / (current_row['close'] + 1e-10)
            
            # 5. 거래량 신호
            volume_signal = (current_row['volume_ratio'] - 1) * 0.1
            
            # 6. 지지저항 신호
            support_resistance_signal = (current_row['price_position'] - 0.5) * 2
            
            # 종합 예측 (앙상블)
            prediction = (
                rsi_signal * 0.15 +
                macd_signal * 0.15 +
                bb_signal * 0.10 +
                ma_short_signal * 0.20 +
                ma_long_signal * 0.15 +
                momentum_short * 0.10 +
                momentum_long * 0.05 +
                vol_signal * 0.05 +
                volume_signal * 0.03 +
                support_resistance_signal * 0.02
            )
            
            # 범위 제한
            prediction = max(min(prediction, 0.1), -0.1)
            predictions.append(prediction)
        
        strong_signals = [p for p in predictions if abs(p) > 0.01]
        print(f"✅ ML 예측 완료")
        print(f"   강한 신호: {len(strong_signals)}개 ({len(strong_signals)/len(predictions)*100:.1f}%)")
        print(f"   예측 범위: {min(predictions):.4f} ~ {max(predictions):.4f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ ML 예측 오류: {e}")
        return [0.0] * len(df)

def analyze_real_market_condition(row):
    """실제 데이터 기반 시장 상황 분석"""
    try:
        # 다양한 지표 종합 분석
        ma_5 = row.get('ma_5', row['close'])
        ma_20 = row.get('ma_20', row['close'])
        ma_50 = row.get('ma_50', row['close'])
        rsi = row.get('rsi', 50)
        volatility = row.get('volatility', 0.02)
        volume_ratio = row.get('volume_ratio', 1.0)
        bb_position = row.get('bb_position', 0.5)
        
        # 추세 강도 계산
        ma_trend_short = (ma_5 - ma_20) / (ma_20 + 1e-10)
        ma_trend_long = (ma_20 - ma_50) / (ma_50 + 1e-10)
        
        # 시장 상황 판단
        if ma_trend_short > 0.005 and ma_trend_long > 0.002 and rsi < 70:
            return 'strong_bullish'
        elif ma_trend_short > 0.002 and volatility < 0.03:
            return 'bullish'
        elif ma_trend_short < -0.005 and ma_trend_long < -0.002 and rsi > 30:
            return 'strong_bearish'
        elif ma_trend_short < -0.002 and volatility < 0.03:
            return 'bearish'
        elif volatility > 0.05 or volume_ratio > 2.0:
            return 'high_volatility'
        elif abs(ma_trend_short) < 0.001 and volatility < 0.02:
            return 'consolidation'
        else:
            return 'neutral'
            
    except:
        return 'neutral'

def real_enhanced_strategy(row, ml_pred, market_condition):
    """실제 데이터 기반 강화된 전략"""
    
    signal = {
        'action': 'HOLD',
        'confidence': 0.0,
        'strategy_used': 'none'
    }
    
    try:
        # 안전한 값 추출
        close = row['close']
        rsi = row.get('rsi', 50)
        macd = row.get('macd', 0)
        macd_signal_line = row.get('macd_signal', 0)
        bb_position = row.get('bb_position', 0.5)
        volume_ratio = row.get('volume_ratio', 1.0)
        atr = row.get('atr', close * 0.02)
        
        # ML 예측 강도
        ml_strength = abs(ml_pred)
        
        # 전략 1: 강한 추세 추종
        if market_condition in ['strong_bullish', 'strong_bearish']:
            confidence = 0.0
            
            if market_condition == 'strong_bullish' and ml_pred > 0.01:
                if rsi < 75 and volume_ratio > 1.1 and macd > macd_signal_line:
                    confidence = min(0.9, ml_strength * 40)
                    signal['action'] = 'BUY'
                    signal['strategy_used'] = 'strong_trend_following'
            
            elif market_condition == 'strong_bearish' and ml_pred < -0.01:
                if rsi > 25 and volume_ratio > 1.1 and macd < macd_signal_line:
                    confidence = min(0.9, ml_strength * 40)
                    signal['action'] = 'SELL'
                    signal['strategy_used'] = 'strong_trend_following'
            
            signal['confidence'] = confidence
        
        # 전략 2: 일반 추세 추종
        elif market_condition in ['bullish', 'bearish']:
            confidence = 0.0
            
            if market_condition == 'bullish' and ml_pred > 0.005:
                if 30 < rsi < 70 and volume_ratio > 0.9:
                    confidence = min(0.7, ml_strength * 30)
                    signal['action'] = 'BUY'
                    signal['strategy_used'] = 'trend_following'
            
            elif market_condition == 'bearish' and ml_pred < -0.005:
                if 30 < rsi < 70 and volume_ratio > 0.9:
                    confidence = min(0.7, ml_strength * 30)
                    signal['action'] = 'SELL'
                    signal['strategy_used'] = 'trend_following'
            
            signal['confidence'] = confidence
        
        # 전략 3: 평균 회귀 (횡보장)
        elif market_condition in ['consolidation', 'neutral']:
            confidence = 0.0
            
            # 볼린저 밴드 기반 평균 회귀
            if bb_position < 0.2 and ml_pred > 0.003:  # 하단 근처에서 상승 신호
                confidence = min(0.6, ml_strength * 25)
                signal['action'] = 'BUY'
                signal['strategy_used'] = 'mean_reversion'
            
            elif bb_position > 0.8 and ml_pred < -0.003:  # 상단 근처에서 하락 신호
                confidence = min(0.6, ml_strength * 25)
                signal['action'] = 'SELL'
                signal['strategy_used'] = 'mean_reversion'
            
            signal['confidence'] = confidence
        
        # 전략 4: 변동성 돌파
        elif market_condition == 'high_volatility':
            confidence = 0.0
            
            if ml_strength > 0.02 and volume_ratio > 1.5:
                confidence = min(0.8, ml_strength * 20)
                if ml_pred > 0:
                    signal['action'] = 'BUY'
                else:
                    signal['action'] = 'SELL'
                signal['strategy_used'] = 'volatility_breakout'
            
            signal['confidence'] = confidence
        
        return signal
        
    except Exception as e:
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'strategy_used': 'error'
        }

def run_real_backtest(df, predictions, min_confidence=0.4):
    """실제 데이터 기반 백테스트 실행"""
    print(f"💰 실제 데이터 백테스트 실행 중... (최소 신뢰도: {min_confidence})")
    
    try:
        initial_capital = 10000000
        capital = initial_capital
        position = 0  # 0: 현금, 1: 롱
        shares = 0
        trades = []
        portfolio_values = []
        
        # 상세 통계
        signal_count = 0
        executed_count = 0
        
        strategy_stats = {
            'strong_trend_following': {'count': 0, 'profit': 0, 'total_trades': 0},
            'trend_following': {'count': 0, 'profit': 0, 'total_trades': 0},
            'mean_reversion': {'count': 0, 'profit': 0, 'total_trades': 0},
            'volatility_breakout': {'count': 0, 'profit': 0, 'total_trades': 0}
        }
        
        print("📊 실제 데이터 백테스트 진행 상황:")
        print("-" * 70)
        
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                current_price = row['close']
                current_time = row.get('datetime', f"포인트_{i}")
                ml_pred = predictions[i] if i < len(predictions) else 0.0
                
                # 진행률 표시
                if i % max(1, len(df)//20) == 0:
                    progress = (i / len(df)) * 100
                    print(f"   진행: {progress:.1f}% | {current_time} | 가격: {current_price:.2f} | 자본: {capital:,.0f}")
                
                # 시장 상황 분석
                market_condition = analyze_real_market_condition(row)
                
                # 강화된 전략 신호 생성
                signal = real_enhanced_strategy(row, ml_pred, market_condition)
                
                # 신호 통계
                if signal['action'] != 'HOLD':
                    signal_count += 1
                    if signal_count <= 5:  # 처음 5개 신호만 로그
                        print(f"   🎯 신호 {signal_count}: {signal['action']} | 신뢰도: {signal['confidence']:.3f} | "
                              f"전략: {signal['strategy_used']} | 시장: {market_condition}")
                
                # 포트폴리오 가치 계산
                if position != 0:
                    portfolio_value = shares * current_price
                else:
                    portfolio_value = capital
                portfolio_values.append(portfolio_value)
                
                # 거래 실행 (롱 포지션만 - 단순화)
                if signal['confidence'] >= min_confidence:
                    if signal['action'] == 'BUY' and position == 0:
                        # 매수
                        shares = capital / current_price
                        position = 1
                        entry_capital = capital
                        capital = 0
                        executed_count += 1
                        
                        trades.append({
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'strategy': signal['strategy_used'],
                            'confidence': signal['confidence'],
                            'datetime': current_time,
                            'ml_pred': ml_pred,
                            'market_condition': market_condition,
                            'entry_capital': entry_capital
                        })
                        
                        print(f"   ✅ 매수 체결 #{executed_count}: {current_price:.2f} | "
                              f"{signal['strategy_used']} | 신뢰도: {signal['confidence']:.2f}")
                        
                        # 전략 통계
                        if signal['strategy_used'] in strategy_stats:
                            strategy_stats[signal['strategy_used']]['count'] += 1
                    
                    elif signal['action'] == 'SELL' and position == 1:
                        # 매도 (롱 포지션 청산)
                        capital = shares * current_price
                        profit = capital - trades[-1]['entry_capital'] if trades else 0
                        
                        position = 0
                        shares = 0
                        executed_count += 1
                        
                        trades.append({
                            'type': 'SELL',
                            'price': current_price,
                            'shares': 0,
                            'strategy': signal['strategy_used'],
                            'confidence': signal['confidence'],
                            'datetime': current_time,
                            'ml_pred': ml_pred,
                            'market_condition': market_condition,
                            'profit': profit
                        })
                        
                        print(f"   ✅ 매도 체결 #{executed_count}: {current_price:.2f} | "
                              f"수익: {profit:+,.0f}원 | {signal['strategy_used']}")
                        
                        # 전략별 수익 기록
                        last_buy_trade = None
                        for trade in reversed(trades):
                            if trade['type'] == 'BUY':
                                last_buy_trade = trade
                                break
                        
                        if last_buy_trade and last_buy_trade['strategy'] in strategy_stats:
                            strategy_stats[last_buy_trade['strategy']]['profit'] += profit
                            strategy_stats[last_buy_trade['strategy']]['total_trades'] += 1
                
            except Exception as e:
                print(f"   ⚠️ 행 처리 오류 (idx={i}): {e}")
                portfolio_values.append(portfolio_values[-1] if portfolio_values else initial_capital)
                continue
        
        # 최종 정산
        if position != 0:
            final_price = df['close'].iloc[-1]
            capital = shares * final_price
            final_profit = capital - trades[-1]['entry_capital'] if trades else 0
            print(f"   🔚 최종 정산: {final_price:.2f} | 수익: {final_profit:+,.0f}원")
        
        print(f"\n📊 백테스트 완료")
        print(f"   총 신호: {signal_count}개, 실행 거래: {executed_count}개")
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'strategy_stats': strategy_stats,
            'signal_count': signal_count,
            'executed_count': executed_count,
            'data_period': f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}",
            'data_points': len(df)
        }
        
    except Exception as e:
        print(f"❌ 백테스트 실행 오류: {e}")
        return {
            'initial_capital': initial_capital,
            'final_capital': initial_capital,
            'trades': [],
            'portfolio_values': [initial_capital],
            'strategy_stats': {},
            'error': str(e)
        }

def analyze_real_results(results):
    """실제 데이터 백테스트 결과 분석"""
    print("\n" + "="*80)
    print("📊 실제 데이터 백테스트 결과 분석")
    print("="*80)
    
    if 'error' in results:
        print(f"❌ 백테스트 실패: {results['error']}")
        return
    
    initial = results['initial_capital']
    final = results['final_capital']
    total_return = (final - initial) / initial * 100
    
    print(f"💰 자본 변화:")
    print(f"   초기 자본: {initial:,.0f}원")
    print(f"   최종 자본: {final:,.0f}원")
    print(f"   절대 수익: {final - initial:+,.0f}원")
    print(f"   수익률: {total_return:+.2f}%")
    
    print(f"\n📈 데이터 정보:")
    print(f"   데이터 기간: {results.get('data_period', 'N/A')}")
    print(f"   데이터 포인트: {results.get('data_points', 0):,}개")
    
    # 거래 분석
    trades = results['trades']
    print(f"\n📊 거래 통계:")
    print(f"   신호 발생: {results.get('signal_count', 0)}개")
    print(f"   실행 거래: {results.get('executed_count', 0)}개")
    print(f"   실행률: {results.get('executed_count', 0)/results.get('signal_count', 1)*100:.1f}%")
    
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        print(f"   총 매수: {len(buy_trades)}회")
        print(f"   총 매도: {len(sell_trades)}회")
        
        # 수익 거래 분석
        profitable_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('profit', 0) < 0]
        
        if sell_trades:
            print(f"   수익 거래: {len(profitable_trades)}회")
            print(f"   손실 거래: {len(losing_trades)}회")
            print(f"   승률: {len(profitable_trades)/len(sell_trades)*100:.1f}%")
            
            if profitable_trades:
                avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades)
                print(f"   평균 수익: {avg_profit:,.0f}원")
            
            if losing_trades:
                avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades)
                print(f"   평균 손실: {avg_loss:,.0f}원")
        
        # 전략별 성과
        strategy_stats = results.get('strategy_stats', {})
        print(f"\n🎯 전략별 성과:")
        for strategy, stats in strategy_stats.items():
            if stats['count'] > 0:
                avg_profit = stats['profit'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
                print(f"   {strategy}:")
                print(f"      신호: {stats['count']}회 | 거래: {stats['total_trades']}회")
                print(f"      총 수익: {stats['profit']:+,.0f}원 | 평균: {avg_profit:+,.0f}원")
        
        # 최근 거래 내역
        print(f"\n📋 최근 거래 내역 (최대 5개):")
        for i, trade in enumerate(trades[-5:]):
            profit_str = f" | 수익: {trade.get('profit', 0):+,.0f}원" if trade['type'] == 'SELL' else ""
            print(f"   {i+1}. {trade['type']} @ {trade['price']:.2f} | "
                  f"{trade['strategy']} | 신뢰도: {trade['confidence']:.2f}{profit_str}")
    
    # 성과 등급
    if total_return > 30:
        grade = "A+ (매우 우수)"
    elif total_return > 15:
        grade = "A (우수)"
    elif total_return > 5:
        grade = "B+ (양호)"
    elif total_return > 0:
        grade = "B (보통)"
    elif total_return > -10:
        grade = "C (개선 필요)"
    else:
        grade = "D (부족)"
    
    print(f"\n🏆 성과 등급: {grade}")
    
    if total_return > 0:
        print("✅ 수익성 있는 전략입니다!")
    elif results.get('executed_count', 0) > 0:
        print("⚠️ 거래는 발생했으나 손실이 발생했습니다.")
    else:
        print("❌ 거래가 발생하지 않았습니다.")
    
    print("="*80)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='실제 데이터 백테스트 시스템')
    parser.add_argument('--source', choices=['binance', 'coingecko', 'ccxt', 'csv'], 
                       default='binance', help='데이터 소스 (기본: binance)')
    parser.add_argument('--symbol', default='BTCUSDT', help='거래 심볼 (기본: BTCUSDT)')
    parser.add_argument('--interval', default='1h', help='시간 간격 (기본: 1h)')
    parser.add_argument('--limit', type=int, default=1000, help='데이터 개수 (기본: 1000)')
    parser.add_argument('--days', type=int, default=30, help='CoinGecko용 일수 (기본: 30)')
    parser.add_argument('--csv-file', help='CSV 파일 경로')
    parser.add_argument('--save-csv', help='데이터를 CSV로 저장할 경로')
    parser.add_argument('--min-confidence', type=float, default=0.4, help='최소 신뢰도 (기본: 0.4)')
    
    args = parser.parse_args()
    
    print("🚀 실제 데이터 백테스트 시스템 시작")
    print(f"📊 데이터 소스: {args.source}")
    print(f"🎯 최소 신뢰도: {args.min_confidence}")
    print("="*60)
    
    try:
        # 1. 실제 데이터 로드
        loader = RealDataLoader()
        df = None
        
        if args.source == 'binance':
            df = loader.load_from_binance_api(args.symbol, args.interval, args.limit)
        elif args.source == 'coingecko':
            df = loader.load_from_coingecko_api('bitcoin', 'usd', args.days)
        elif args.source == 'ccxt':
            df = loader.load_from_ccxt('binance', args.symbol.replace('USDT', '/USDT'), args.interval, args.limit)
        elif args.source == 'csv':
            if not args.csv_file:
                print("❌ CSV 파일 경로를 지정해주세요: --csv-file 경로")
                return
            df = loader.load_from_csv(args.csv_file)
        
        if df is None or len(df) == 0:
            print("❌ 데이터 로드 실패")
            return
        
        # 2. CSV 저장 (옵션)
        if args.save_csv:
            loader.save_to_csv(df, args.save_csv)
        
        # 3. 기술적 지표 계산
        df = calculate_real_indicators(df)
        
        # 4. ML 예측
        predictions = real_ml_prediction(df)
        
        # 5. 백테스트 실행
        results = run_real_backtest(df, predictions, args.min_confidence)
        
        # 6. 결과 분석
        analyze_real_results(results)
        
        print(f"\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 실제 데이터 백테스트가 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()