"""
실제 백테스트 데이터 다운로드 및 관리 시스템
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import time

logger = logging.getLogger(__name__)

class DataManager:
    """실제 시장 데이터 다운로드 및 관리"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.exchange = None
        self.symbol_cache = {}
        self.price_cache = {}
        self.cache_ttl = 300  # 5분 캐시
        
        # 데이터 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/raw", exist_ok=True)
        os.makedirs(f"{data_dir}/processed", exist_ok=True)
        
        self.init_exchange()
    
    def init_exchange(self):
        """바이낸스 거래소 초기화"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
                'timeout': 30000,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 선물 거래
                }
            })
            logger.info("바이낸스 거래소 초기화 완료")
        except Exception as e:
            logger.error(f"바이낸스 거래소 초기화 실패: {e}")
            # API 키가 없는 경우 샌드박스 모드로 초기화
            self.exchange = ccxt.binance({
                'sandbox': True,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
    
    async def download_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        실제 히스토리컬 데이터 다운로드
        
        Args:
            symbol: 거래 심볼 (예: 'BTC/USDT')
            timeframe: 시간프레임 ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: 시작 날짜
            end_date: 종료 날짜
            limit: 최대 캔들 수
            
        Returns:
            DataFrame: OHLCV 데이터
        """
        try:
            # 캐시 확인
            cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
            if cache_key in self.price_cache:
                cache_time, data = self.price_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    logger.info(f"캐시된 데이터 반환: {symbol}")
                    return data
            
            # 심볼 정규화
            if '/' not in symbol:
                symbol = symbol.replace('USDT', '/USDT')
            
            # 기본 날짜 설정
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # 타임스탬프 변환
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
            logger.info(f"데이터 다운로드 시작: {symbol} ({timeframe})")
            
            all_candles = []
            current_since = since
            
            while current_since < until:
                try:
                    # 바이낸스 API 호출
                    ohlcv = await self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=current_since, 
                        limit=limit
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_candles.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"데이터 다운로드 오류: {e}")
                    break
            
            # DataFrame 생성
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 중복 제거 및 정렬
            df = df.drop_duplicates().sort_index()
            
            # 캐시 저장
            self.price_cache[cache_key] = (time.time(), df)
            
            # 파일로 저장
            filename = f"{self.data_dir}/raw/{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            df.to_csv(filename)
            
            logger.info(f"데이터 다운로드 완료: {len(df)} 캔들")
            return df
            
        except Exception as e:
            logger.error(f"데이터 다운로드 실패: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols_data(
        self, 
        symbols: List[str], 
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 심볼의 데이터 동시 다운로드
        
        Args:
            symbols: 심볼 리스트
            timeframe: 시간프레임
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            Dict[str, DataFrame]: 심볼별 데이터
        """
        async def download_all():
            tasks = []
            for symbol in symbols:
                task = self.download_historical_data(symbol, timeframe, start_date, end_date)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data_dict = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"심볼 {symbol} 다운로드 실패: {result}")
                    data_dict[symbol] = pd.DataFrame()
                else:
                    data_dict[symbol] = result
            
            return data_dict
        
        return asyncio.run(download_all())
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가
        
        Args:
            df: OHLCV 데이터
            
        Returns:
            DataFrame: 기술적 지표가 추가된 데이터
        """
        try:
            # 이동평균선
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()
            
            # 지수이동평균선
            df['EMA_12'] = df['close'].ewm(span=12).mean()
            df['EMA_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # 볼린저 밴드
            df['BB_Middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # 거래량 지표
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
            
            # 가격 변화율
            df['Price_Change'] = df['close'].pct_change()
            df['Price_Change_5'] = df['close'].pct_change(5)
            df['Price_Change_10'] = df['close'].pct_change(10)
            
            # 변동성
            df['Volatility'] = df['Price_Change'].rolling(window=20).std()
            
            logger.info("기술적 지표 추가 완료")
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}")
            return df
    
    def save_processed_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        처리된 데이터 저장
        
        Args:
            df: 처리된 데이터
            symbol: 심볼
            timeframe: 시간프레임
        """
        try:
            filename = f"{self.data_dir}/processed/{symbol.replace('/', '_')}_{timeframe}_processed.csv"
            df.to_csv(filename)
            logger.info(f"처리된 데이터 저장: {filename}")
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}")
    
    def load_processed_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        처리된 데이터 로드
        
        Args:
            symbol: 심볼
            timeframe: 시간프레임
            
        Returns:
            DataFrame: 처리된 데이터
        """
        try:
            filename = f"{self.data_dir}/processed/{symbol.replace('/', '_')}_{timeframe}_processed.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                logger.info(f"처리된 데이터 로드: {filename}")
                return df
            else:
                logger.warning(f"처리된 데이터 파일이 없음: {filename}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        데이터 품질 보고서 생성
        
        Args:
            df: 데이터프레임
            
        Returns:
            Dict: 데이터 품질 보고서
        """
        try:
            report = {
                'total_records': len(df),
                'date_range': {
                    'start': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index.max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict(),
                'basic_stats': df.describe().to_dict(),
                'duplicates': df.duplicated().sum(),
                'completeness': (1 - df.isnull().sum() / len(df)).to_dict()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"데이터 품질 보고서 생성 실패: {e}")
            return {}

class MLDataProcessor:
    """머신러닝을 위한 데이터 전처리기"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_column = 'target'
        self.scaler = None
    
    def prepare_features(self, df: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame:
        """
        ML 모델용 피처 준비
        
        Args:
            df: 원본 데이터
            prediction_horizon: 예측 지평선 (몇 기간 후 예측할지)
            
        Returns:
            DataFrame: 피처가 준비된 데이터
        """
        try:
            ml_df = df.copy()
            
            # 라벨 생성 (다음 기간 가격 상승/하락)
            ml_df['future_return'] = ml_df['close'].shift(-prediction_horizon) / ml_df['close'] - 1
            ml_df['target'] = (ml_df['future_return'] > 0).astype(int)
            
            # 기본 피처들
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_20', 'SMA_50', 'SMA_200',
                'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Middle', 'BB_Lower',
                'ATR', 'Volume_Ratio',
                'Price_Change', 'Price_Change_5', 'Price_Change_10',
                'Volatility'
            ]
            
            # 추가 피처 생성
            ml_df['Price_Position'] = (ml_df['close'] - ml_df['BB_Lower']) / (ml_df['BB_Upper'] - ml_df['BB_Lower'])
            ml_df['RSI_Oversold'] = (ml_df['RSI'] < 30).astype(int)
            ml_df['RSI_Overbought'] = (ml_df['RSI'] > 70).astype(int)
            ml_df['Volume_Spike'] = (ml_df['Volume_Ratio'] > 2).astype(int)
            
            # 시간 기반 피처
            ml_df['Hour'] = ml_df.index.hour
            ml_df['DayOfWeek'] = ml_df.index.dayofweek
            ml_df['Month'] = ml_df.index.month
            
            # 상대적 위치 피처
            ml_df['High_Low_Ratio'] = ml_df['high'] / ml_df['low']
            ml_df['Close_High_Ratio'] = ml_df['close'] / ml_df['high']
            ml_df['Close_Low_Ratio'] = ml_df['close'] / ml_df['low']
            
            features.extend([
                'Price_Position', 'RSI_Oversold', 'RSI_Overbought', 'Volume_Spike',
                'Hour', 'DayOfWeek', 'Month',
                'High_Low_Ratio', 'Close_High_Ratio', 'Close_Low_Ratio'
            ])
            
            self.feature_columns = features
            
            # NaN 값 처리
            ml_df = ml_df.dropna()
            
            logger.info(f"ML 피처 준비 완료: {len(features)} 피처, {len(ml_df)} 샘플")
            return ml_df
            
        except Exception as e:
            logger.error(f"ML 피처 준비 실패: {e}")
            return pd.DataFrame()
    
    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        훈련/테스트 데이터 분할
        
        Args:
            df: 데이터프레임
            test_size: 테스트 데이터 비율
            
        Returns:
            Tuple: (훈련 데이터, 테스트 데이터)
        """
        try:
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            logger.info(f"데이터 분할: 훈련 {len(train_df)}, 테스트 {len(test_df)}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"데이터 분할 실패: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        피처 중요도 계산
        
        Args:
            model: 훈련된 모델
            feature_names: 피처 이름들
            
        Returns:
            Dict: 피처 중요도
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return dict(zip(feature_names, importance))
            else:
                return {}
        except Exception as e:
            logger.error(f"피처 중요도 계산 실패: {e}")
            return {}