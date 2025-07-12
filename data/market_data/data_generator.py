import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketDataGenerator:
    """실제 데이터 우선, 필요시 시뮬레이션 데이터 생성기"""
    def __init__(self, start_date='2020-01-01', end_date='2023-01-01'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data_dir = Path(__file__).parent
        
        # 실제 데이터 존재 여부 확인
        self.real_data_available = self._check_real_data_availability()

    def _check_real_data_availability(self) -> bool:
        """실제 데이터 파일 존재 여부 확인"""
        csv_files = list(self.data_dir.glob("*_USDT*.csv"))
        return len(csv_files) > 0
    
    def _load_real_data(self, symbol: str = None) -> pd.DataFrame:
        """실제 데이터 로드"""
        try:
            if symbol:
                # 특정 심볼 데이터 찾기
                pattern = f"{symbol.replace('/', '_')}*.csv"
                files = list(self.data_dir.glob(pattern))
                if files:
                    return pd.read_csv(files[0], index_col=0, parse_dates=True)
            
            # 가장 큰 BTC 데이터 파일 찾기
            btc_files = list(self.data_dir.glob("BTC_USDT*.csv"))
            if btc_files:
                # 가장 큰 파일 선택 (데이터가 많은 것)
                largest_file = max(btc_files, key=lambda f: f.stat().st_size)
                return pd.read_csv(largest_file, index_col=0, parse_dates=True)
            
            # 아무 USDT 데이터나 사용
            usdt_files = list(self.data_dir.glob("*_USDT*.csv"))
            if usdt_files:
                return pd.read_csv(usdt_files[0], index_col=0, parse_dates=True)
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"실제 데이터 로드 실패: {e}")
            return pd.DataFrame()

    def generate(self, n=1000, symbol: str = None):
        """데이터 생성 - 실제 데이터 우선, 필요시 시뮬레이션"""
        # 1. 실제 데이터 시도
        if self.real_data_available:
            real_data = self._load_real_data(symbol)
            if not real_data.empty:
                logger.info(f"실제 데이터 사용: {len(real_data)} 레코드")
                # 요청된 기간에 맞게 필터링
                if 'timestamp' in real_data.columns:
                    real_data.index = pd.to_datetime(real_data['timestamp'])
                mask = (real_data.index >= self.start_date) & (real_data.index <= self.end_date)
                filtered_data = real_data[mask]
                if len(filtered_data) >= n // 2:  # 최소 절반 이상의 데이터가 있으면 사용
                    return filtered_data.head(n)
        
        # 2. 시뮬레이션 데이터 생성 (최후 수단)
        logger.warning("실제 데이터 없음 - 시뮬레이션 데이터 생성")
        dates = pd.date_range(self.start_date, self.end_date, periods=n)
        
        # 환경변수에서 기본 가격 설정
        base_price = float(os.getenv('DEFAULT_CRYPTO_PRICE', '50000'))
        
        # 더 현실적인 가격 변동 생성
        returns = np.random.normal(0.0005, 0.02, n)  # 일일 평균 0.05% 수익률, 2% 변동성
        price = base_price * np.cumprod(1 + returns)
        
        volume = np.random.lognormal(mean=10, sigma=1, size=n)  # 로그정규분포 거래량
        
        df = pd.DataFrame({
            'close': price, 
            'volume': volume,
            'high': price * (1 + np.random.uniform(0.001, 0.05, n)),
            'low': price * (1 - np.random.uniform(0.001, 0.05, n)),
            'open': price * (1 + np.random.uniform(-0.02, 0.02, n))
        }, index=dates)
        
        return df

    def generate_historical_data(self, years=3, start_date=None, end_date=None, timeframe='D', symbols=None):
        if end_date is None:
            end_date = pd.Timestamp.today()
        else:
            end_date = pd.to_datetime(end_date)
        if start_date is None:
            start_date = end_date - pd.DateOffset(years=years)
        else:
            start_date = pd.to_datetime(start_date)
        dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
        n = len(dates)

        if symbols is None:
            symbols = ['SAMPLE']

        data = {}
        for symbol in symbols:
            price = np.cumsum(np.random.randn(n)) + 100
            volume = np.random.randint(100, 1000, n)
            # 반드시 'close'와 'volume' 컬럼명으로 생성!
            df = pd.DataFrame({'close': price, 'volume': volume}, index=dates)
            df['symbol'] = symbol
            data[symbol] = df

        # 단일 심볼이면 바로 DataFrame 반환
        if len(symbols) == 1:
            return data[symbols[0]]
        else:
            return pd.concat(data.values(), axis=0)        
    
# 사용 예시
if __name__ == "__main__":
    gen = MarketDataGenerator()
    df = gen.generate(500)
    print(df.head())
    df = gen.generate_historical_data(years=3)
    df.to_csv('data/market_data/sample_train_data.csv')
    print("샘플 훈련 데이터가 data/market_data/sample_train_data.csv에 저장되었습니다.") 