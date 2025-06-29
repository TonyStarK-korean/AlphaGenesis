import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketDataGenerator:
    """랜덤 또는 시뮬레이션 기반 시장 데이터 생성기"""
    def __init__(self, start_date='2020-01-01', end_date='2023-01-01'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

    def generate(self, n=1000):
        dates = pd.date_range(self.start_date, self.end_date, periods=n)
        price = np.cumsum(np.random.randn(n)) + 100
        volume = np.random.randint(100, 1000, n)
        df = pd.DataFrame({'close': price, 'volume': volume}, index=dates)
        return df

    def generate_historical_data(self, years=3, freq='D'):
        """지정한 연수(years)만큼의 랜덤 OHLCV 데이터 생성"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=years)
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n = len(dates)
        price = np.cumsum(np.random.randn(n)) + 100
        volume = np.random.randint(100, 1000, n)
        df = pd.DataFrame({'close': price, 'volume': volume}, index=dates)
        return df    

# 사용 예시
if __name__ == "__main__":
    gen = MarketDataGenerator()
    df = gen.generate(500)
    print(df.head()) 