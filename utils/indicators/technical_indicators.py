import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def moving_average(series, window):
        return series.rolling(window).mean()

    @staticmethod
    def rsi(series, window=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window).mean()
        ma_down = down.rolling(window).mean()
        return 100 - (100 / (1 + ma_up / (ma_down + 1e-9))) 
    
    def add_all_indicators(self, df):
        df['ma_5'] = self.moving_average(df['close'], 5)
        df['ma_20'] = self.moving_average(df['close'], 20)
        df['rsi_14'] = self.rsi(df['close'], 14)
        df['volatility_10'] = df['close'].rolling(10).std()
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df = df.dropna()
        return df    