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
        ma_up = up.rolling(max(1, window)).mean()
        ma_down = down.rolling(max(1, window)).mean()
        return 100 - (100 / (1 + ma_up / (ma_down + 1e-9))) 

    @staticmethod
    def ema(series, window):
        return series.ewm(span=max(1, window), adjust=False).mean()

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=max(1, signal), adjust=False).mean()
        return macd, macd_signal

    @staticmethod
    def bollinger_bands(series, window=20, num_std=2):
        ma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return ma, upper, lower

    @staticmethod
    def vwap(df, window=20):
        pv = df['close'] * df['volume']
        vwap = pv.rolling(window).sum() / df['volume'].rolling(window).sum()
        return vwap

    @staticmethod
    def stoch_kd(series, window_k=14, window_d=3):
        low_min = series.rolling(window_k).min()
        high_max = series.rolling(window_k).max()
        k = 100 * (series - low_min) / (high_max - low_min + 1e-9)
        d = k.rolling(window_d).mean()
        return k, d

    def add_all_indicators(self, df):
        df['ma_5'] = self.moving_average(df['close'], 5)
        df['ma_20'] = self.moving_average(df['close'], 20)
        df['rsi_14'] = self.rsi(df['close'], 14)
        df['volatility_10'] = df['close'].rolling(10).std()
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df = df.dropna()
        return df

    def add_multi_timeframe_indicators(self, df, timeframes=[('1h', 1), ('4h', 4), ('5m', 1/12)]):
        # timeframes: list of (label, 배수) ex) ('4h', 4) means 4배 window
        for label, mult in timeframes:
            base = int(20 * mult)
            df[f'ema_20_{label}'] = self.ema(df['close'], max(1, int(20 * mult)))
            df[f'ema_50_{label}'] = self.ema(df['close'], max(1, int(50 * mult)))
            df[f'ema_120_{label}'] = self.ema(df['close'], max(1, int(120 * mult)))
            macd, macd_signal = self.macd(df['close'], fast=max(1, int(12*mult)), slow=max(1, int(26*mult)), signal=max(1, int(9*mult)))
            df[f'macd_{label}'] = macd
            df[f'macd_signal_{label}'] = macd_signal
            df[f'rsi_14_{label}'] = self.rsi(df['close'], max(1, int(14 * mult)))
            ma, upper, lower = self.bollinger_bands(df['close'], window=max(1, int(20*mult)), num_std=2)
            df[f'bb_ma_{label}'] = ma
            df[f'bb_upper_{label}'] = upper
            df[f'bb_lower_{label}'] = lower
            df[f'vwap_{label}'] = self.vwap(df, window=max(1, int(20*mult)))
            k, d = self.stoch_kd(df['close'], window_k=max(1, int(14*mult)), window_d=3)
            df[f'stoch_k_{label}'] = k
            df[f'stoch_d_{label}'] = d
        return df    