import pandas as pd
import numpy as np

def add_technical_features(df):
    df = df.copy()
    df['return_1'] = df['close'].pct_change()
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(10).std()
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + ma_up / (ma_down + 1e-9)))
    df = df.dropna()
    return df 