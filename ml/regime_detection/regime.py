import numpy as np

def detect_regime(prices, window=20):
    ma = prices.rolling(window).mean()
    regime = np.where(prices > ma, 1, -1)
    return regime 