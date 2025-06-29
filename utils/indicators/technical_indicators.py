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