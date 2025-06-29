import numpy as np

class Calculators:
    @staticmethod
    def annualized_return(returns, periods_per_year=252):
        compounded_growth = np.prod(1 + returns)
        n_periods = len(returns)
        return compounded_growth ** (periods_per_year / n_periods) - 1

    @staticmethod
    def max_drawdown(cumulative_returns):
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()