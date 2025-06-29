class RiskManager:
    def __init__(self, max_risk_per_trade=0.01, max_drawdown=0.2):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.current_drawdown = 0.0

    def check_risk(self, capital, position_size, stop_loss_pct):
        risk_amount = position_size * stop_loss_pct
        if risk_amount > capital * self.max_risk_per_trade:
            return False
        return True

    def update_drawdown(self, peak, current):
        self.current_drawdown = (peak - current) / peak
        return self.current_drawdown < self.max_drawdown 