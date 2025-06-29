class PositionManager:
    def __init__(self):
        self.positions = []

    def open_position(self, symbol, size, price, side):
        self.positions.append({'symbol': symbol, 'size': size, 'price': price, 'side': side})

    def close_position(self, symbol):
        self.positions = [p for p in self.positions if p['symbol'] != symbol]

    def get_open_positions(self):
        return self.positions 