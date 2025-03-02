import numpy as np
import talib

class MACD:
    """
    MACD calculator with configurable parameters
    """
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.prices = []

    def update(self, price):
        """Update MACD calculation with new price"""
        self.prices.append(price)
        if len(self.prices) > self.slow_period * 2:
            self.prices.pop(0)
            
        if len(self.prices) >= self.slow_period:
            macd, signal, _ = talib.MACD(np.array(self.prices),
                                      fastperiod=self.fast_period,
                                      slowperiod=self.slow_period,
                                      signalperiod=self.signal_period)
            return macd[-1], signal[-1]
        return None, None