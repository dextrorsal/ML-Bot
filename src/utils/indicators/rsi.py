import numpy as np
import talib

class RSI:
    """
    Relative Strength Index (RSI) calculator with configurable parameters
    """
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.prices = []

    def update(self, price):
        """Update RSI calculation with new price"""
        self.prices.append(price)
        if len(self.prices) > self.period * 2:  # Keep sufficient history
            self.prices.pop(0)
            
        if len(self.prices) >= self.period:
            return talib.RSI(np.array(self.prices), timeperiod=self.period)[-1]
        return None