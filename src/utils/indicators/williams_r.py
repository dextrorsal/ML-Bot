# src/utils/indicators/williams_r.py

import numpy as np
import pandas as pd
import talib

class WilliamsR:
    """
    Williams %R indicator implementation.
    Momentum indicator that measures overbought/oversold levels.
    Similar to Stochastic but scaled differently (-100 to 0).
    """
    def __init__(self, period=14):
        """
        Initialize Williams %R.
        
        Args:
            period: Calculation period
        """
        self.period = period
        self.highs = []
        self.lows = []
        self.closes = []
        
    def update(self, high, low, close):
        """
        Update Williams %R with new price data.
        
        Args:
            high: Current high price
            low: Current low price
            close: Current close price
            
        Returns:
            Williams %R value
        """
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Keep only necessary data points for calculation
        if len(self.closes) > self.period * 2:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            
        if len(self.closes) >= self.period:
            # Calculate using talib for consistency with wrapper
            willr = talib.WILLR(
                np.array(self.highs),
                np.array(self.lows),
                np.array(self.closes),
                timeperiod=self.period
            )
            return willr[-1]
        
        return None
    
    def calculate(self, highs, lows, closes):
        """
        Calculate Williams %R for series of prices.
        
        Args:
            highs: Array or list of high prices
            lows: Array or list of low prices
            closes: Array or list of close prices
            
        Returns:
            Array of Williams %R values
        """
        if len(closes) < self.period:
            return None
            
        return talib.WILLR(
            np.array(highs),
            np.array(lows),
            np.array(closes),
            timeperiod=self.period
        )
    
    @staticmethod
    def is_overbought(willr, threshold=-20):
        """Check if Williams %R indicates overbought condition."""
        return willr > threshold
    
    @staticmethod
    def is_oversold(willr, threshold=-80):
        """Check if Williams %R indicates oversold condition."""
        return willr < threshold
        
    @staticmethod
    def midline_cross(prev_willr, curr_willr):
        """
        Detect Williams %R crossing the -50 midline.
        Returns: 1 for bullish cross (below to above -50), 
                -1 for bearish cross (above to below -50),
                 0 for no cross
        """
        if prev_willr < -50 and curr_willr > -50:
            return 1
        elif prev_willr > -50 and curr_willr < -50:
            return -1
        return 0
    
    @staticmethod
    def level_cross(prev_willr, curr_willr, level):
        """
        Detect Williams %R crossing a specific level.
        Returns: 1 for upward cross, -1 for downward cross, 0 for no cross
        """
        if prev_willr < level and curr_willr > level:
            return 1
        elif prev_willr > level and curr_willr < level:
            return -1
        return 0

# Example usage
if __name__ == "__main__":
    # Test the Williams %R indicator
    highs = np.array([10, 12, 15, 11, 9, 8, 10, 11, 12, 15, 14, 13, 12, 11, 10, 9, 8, 7, 8, 9, 10])
    lows = np.array([8, 7, 10, 9, 7, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 5, 6, 7, 8])
    closes = np.array([9, 11, 14, 10, 8, 7, 9, 10, 10, 13, 12, 11, 10, 9, 8, 7, 6, 6, 7, 8, 9])
    
    willr_indicator = WilliamsR(period=14)
    willr = willr_indicator.calculate(highs, lows, closes)
    
    print("Final Williams %R value:", willr[-1])
    print("Overbought:", willr_indicator.is_overbought(willr[-1]))
    print("Oversold:", willr_indicator.is_oversold(willr[-1]))