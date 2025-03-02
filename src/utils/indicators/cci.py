# src/utils/indicators/cci.py

import numpy as np
import pandas as pd
import talib

class CCI:
    """
    Commodity Channel Index (CCI) indicator implementation.
    Measures current price level relative to an average price level over a period.
    """
    def __init__(self, period=20):
        """
        Initialize CCI.
        
        Args:
            period: Calculation period
        """
        self.period = period
        self.highs = []
        self.lows = []
        self.closes = []
        
    def update(self, high, low, close):
        """
        Update CCI with new price data.
        
        Args:
            high: Current high price
            low: Current low price
            close: Current close price
            
        Returns:
            CCI value
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
            cci = talib.CCI(
                np.array(self.highs),
                np.array(self.lows),
                np.array(self.closes),
                timeperiod=self.period
            )
            return cci[-1]
        
        return None
    
    def calculate(self, highs, lows, closes):
        """
        Calculate CCI for series of prices.
        
        Args:
            highs: Array or list of high prices
            lows: Array or list of low prices
            closes: Array or list of close prices
            
        Returns:
            Array of CCI values
        """
        if len(closes) < self.period:
            return None
            
        return talib.CCI(
            np.array(highs),
            np.array(lows),
            np.array(closes),
            timeperiod=self.period
        )
    
    @staticmethod
    def is_overbought(cci, threshold=100):
        """Check if CCI indicates overbought condition."""
        return cci > threshold
    
    @staticmethod
    def is_oversold(cci, threshold=-100):
        """Check if CCI indicates oversold condition."""
        return cci < threshold
        
    @staticmethod
    def zero_line_cross(prev_cci, curr_cci):
        """
        Detect CCI crossing the zero line.
        Returns: 1 for bullish cross (negative to positive), 
                -1 for bearish cross (positive to negative),
                 0 for no cross
        """
        if prev_cci < 0 and curr_cci > 0:
            return 1
        elif prev_cci > 0 and curr_cci < 0:
            return -1
        return 0

# Example usage
if __name__ == "__main__":
    # Test the CCI indicator
    highs = np.array([10, 12, 15, 11, 9, 8, 10, 11, 12, 15, 14, 13, 12, 11, 10, 9, 8, 7, 8, 9, 10, 11])
    lows = np.array([8, 7, 10, 9, 7, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 5, 6, 7, 8, 9])
    closes = np.array([9, 11, 14, 10, 8, 7, 9, 10, 10, 13, 12, 11, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10])
    
    cci_indicator = CCI(period=14)
    cci = cci_indicator.calculate(highs, lows, closes)
    
    print("Final CCI value:", cci[-1])
    print("Overbought:", cci_indicator.is_overbought(cci[-1]))
    print("Oversold:", cci_indicator.is_oversold(cci[-1]))