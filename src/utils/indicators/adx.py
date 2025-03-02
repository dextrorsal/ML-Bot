# src/utils/indicators/adx.py

import numpy as np
import pandas as pd
import talib

class ADX:
    """
    Average Directional Index (ADX) indicator implementation.
    Measures trend strength without regard to trend direction.
    """
    def __init__(self, period=14):
        """
        Initialize ADX.
        
        Args:
            period: Calculation period
        """
        self.period = period
        self.highs = []
        self.lows = []
        self.closes = []
        
    def update(self, high, low, close):
        """
        Update ADX with new price data.
        
        Args:
            high: Current high price
            low: Current low price
            close: Current close price
            
        Returns:
            Tuple of (ADX, +DI, -DI) values
        """
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Keep only necessary data points for calculation
        if len(self.closes) > self.period * 3:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            
        if len(self.closes) >= self.period:
            # Calculate using talib for consistency with wrapper
            adx, plus_di, minus_di = talib.ADXR(
                np.array(self.highs),
                np.array(self.lows),
                np.array(self.closes),
                timeperiod=self.period
            )
            return adx[-1], plus_di[-1], minus_di[-1]
        
        return None, None, None
    
    def calculate(self, highs, lows, closes):
        """
        Calculate ADX for series of prices.
        
        Args:
            highs: Array or list of high prices
            lows: Array or list of low prices
            closes: Array or list of close prices
            
        Returns:
            Tuple of arrays (adx, +di, -di)
        """
        if len(closes) < self.period:
            return None, None, None
            
        return talib.ADX(
            np.array(highs),
            np.array(lows),
            np.array(closes),
            timeperiod=self.period
        ), talib.PLUS_DI(
            np.array(highs),
            np.array(lows),
            np.array(closes),
            timeperiod=self.period
        ), talib.MINUS_DI(
            np.array(highs),
            np.array(lows),
            np.array(closes),
            timeperiod=self.period
        )
    
    @staticmethod
    def is_strong_trend(adx, threshold=25):
        """Check if ADX indicates a strong trend (above threshold)."""
        return adx > threshold
    
    @staticmethod
    def is_weak_trend(adx, threshold=20):
        """Check if ADX indicates a weak trend (below threshold)."""
        return adx < threshold
        
    @staticmethod
    def trend_direction(plus_di, minus_di):
        """
        Determine trend direction based on +DI and -DI.
        Returns: 1 for bullish (+DI > -DI), 
                -1 for bearish (+DI < -DI),
                 0 for no trend (+DI == -DI)
        """
        if plus_di > minus_di:
            return 1
        elif plus_di < minus_di:
            return -1
        return 0
    
    @staticmethod
    def di_crossover(prev_plus_di, prev_minus_di, curr_plus_di, curr_minus_di):
        """
        Detect DI crossover.
        Returns: 1 for bullish cross (+DI crosses above -DI), 
                -1 for bearish cross (+DI crosses below -DI),
                 0 for no cross
        """
        if prev_plus_di < prev_minus_di and curr_plus_di > curr_minus_di:
            return 1
        elif prev_plus_di > prev_minus_di and curr_plus_di < curr_minus_di:
            return -1
        return 0

# Example usage
if __name__ == "__main__":
    # Test the ADX indicator
    highs = np.array([10, 12, 15, 15, 15, 15, 16, 17, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11])
    lows = np.array([8, 7, 10, 10, 10, 10, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7])
    closes = np.array([9, 11, 14, 14, 14, 14, 15, 16, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10])
    
    adx_indicator = ADX(period=14)
    adx, plus_di, minus_di = adx_indicator.calculate(highs, lows, closes)
    
    print("Final ADX value:", adx[-1])
    print("Final +DI value:", plus_di[-1])
    print("Final -DI value:", minus_di[-1])
    print("Is strong trend:", adx_indicator.is_strong_trend(adx[-1]))
    print("Trend direction:", adx_indicator.trend_direction(plus_di[-1], minus_di[-1]))