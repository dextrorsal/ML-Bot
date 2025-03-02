# src/utils/indicators/stochastic.py

import numpy as np
import pandas as pd
import talib

class Stochastic:
    """
    Stochastic Oscillator indicator implementation.
    Measures momentum by comparing closing price to price range over a period.
    """
    def __init__(self, k_period=14, d_period=3, slowing=3):
        """
        Initialize Stochastic Oscillator.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D calculation (moving average of %K)
            slowing: Slowing period (moving average of raw %K)
        """
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        self.highs = []
        self.lows = []
        self.closes = []
        
    def update(self, high, low, close):
        """
        Update Stochastic Oscillator with new price data.
        
        Args:
            high: Current high price
            low: Current low price
            close: Current close price
            
        Returns:
            Tuple of (k, d) values
        """
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Keep only necessary data points for calculation
        if len(self.closes) > self.k_period + self.d_period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            
        if len(self.closes) >= self.k_period:
            # Calculate using talib for consistency with wrapper
            k, d = talib.STOCH(
                np.array(self.highs),
                np.array(self.lows),
                np.array(self.closes),
                fastk_period=self.k_period,
                slowk_period=self.slowing,
                slowk_matype=0,
                slowd_period=self.d_period,
                slowd_matype=0
            )
            return k[-1], d[-1]
        
        return None, None
    
    def calculate(self, highs, lows, closes):
        """
        Calculate Stochastic Oscillator for series of prices.
        
        Args:
            highs: Array or list of high prices
            lows: Array or list of low prices
            closes: Array or list of close prices
            
        Returns:
            Tuple of arrays (k, d)
        """
        if len(closes) < self.k_period:
            return None, None
            
        return talib.STOCH(
            np.array(highs),
            np.array(lows),
            np.array(closes),
            fastk_period=self.k_period,
            slowk_period=self.slowing,
            slowk_matype=0,
            slowd_period=self.d_period,
            slowd_matype=0
        )
    
    @staticmethod
    def is_overbought(k, d, threshold=80):
        """Check if stochastic indicates overbought condition."""
        return k > threshold and d > threshold
    
    @staticmethod
    def is_oversold(k, d, threshold=20):
        """Check if stochastic indicates oversold condition."""
        return k < threshold and d < threshold
        
    @staticmethod
    def has_bullish_crossover(prev_k, prev_d, curr_k, curr_d):
        """Check for bullish crossover (%K crosses above %D)."""
        return prev_k < prev_d and curr_k > curr_d
        
    @staticmethod
    def has_bearish_crossover(prev_k, prev_d, curr_k, curr_d):
        """Check for bearish crossover (%K crosses below %D)."""
        return prev_k > prev_d and curr_k < curr_d

# Example usage
if __name__ == "__main__":
    # Test the Stochastic Oscillator
    highs = np.array([10, 12, 15, 11, 9, 8, 10, 11, 12, 15, 14, 13, 12, 11, 10])
    lows = np.array([8, 7, 10, 9, 7, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7])
    closes = np.array([9, 11, 14, 10, 8, 7, 9, 10, 10, 13, 12, 11, 10, 9, 8])
    
    stochastic = Stochastic(k_period=5, d_period=3, slowing=3)
    k, d = stochastic.calculate(highs, lows, closes)
    
    print("Final %K value:", k[-1])
    print("Final %D value:", d[-1])
    print("Overbought:", stochastic.is_overbought(k[-1], d[-1]))
    print("Oversold:", stochastic.is_oversold(k[-1], d[-1]))