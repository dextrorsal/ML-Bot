# src/utils/indicators/bollinger_bands.py

import numpy as np
import pandas as pd
import talib

class BollingerBands:
    """
    Bollinger Bands indicator implementation.
    Calculates upper and lower bands based on price volatility.
    """
    def __init__(self, period=20, deviations=2, matype=0):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: Period for the moving average
            deviations: Number of standard deviations for bands
            matype: Moving average type (0=SMA, 1=EMA, etc.)
        """
        self.period = period
        self.deviations = deviations
        self.matype = matype
        self.prices = []
        
    def update(self, price):
        """
        Update Bollinger Bands with a new price.
        
        Args:
            price: New price to update with
            
        Returns:
            Tuple of (upper, middle, lower) band values
        """
        self.prices.append(price)
        
        # Keep only necessary data points for calculation
        if len(self.prices) > self.period * 2:
            self.prices.pop(0)
            
        if len(self.prices) >= self.period:
            # Calculate using talib for consistency with wrapper
            upper, middle, lower = talib.BBANDS(
                np.array(self.prices),
                timeperiod=self.period,
                nbdevup=self.deviations,
                nbdevdn=self.deviations,
                matype=self.matype
            )
            return upper[-1], middle[-1], lower[-1]
        
        return None, None, None
    
    def calculate(self, prices):
        """
        Calculate Bollinger Bands for a series of prices.
        
        Args:
            prices: Array or list of prices
            
        Returns:
            Tuple of arrays (upper, middle, lower)
        """
        if len(prices) < self.period:
            return None, None, None
            
        return talib.BBANDS(
            np.array(prices),
            timeperiod=self.period,
            nbdevup=self.deviations,
            nbdevdn=self.deviations,
            matype=self.matype
        )
    
    @staticmethod
    def percent_b(price, upper, lower):
        """
        Calculate %B - position of price within the bands (0-1).
        
        Args:
            price: Current price
            upper: Upper band value
            lower: Lower band value
            
        Returns:
            %B value (0-1 normally, can be outside this range)
        """
        if upper == lower:  # Avoid division by zero
            return 0.5
            
        return (price - lower) / (upper - lower)
    
    @staticmethod
    def bandwidth(upper, middle, lower):
        """
        Calculate bandwidth - shows volatility.
        
        Args:
            upper: Upper band value
            middle: Middle band value
            lower: Lower band value
            
        Returns:
            Bandwidth as percentage of middle band
        """
        if middle == 0:  # Avoid division by zero
            return 0
            
        return (upper - lower) / middle
        
def is_squeeze(bandwidths, window=20):
    """
    Detect Bollinger Band squeeze - volatility contraction.
    
    Args:
        bandwidths: Series of bandwidth values
        window: Lookback window
        
    Returns:
        True if current bandwidth is lower than minimum of previous window
    """
    if len(bandwidths) < window + 1:
        return False
        
    current = bandwidths.iloc[-1]
    historical_min = bandwidths.iloc[-window-1:-1].min()
    
    return current < historical_min

# Example usage
if __name__ == "__main__":
    # Test the Bollinger Bands indicator
    data = np.random.normal(0, 1, 100) + np.linspace(0, 5, 100)  # Random data with uptrend
    bb = BollingerBands(period=20, deviations=2)
    upper, middle, lower = bb.calculate(data)
    
    print("Upper Band:", upper[-1])
    print("Middle Band:", middle[-1])
    print("Lower Band:", lower[-1])
    print("Bandwidth:", bb.bandwidth(upper[-1], middle[-1], lower[-1]))
    print("%B:", bb.percent_b(data[-1], upper[-1], lower[-1]))