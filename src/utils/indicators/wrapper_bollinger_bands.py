# src/utils/indicators/wrapper_bollinger_bands.py

import pandas as pd
import talib
from .base_indicator import BaseIndicator
from src.core.config import Config

class BollingerBandsIndicator(BaseIndicator):
    def __init__(self, config_path='config/indicator_settings.json'):
        config = Config()
        bb_config = config.get('indicators', {}).get('bollinger_bands', {})
        
        self.period = bb_config.get('period', 20)
        self.deviations = bb_config.get('deviations', 2)
        self.filter_signals_by = bb_config.get('filter_signals_by', 'None')
        self.holding_period = bb_config.get('holding_period', 3)
        self.debug = False  # Set to True for debugging output
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using Bollinger Bands.
           1 = buy (price below lower band)
           -1 = sell (price above upper band)
           0 = no signal (price between bands)
        """
        signals = pd.Series(0, index=df.index, dtype=int)
        current_signal = 0
        bar_count = 0
        
        if 'close' not in df.columns:
            return signals
            
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'], 
            timeperiod=self.period, 
            nbdevup=self.deviations, 
            nbdevdn=self.deviations,
            matype=0  # Simple Moving Average
        )
        
        for i in range(len(df)):
            if pd.isna(upper[i]) or pd.isna(lower[i]):
                signals.iloc[i] = 0
                continue
                
            # Generate signals based on Bollinger Bands
            new_signal = 0
            
            # Buy signal: price crosses below lower band
            if df['close'].iloc[i] < lower[i]:
                new_signal = 1
                
            # Sell signal: price crosses above upper band
            elif df['close'].iloc[i] > upper[i]:
                new_signal = -1
            
            # Apply filters if configured
            if not self._passes_filter(df, i):
                new_signal = 0
                
            # Apply holding period logic
            if new_signal != current_signal:
                bar_count = 0
            else:
                bar_count += 1
                
            if bar_count >= self.holding_period and new_signal != 0:
                new_signal = 0
                bar_count = 0
                
            signals.iloc[i] = new_signal
            current_signal = new_signal
            
            # Debug output for every 100th bar
            if self.debug and i % 100 == 0:
                print(f"Bar {i}: Close={df['close'].iloc[i]:.2f}, Upper={upper[i]:.2f}, "
                      f"Middle={middle[i]:.2f}, Lower={lower[i]:.2f}, Signal={new_signal}")
                
        return signals
    
    def _passes_filter(self, df, i):
        # Same filter implementation as the user's other indicators
        if self.filter_signals_by == 'None':
            return True
        if i < 10:
            return True

        if self.filter_signals_by in ('Volatility', 'Both'):
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            atr1 = talib.ATR(high, low, close, 1)
            atr10 = talib.ATR(high, low, close, 10)
            if atr1[i] <= atr10[i]:
                return False

        if self.filter_signals_by in ('Volume', 'Both'):
            if 'volume' not in df.columns:
                return True
            vol = df['volume'].values
            rsi_vol = talib.RSI(vol, 14)
            if rsi_vol[i] <= 49:
                return False

        return True