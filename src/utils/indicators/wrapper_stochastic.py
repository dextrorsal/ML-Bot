# src/utils/indicators/wrapper_stochastic.py

import pandas as pd
import talib
from .base_indicator import BaseIndicator
from src.core.config import Config

class StochasticIndicator(BaseIndicator):
    def __init__(self, config_path='config/indicator_settings.json'):
        config = Config()
        stoch_config = config.get('indicators', {}).get('stochastic', {})
        
        self.k_period = stoch_config.get('k_period', 14)
        self.d_period = stoch_config.get('d_period', 3)
        self.smoothing = stoch_config.get('smoothing', 3)
        self.overbought = stoch_config.get('overbought', 80)
        self.oversold = stoch_config.get('oversold', 20)
        self.filter_signals_by = stoch_config.get('filter_signals_by', 'None')
        self.holding_period = stoch_config.get('holding_period', 3)
        self.debug = False  # Set to True for debugging output
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using Stochastic Oscillator.
           1 = buy (oversold with bullish crossover)
           -1 = sell (overbought with bearish crossover)
           0 = no signal
        """
        signals = pd.Series(0, index=df.index, dtype=int)
        current_signal = 0
        bar_count = 0
        
        # Required columns check
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return signals
            
        # Calculate Stochastic Oscillator
        k, d = talib.STOCH(
            df['high'], 
            df['low'], 
            df['close'],
            fastk_period=self.k_period,
            slowk_period=self.smoothing,
            slowk_matype=0,
            slowd_period=self.d_period,
            slowd_matype=0
        )
        
        # Previous values for crossover detection
        prev_k, prev_d = None, None
        
        for i in range(len(df)):
            if pd.isna(k[i]) or pd.isna(d[i]):
                signals.iloc[i] = 0
                continue
                
            # Current values
            curr_k, curr_d = k[i], d[i]
            
            # Generate signals based on Stochastic
            new_signal = 0
            
            # Only generate signals if we have previous values
            if prev_k is not None and prev_d is not None:
                # Buy signal: Oversold with bullish crossover
                if curr_k < self.oversold and curr_d < self.oversold and prev_k < prev_d and curr_k > curr_d:
                    new_signal = 1
                    
                # Sell signal: Overbought with bearish crossover
                elif curr_k > self.overbought and curr_d > self.overbought and prev_k > prev_d and curr_k < curr_d:
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
            
            # Update previous values
            prev_k, prev_d = curr_k, curr_d
            
            # Debug output for every 100th bar
            if self.debug and i % 100 == 0:
                print(f"Bar {i}: %K={curr_k:.2f}, %D={curr_d:.2f}, Signal={new_signal}")
                
        return signals
    
    def _passes_filter(self, df, i):
        # Same filter implementation as other indicators
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