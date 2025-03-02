import pandas as pd
import talib
from .base_indicator import BaseIndicator
from src.core.config import Config

class RsiIndicator(BaseIndicator):
    def __init__(self, config_path='config/indicator_settings.json'):
        config = Config()
        rsi_config = config.get('indicators', {}).get('rsi', {})
        
        self.period = rsi_config.get('period', 14)
        self.overbought = rsi_config.get('overbought', 70)
        self.oversold = rsi_config.get('oversold', 30)
        self.filter_signals_by = rsi_config.get('filter_signals_by', 'None')
        self.holding_period = rsi_config.get('holding_period', 5)
        self.model = RSI(self.period, self.overbought, self.oversold)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        current_signal = 0
        bar_count = 0

        if 'close' not in df.columns:
            return signals

        rsi_values = talib.RSI(df['close'], timeperiod=self.period)

        for i in range(len(df)):
            if pd.isna(rsi_values[i]):
                signals.iloc[i] = 0
                continue

            new_signal = 0
            if rsi_values[i] > self.overbought:
                new_signal = -1
            elif rsi_values[i] < self.oversold:
                new_signal = 1

            if not self._passes_filter(df, i):
                new_signal = 0

            if new_signal != current_signal:
                bar_count = 0
            else:
                bar_count += 1
                
            if bar_count >= self.holding_period and new_signal != 0:
                new_signal = 0
                bar_count = 0

            signals.iloc[i] = new_signal
            current_signal = new_signal

            if self.debug and i % 100 == 0:
                print(f"Bar {i}: RSI={rsi_values[i]:.2f}, Signal={new_signal}")

        return signals

    def _passes_filter(self, df, i):
        # Same filter logic as your existing indicator
        if self.filter_signals_by == 'None':
            return True
        if i < 10:
            return True

        # Volatility filter
        if self.filter_signals_by in ('Volatility', 'Both'):
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            atr1 = talib.ATR(high, low, close, 1)
            atr10 = talib.ATR(high, low, close, 10)
            if atr1[i] <= atr10[i]:
                return False

        # Volume filter
        if self.filter_signals_by in ('Volume', 'Both'):
            if 'volume' not in df.columns:
                return True
            vol = df['volume'].values
            rsi_vol = talib.RSI(vol, 14)
            if rsi_vol[i] <= 49:
                return False

        return True