import pandas as pd
import talib
from .base_indicator import BaseIndicator
from src.core.config import Config

class MacdIndicator(BaseIndicator):
    def __init__(self, config_path='config/indicator_settings.json'):
        config = Config()
        macd_config = config.get('indicators', {}).get('macd', {})
        
        self.fast_period = macd_config.get('fast_period', 12)
        self.slow_period = macd_config.get('slow_period', 26)
        self.signal_period = macd_config.get('signal_period', 9)
        self.filter_signals_by = macd_config.get('filter_signals_by', 'None')
        self.holding_period = macd_config.get('holding_period', 5)
        self.model = MACD(self.fast_period, self.slow_period, self.signal_period)
        self.prev_macd = None
        self.prev_signal = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        current_signal = 0
        bar_count = 0

        macd, signal, _ = talib.MACD(df['close'],
                                   fastperiod=self.fast_period,
                                   slowperiod=self.slow_period,
                                   signalperiod=self.signal_period)

        for i in range(len(df)):
            if pd.isna(macd[i]) or pd.isna(signal[i]):
                signals.iloc[i] = 0
                continue

            new_signal = 0
            if self.prev_macd is not None and self.prev_signal is not None:
                if self.prev_macd < self.prev_signal and macd[i] > signal[i]:
                    new_signal = 1
                elif self.prev_macd > self.prev_signal and macd[i] < signal[i]:
                    new_signal = -1

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
            self.prev_macd = macd[i]
            self.prev_signal = signal[i]

            if self.debug and i % 100 == 0:
                print(f"Bar {i}: MACD={macd[i]:.2f}, Signal={signal[i]:.2f}, Decision={new_signal}")

        return signals

    def _passes_filter(self, df, i):
        # Same filter implementation as RSI
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