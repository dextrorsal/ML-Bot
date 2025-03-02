# src/utils/indicators/wrapper_knn.py

import pandas as pd
from .knn import knnStrategy
from .base_indicator import BaseIndicator
from src.core.config import Config

class knn(BaseIndicator):
    def __init__(self, config_path='config/indicator_settings.json'):
        """
        Load config -> read "knn" -> pass short/long/base_neighbors 
        into knnStrategy. If no config is found, fallback to defaults.
        """
        config = Config()
        # If config is None or fails, fallback to {}
        if not config or not isinstance(config, dict):
            knn_config = {}
        else:
            knn_config = config.get("knn") or {}

        # Defensive handling for each param
        short_p = max(2, min(knn_config.get('short_period', 14), 50))
        long_p = max(short_p + 1, min(knn_config.get('long_period', 28), 100))
        base_n = max(3, min(knn_config.get('base_neighbors', 252), 252))
        bar_t  = max(10, min(knn_config.get('bar_threshold', 300), 300))
        vol_filter = knn_config.get('volatility_filter', False)

        self.model = knnStrategy(
            short_period=short_p,
            long_period=long_p,
            base_neighbors=base_n,
            bar_threshold=bar_t,
            volatility_filter=vol_filter
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Produce a full -1/0/1 signals Series, 
        by calling self.model.calculate(...) for each row i 
        from [long_period + 10, ...].
        """
        signals = pd.Series(0, index=df.index, dtype=int)

        if len(df) < self.model.long_period + 10:
            print(f"Insufficient data: need at least {self.model.long_period + 10} points")
            return signals

        for i in range(self.model.long_period + 10, len(df)):
            window_data = df.iloc[:i+1]
            try:
                signal = self.model.calculate(
                    window_data['close'].values,
                    window_data['volume'].values
                )
                if signal is not None:
                    signals.iloc[i] = signal
            except Exception as e:
                print(f"Error at index {i}: {str(e)}")
                continue

        # Forward-fill and fillna
        signals = signals.ffill().fillna(0)
        return signals
