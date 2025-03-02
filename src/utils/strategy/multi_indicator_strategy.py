# src/strategy/multi_indicator_strategy.py
from typing import Dict, List, Optional
import pandas as pd
from .base import BaseStrategy

class MultiIndicatorStrategy(BaseStrategy):
    """Strategy implementation using multiple indicators with weights."""
    
    def __init__(self, config: Dict, indicators: List, weights: Optional[List[float]] = None) -> None:
        """
        :param config: Configuration dictionary.
        :param indicators: List of indicator objects. Each must have a method generate_signals(df) -> pd.Series.
        :param weights: Optional list of weights for each indicator. 
                        If not provided and exactly 4 indicators are used, defaults to [2, 2, 1, 1].
                        Otherwise, equal weights are used.
        """
        super().__init__(config)
        self.indicators = indicators
        
        if weights is not None:
            if len(weights) != len(indicators):
                raise ValueError("Length of weights must match number of indicators.")
            self.weights = weights
        else:
            # If exactly 4 indicators, default to: Lorentzian:2, Supertrend:2, Logistic:1, KNN:1
            if len(indicators) == 4:
                self.weights = [2, 2, 1, 1]
            else:
                self.weights = [1] * len(indicators)
        
        # Optional consensus threshold (default 0)
        self.threshold = self.config.get("consensus_threshold", 0)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the weighted consensus of multiple indicators.
        For each bar, each indicator's signal is multiplied by its weight.
        The weighted signals are summed, and if the sum exceeds the threshold, the final signal is 1 (LONG);
        if below the negative threshold, -1 (SHORT); otherwise, 0.
        """
        weighted_signals_list = []
        for indicator, weight in zip(self.indicators, self.weights):
            # Each indicator returns a pd.Series of signals aligned with df.
            signals = indicator.generate_signals(df)
            weighted_signals_list.append(signals * weight)
        
        # If no signals are produced, default to a neutral series.
        if not weighted_signals_list:
            return pd.Series(0, index=df.index)
        
        # Combine weighted signals into a DataFrame.
        combined_signals_df = pd.concat(weighted_signals_list, axis=1)
        # Sum across columns for each time point.
        summed_signals = combined_signals_df.sum(axis=1)
        
        # Final signal based on the threshold.
        final_signals = summed_signals.apply(
            lambda x: 1 if x > self.threshold else (-1 if x < -self.threshold else 0)
        )
        return final_signals


