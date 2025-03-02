# src/strategy/segmented.py
from typing import Dict, List, Optional
import pandas as pd
from .base import BaseStrategy

class SegmentedStrategy(BaseStrategy):
    """
    Strategy that adapts to different market regimes (trending, ranging, volatile)
    by using different sets of indicators appropriate for each regime.
    """
    
    def __init__(self, config: Dict, trend_indicators: List, range_indicators: List, 
                volatile_indicators: List, 
                trend_weights: Optional[List[float]] = None,
                range_weights: Optional[List[float]] = None,
                volatile_weights: Optional[List[float]] = None) -> None:
        """
        Initialize the segmented strategy with different indicator sets for each market regime.
        
        Parameters:
            config: Configuration dictionary with strategy parameters
            trend_indicators: List of indicators optimized for trending markets
            range_indicators: List of indicators optimized for ranging markets
            volatile_indicators: List of indicators optimized for volatile markets
            trend_weights: Optional weights for trend indicators (default: equal weights)
            range_weights: Optional weights for range indicators (default: equal weights)
            volatile_weights: Optional weights for volatile indicators (default: equal weights)
        """
        super().__init__(config)
        self.trend_indicators = trend_indicators
        self.range_indicators = range_indicators
        self.volatile_indicators = volatile_indicators
        
        # Initialize weights (with optional custom values)
        self.trend_weights = trend_weights if trend_weights is not None else [1] * len(trend_indicators)
        self.range_weights = range_weights if range_weights is not None else [1] * len(range_indicators)
        self.volatile_weights = volatile_weights if volatile_weights is not None else [1] * len(volatile_indicators)
        
        # Validate weights match indicator counts
        if len(self.trend_weights) != len(trend_indicators):
            raise ValueError("Length of trend_weights must match number of trend_indicators")
        if len(self.range_weights) != len(range_indicators):
            raise ValueError("Length of range_weights must match number of range_indicators")
        if len(self.volatile_weights) != len(volatile_indicators):
            raise ValueError("Length of volatile_weights must match number of volatile_indicators")
            
        # Configuration parameters
        self.threshold = self.config.get("consensus_threshold", 0)
        self.regime_window = self.config.get("regime_window", 20)
        
        # Regime detection parameters (configurable)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.03)
        self.trend_threshold = self.config.get("trend_threshold", 0.5)
        
        # Track detected regimes for analysis
        self.detected_regimes = []
        
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime: trending, ranging, or volatile.
        
        This uses a combination of directional movement strength and volatility
        to classify the current market condition.
        
        Returns:
            str: "trend", "range", or "volatile"
        """
        if len(df) < self.regime_window:
            return "trend"  # Default to trend when insufficient data
            
        # Calculate key metrics over the window
        window = df.iloc[-self.regime_window:]
        
        # Directional movement - ADX-inspired metric
        price_change = abs(window['close'].iloc[-1] - window['close'].iloc[0])
        price_range = window['high'].max() - window['low'].min()
        directional_strength = price_change / price_range if price_range > 0 else 0
        
        # Volatility measure (standard deviation of returns)
        daily_returns = window['close'].pct_change().dropna()
        volatility = daily_returns.std()
        
        # Detect regime based on metrics
        if volatility > self.volatility_threshold:  # High volatility threshold
            return "volatile"
        elif directional_strength > self.trend_threshold:  # Strong directional movement
            return "trend"
        else:
            return "range"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on detected market regime.
        
        The strategy automatically detects the current market regime and uses
        the appropriate set of indicators for signal generation.
        
        Parameters:
            df: DataFrame with OHLCV market data
            
        Returns:
            pd.Series: Trading signals (1=long, -1=short, 0=neutral)
        """
        # Detect current market regime
        regime = self._detect_market_regime(df)
        self.detected_regimes.append(regime)  # Store for analysis
        
        # Select appropriate indicators and weights based on regime
        if regime == "trend":
            indicators = self.trend_indicators
            weights = self.trend_weights
        elif regime == "range":
            indicators = self.range_indicators
            weights = self.range_weights
        else:  # volatile
            indicators = self.volatile_indicators
            weights = self.volatile_weights
            
        # Generate signals using selected indicators
        weighted_signals = []
        for indicator, weight in zip(indicators, weights):
            signals = indicator.generate_signals(df)
            weighted_signals.append(signals * weight)
            
        # If no signals are produced, default to neutral
        if not weighted_signals:
            return pd.Series(0, index=df.index)
            
        # Combine signals
        combined_df = pd.concat(weighted_signals, axis=1)
        summed_signals = combined_df.sum(axis=1)
        
        # Final signals based on threshold
        final_signals = summed_signals.apply(
            lambda x: 1 if x > self.threshold else (-1 if x < -self.threshold else 0)
        )
        
        return final_signals
        
    def get_current_regime(self) -> str:
        """
        Get the most recently detected market regime.
        
        Returns:
            str: The current market regime ("trend", "range", or "volatile")
        """
        if not self.detected_regimes:
            return "unknown"
        return self.detected_regimes[-1]
    
    def get_regime_statistics(self) -> Dict:
        """
        Get statistics about detected regimes during strategy execution.
        
        Returns:
            Dict: Statistics about regime distributions and transitions
        """
        if not self.detected_regimes:
            return {"count": 0}
            
        # Count occurrences of each regime
        trend_count = self.detected_regimes.count("trend")
        range_count = self.detected_regimes.count("range")
        volatile_count = self.detected_regimes.count("volatile")
        total = len(self.detected_regimes)
        
        # Count regime transitions
        transitions = 0
        for i in range(1, len(self.detected_regimes)):
            if self.detected_regimes[i] != self.detected_regimes[i-1]:
                transitions += 1
                
        return {
            "count": total,
            "trend_percentage": (trend_count / total) * 100 if total > 0 else 0,
            "range_percentage": (range_count / total) * 100 if total > 0 else 0,
            "volatile_percentage": (volatile_count / total) * 100 if total > 0 else 0,
            "transitions": transitions,
            "avg_regime_duration": total / (transitions + 1) if transitions > 0 else total
        }