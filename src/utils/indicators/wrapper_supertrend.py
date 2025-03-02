# src/utils/indicators/wrapper_supertrend.py
import pandas as pd
from .supertrend import adaptive_supertrend  # Updated: Relative import
from .base_indicator import BaseIndicator  # Updated: Relative import
from src.core.config import Config  # Removed or Updated if needed: Corrected path, only if used


class SupertrendIndicator(BaseIndicator):
    def __init__(self, config_path='config/indicator_settings.json', **kwargs):
        """Initialize Supertrend indicator with configuration settings"""
        # Load the entire configuration
        config = Config()

        # Extract Supertrend-specific settings - safely handle None
        st_config_indicators = config.get('indicators', {}) or {}
        
        # Get supertrend config with safe fallback
        st_config = {}
        if st_config_indicators is not None:
            st_config = st_config_indicators.get('supertrend', {}) or {}
        
        # Apply configuration settings with defaults
        self.atr_length = st_config.get('atr_length', 10)
        self.factor = st_config.get('factor', 3)
        self.training_period = st_config.get('training_length', 100)
        kmeans_settings = st_config.get('kmeans_settings', {}) or {}
        self.highvol = kmeans_settings.get('high_volatility_percentile', 0.75)
        self.midvol = kmeans_settings.get('medium_volatility_percentile', 0.50)
        self.lowvol = kmeans_settings.get('low_volatility_percentile', 0.25)

        # Override with any additional parameters passed in
        self.params = {**st_config, **kwargs}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using the Supertrend indicator"""
        # Reset index on a copy to avoid float/date indexing issues.
        df_copy = df.copy()
        df_copy.reset_index(drop=True, inplace=True)

        # Calculate Supertrend values and direction
        st_values, st_direction = adaptive_supertrend(
            df_copy['high'],
            df_copy['low'],
            df_copy['close'],
            atr_length=self.params.get("atr_length", self.atr_length),
            factor=self.params.get("factor", self.factor),
            training_period=self.params.get("training_period", self.training_period),
            highvol=self.params.get("highvol", self.highvol),
            midvol=self.params.get("midvol", self.midvol),
            lowvol=self.params.get("lowvol", self.lowvol)
        )

        # Convert direction array to Pandas Series aligned with the original DataFrame
        signals = pd.Series(st_direction, index=df.index)
        return signals