# src/strategy/base.py
from typing import Dict
import pandas as pd

class BaseStrategy:
    """Base class for trading strategies."""
    def __init__(self, config: Dict):
        self.config = config
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals. Must be implemented by subclass."""
        raise NotImplementedError("Subclasses should implement this method.")

