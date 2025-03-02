# src/utils/indicators/base_indicator.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseIndicator(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the given DataFrame.
        
        Returns:
            A Pandas Series of signals aligned with the DataFrame's index.
        """
        pass
