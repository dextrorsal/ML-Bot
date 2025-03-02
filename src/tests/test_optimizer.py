import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtesting.optimizer import StrategyOptimizer
from src.backtesting.backtester import Backtester, BacktestResult
from src.utils.indicators.base_indicator import BaseIndicator

# ------------------------------------------------------------------------------
# Mock indicator class with configurable parameters for testing
# ------------------------------------------------------------------------------
class ParameterizedIndicator(BaseIndicator):
    def __init__(self, param_a=10, param_b=20):
        self.param_a = param_a
        self.param_b = param_b
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on parameters.
        For example, assign a buy signal (1) every time the index is divisible by param_a,
        and a sell signal (-1) every time it is divisible by param_b.
        """
        signals = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if i % self.param_a == 0:
                signals.iloc[i] = 1
            elif i % self.param_b == 0:
                signals.iloc[i] = -1
        return signals

# ------------------------------------------------------------------------------
# Test class for StrategyOptimizer
# ------------------------------------------------------------------------------
class TestStrategyOptimizer:
    @pytest.fixture
    def mock_backtester(self):
        """Create a mock backtester for testing."""
        mock = Mock(spec=Backtester)
        # Configure the mock to return a predetermined BacktestResult.
        mock.run_backtest.return_value = Mock(
            spec=BacktestResult,
            metrics={'sharpe_ratio': 1.5, 'max_drawdown': 10.0},
            portfolio=Mock(positions=[])
        )
        return mock
    
    @pytest.fixture
    def test_data(self):
        """Create test data for optimization."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(102, 5, 100),
            'volume': np.random.randint(1000, 5000, 100)
        })
    
    def test_generate_parameter_combinations(self, mock_backtester):
        """Test parameter grid generation."""
        optimizer = StrategyOptimizer(backtester=mock_backtester)
        param_grid = {
            'param_a': [5, 10, 15],
            'param_b': [20, 30]
        }
        combinations = optimizer._generate_parameter_combinations(param_grid)
        
        # There should be 3 x 2 = 6 combinations
        assert len(combinations) == 6
        assert {'param_a': 5, 'param_b': 20} in combinations
        assert {'param_a': 15, 'param_b': 30} in combinations
    
    def test_optimize(self, mock_backtester, test_data):
        """Test the optimization process."""
        optimizer = StrategyOptimizer(backtester=mock_backtester)
        
        # Define a simple parameter grid
        param_grid = {
            'param_a': [5, 10],
            'param_b': [20, 30]
        }
        
        # Run optimization using the ParameterizedIndicator
        results = optimizer.optimize(
            df=test_data,
            strategy_class=ParameterizedIndicator,
            strategy_param_grid=param_grid,
            metric_name='sharpe_ratio'
        )
        
        # Verify that the results contain best parameters and metrics,
        # and that we have 4 total results (2 x 2 combinations)
        assert 'best_params' in results
        assert 'best_metrics' in results
        assert len(results['all_results']) == 4
        
        # Verify that the backtester was called once per parameter combination.
        assert mock_backtester.run_backtest.call_count == 4
