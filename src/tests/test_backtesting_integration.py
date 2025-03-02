import pytest
import pandas as pd
import numpy as np
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.storage.processed import ProcessedDataStorage
from src.utils.indicators.wrapper_supertrend import SupertrendIndicator
from src.backtesting.backtester import Backtester, BacktestResult, Position, Portfolio
from src.backtesting.optimizer import StrategyOptimizer
from src.backtesting.performance_metrics import PerformanceAnalyzer
from src.backtesting.risk_analysis import RiskAnalyzer, MonteCarloSimulator

# Use TkAgg backend for interactive testing if needed (otherwise, Agg is also fine)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Fixture: Mock data storage that returns realistic price movement DataFrame
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_data_storage():
    """Create a mock ProcessedDataStorage with realistic price movements."""
    mock_storage = Mock(spec=ProcessedDataStorage)
    
    # Create realistic price data with trends
    dates = pd.date_range(start='2023-01-01', end='2023-06-01', freq='D')
    prices = []
    current_price = 20000.0
    
    # Generate price with trends that will trigger signals
    for i in range(len(dates)):
        if i < 50:
            change = np.random.normal(50, 20)       # Uptrend
        elif i < 100:
            change = np.random.normal(-40, 20)      # Downtrend
        elif i < 200:
            change = np.random.normal(100, 30)      # Strong uptrend
        elif i < 350:
            change = np.random.normal(0, 50)        # Sideways
        else:
            change = np.random.normal(-80, 25)      # Downtrend
            
        current_price += change
        current_price = max(current_price, 10000)   # Ensure price doesn't go too low
        prices.append(current_price)
    
    # Create a DataFrame with OHLCV data
    mock_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'volume': [np.random.uniform(1000, 5000) for _ in prices]
    })
    
    # Set load_candles to return our mock DataFrame (simulate async behavior)
    mock_storage.load_candles = AsyncMock(return_value=mock_df)
    return mock_storage

# ------------------------------------------------------------------------------
# Integration tests for the full backtesting pipeline
# ------------------------------------------------------------------------------

class TestBacktestingIntegration:

    @pytest.mark.asyncio
    async def test_backtester_with_data_storage(self, mock_data_storage):
        """Test that the Backtester loads data from storage and runs a backtest using a Supertrend indicator."""
        backtester = Backtester(data_storage=mock_data_storage)

        # Load data (using daily resolution; note that 2023-01-01 to 2023-06-01 gives ~152 data points)
        df = await backtester.load_data(
            exchange="binance",
            market="BTC-USDT",
            resolution="1D",
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 6, 1)
        )

        # Verify data loading
        assert len(df) == 152
        mock_data_storage.load_candles.assert_called_once()

        # Create indicator with a test fixture config (update the path if needed)
        indicator = SupertrendIndicator(
            config_path='src/tests/fixtures/indicator_settings.json', 
            atr_length=10, 
            factor=3
        )
        result = backtester.run_backtest(
            df=df,
            strategy=indicator,
            initial_capital=10000.0
        )

        # Check that the backtest produced positions and an equity curve
        assert isinstance(result, BacktestResult)
        assert result.portfolio.positions, "No positions were created."
        assert len(result.portfolio.equity_curve) > 0, "Equity curve is empty."

    @pytest.mark.asyncio
    async def test_optimizer_with_backtester(self, mock_data_storage):
        """Test that the StrategyOptimizer finds a best parameter set for the Supertrend strategy."""
        backtester = Backtester(data_storage=mock_data_storage)
        optimizer = StrategyOptimizer(backtester=backtester, data_storage=mock_data_storage)

        # Define parameter grid for Supertrend
        param_grid = {
            'atr_length': [10, 14, 20],
            'factor': [2.0, 3.0, 4.0]
        }

        # Load data
        df = await optimizer.load_data(
            exchange="binance",
            market="BTC-USDT",
            resolution="1D",
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 6, 1)
        )

        # Run optimization (ensure metric_name is supported by your new code)
        results = optimizer.optimize(
            df=df,
            strategy_class=SupertrendIndicator,
            strategy_param_grid=param_grid,
            metric_name='sharpe_ratio',
            strategy_config_path='src/tests/fixtures/indicator_settings.json'
        )

        assert 'best_params' in results
        assert len(results['all_results']) == 9  # 3 × 3 combinations expected

    @pytest.mark.asyncio
    async def test_full_backtest_pipeline(self, mock_data_storage):
        """Test the full backtesting pipeline including performance and risk analysis and Monte Carlo simulation."""
        # Initialize components
        backtester = Backtester(data_storage=mock_data_storage)
        performance_analyzer = PerformanceAnalyzer()
        risk_analyzer = RiskAnalyzer()
        monte_carlo = MonteCarloSimulator()

        # 1. Load data
        df = await backtester.load_data(
            exchange="binance",
            market="BTC-USDT",
            resolution="1D",
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 6, 1)
        )

        # 2. Run backtest using a Supertrend indicator
        indicator = SupertrendIndicator(
            config_path='src/tests/fixtures/indicator_settings.json', 
            atr_length=14, 
            factor=3
        )
        backtest_result = backtester.run_backtest(
            df=df,
            strategy=indicator,
            initial_capital=10000.0,
            position_size_pct=10.0,
            stop_loss_pct=5.0
        )

        # 3. Analyze performance: Build an equity curve Series from portfolio data
        equity_curve = pd.Series(
            [point['equity'] for point in backtest_result.portfolio.equity_curve],
            index=pd.to_datetime([point['timestamp'] for point in backtest_result.portfolio.equity_curve])
        )
        performance_report = performance_analyzer.analyze_equity_curve(
            equity_curve=equity_curve,
            trades=[p.to_dict() for p in backtest_result.portfolio.closed_positions]
        )

        # 4. Analyze risk using daily returns
        returns = equity_curve.pct_change().dropna()
        risk_metrics = risk_analyzer.calculate_risk_metrics(returns)

        # 5. Run Monte Carlo simulation on the backtest result
        mc_results = monte_carlo.run_trade_simulation(
            backtest_result=backtest_result,
            num_simulations=100
        )

        # Verify complete pipeline outputs
        assert backtest_result.metrics['total_return_pct']
        assert performance_report.return_stats.annual_return_pct
        assert risk_metrics['sharpe_ratio']
        assert mc_results['final_capital_mean']
