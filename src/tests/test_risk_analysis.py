import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.backtesting.risk_analysis import MonteCarloSimulator, RiskAnalyzer
from src.backtesting.backtester import BacktestResult, Portfolio, Position

class TestRiskAnalyzer:
    @pytest.fixture
    def sample_returns(self):
        """Create a sample returns series for testing."""
        np.random.seed(42)  # For reproducibility
        return pd.Series(np.random.normal(0.001, 0.01, 252))
    
    def test_calculate_var(self, sample_returns):
        """Test Value at Risk (VaR) calculation using various methods."""
        analyzer = RiskAnalyzer()
        
        # Test different VaR methods
        historical_var = analyzer.calculate_var(sample_returns, 0.95, 'historical')
        parametric_var = analyzer.calculate_var(sample_returns, 0.95, 'parametric')
        monte_carlo_var = analyzer.calculate_var(sample_returns, 0.95, 'monte_carlo')
        
        # Assert that VaR values are in a reasonable range (adjust as needed)
        assert 0 < historical_var < 5
        assert 0 < parametric_var < 5
        assert 0 < monte_carlo_var < 5
        
    def test_calculate_drawdowns(self, sample_returns):
        """Test drawdown calculation based on an equity curve derived from returns."""
        analyzer = RiskAnalyzer()
        
        # Convert returns to an equity curve
        equity = (1 + sample_returns).cumprod() * 10000
        
        drawdown_result = analyzer.calculate_drawdowns(equity)
        
        # Check that the returned structure contains expected keys
        assert 'drawdowns' in drawdown_result
        assert 'metadata' in drawdown_result
        assert 'max_drawdown_pct' in drawdown_result['metadata']
        
    def test_kelly_criterion(self):
        """Test the Kelly criterion calculation."""
        analyzer = RiskAnalyzer()
        
        # Test a scenario with an edge (win rate > 0.5, win_loss ratio > 1)
        high_edge = analyzer.kelly_criterion(win_rate=0.6, win_loss_ratio=2.0)
        # Test a fair game scenario where no edge exists
        no_edge = analyzer.kelly_criterion(win_rate=0.5, win_loss_ratio=1.0)
        
        assert high_edge > 0
        assert no_edge == 0

class TestMonteCarloSimulator:
    @pytest.fixture
    def mock_backtest_result(self):
        """Create a mock BacktestResult for testing Monte Carlo simulations."""
        # Create a mock portfolio with positions
        portfolio = Mock(spec=Portfolio)
        portfolio.initial_capital = 10000
        portfolio.current_capital = 12000
        
        # Create a list of mock positions
        positions = []
        entry_time = datetime(2023, 1, 1)
        for i in range(30):
            pos = Mock(spec=Position)
            pos.entry_price = 100
            pos.exit_price = 110 if i % 3 != 0 else 90
            pos.entry_time = entry_time
            pos.exit_time = entry_time + timedelta(days=5)
            pos.direction = 1 if i % 2 == 0 else -1
            pos.pnl = 100 if pos.exit_price > pos.entry_price else -100
            pos.status = "CLOSED"
            positions.append(pos)
            entry_time += timedelta(days=7)
            
        portfolio.positions = positions
        portfolio.closed_positions = positions
        
        # Create an equity curve as a list of dicts with timestamp and equity
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        values = np.linspace(10000, 12000, 100)
        portfolio.equity_curve = [{'timestamp': d, 'equity': v} for d, v in zip(dates, values)]
        
        result = Mock(spec=BacktestResult)
        result.portfolio = portfolio
        result.metrics = {'sharpe_ratio': 1.5, 'max_drawdown': 10.0}
        
        return result
    
    def test_run_trade_simulation(self, mock_backtest_result):
        """Test the trade-based Monte Carlo simulation method."""
        simulator = MonteCarloSimulator()
        
        result = simulator.run_trade_simulation(
            backtest_result=mock_backtest_result,
            num_simulations=100,
            method='bootstrap'
        )
        
        assert 'final_capital_mean' in result
        assert 'drawdown_mean' in result
        assert 'win_probability' in result
        
    def test_run_returns_simulation(self, mock_backtest_result):
        """Test the returns-based Monte Carlo simulation method."""
        simulator = MonteCarloSimulator()
        
        result = simulator.run_returns_simulation(
            backtest_result=mock_backtest_result,
            num_simulations=100,
            future_periods=252
        )
        
        assert 'final_value_mean' in result
        assert 'cagr_mean' in result
        assert 'profit_probability' in result
