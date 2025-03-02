import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting.performance_metrics import (
    TradeStats, ReturnStats, DrawdownStats, RiskMetrics,
    PerformanceReport, PerformanceAnalyzer
)

class TestPerformanceAnalyzer:
    @pytest.fixture
    def sample_equity_curve(self):
        """Create a sample equity curve for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # Generate an equity curve that fluctuates around an upward trend.
        values = [10000]
        for i in range(1, 100):
            change = np.random.normal(0.001, 0.01)  # small random returns
            values.append(values[-1] * (1 + change))
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data for testing."""
        trades = []
        entry_time = datetime(2023, 1, 1)
        for i in range(20):
            # Alternate between long and short trades
            direction = 1 if i % 2 == 0 else -1
            profit = 100 if i % 3 != 0 else -80
            trade = {
                'entry_time': entry_time,
                'exit_time': entry_time + timedelta(days=5),
                'entry_price': 100,
                'exit_price': 110 if profit > 0 else 90,
                'direction': direction,
                'pnl': profit * direction,
                'exit_reason': 'SIGNAL' if i % 4 != 0 else 'STOP_LOSS'
            }
            trades.append(trade)
            entry_time += timedelta(days=7)
        return trades
    
    def test_calculate_trade_statistics(self, sample_trades):
        """Test trade statistics calculation."""
        analyzer = PerformanceAnalyzer()
        stats = analyzer._calculate_trade_statistics(sample_trades)
        
        assert isinstance(stats, TradeStats)
        assert stats.total_trades == 20
        # You can add additional assertions here for average profit, win rate, etc.
    
    def test_calculate_returns(self, sample_equity_curve):
        """Test return statistics calculation."""
        analyzer = PerformanceAnalyzer()
        stats = analyzer._calculate_returns(sample_equity_curve)
        
        # Ensure the key metrics are present in the returned dictionary.
        assert 'total_return_pct' in stats
        assert 'annual_return_pct' in stats
        assert 'volatility_annual_pct' in stats
        # Optionally, check that values are within expected ranges:
        assert isinstance(stats['total_return_pct'], float)
        assert isinstance(stats['annual_return_pct'], float)
        assert isinstance(stats['volatility_annual_pct'], float)
    
    def test_analyze_equity_curve(self, sample_equity_curve, sample_trades):
        """Test full equity curve analysis returns a valid performance report."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze_equity_curve(
            equity_curve=sample_equity_curve,
            trades=sample_trades
        )
        
        # Check that the report is of the correct type.
        assert isinstance(report, PerformanceReport)
        # Verify trade statistics
        assert report.trade_stats.total_trades == 20
        # Check return statistics are computed as floats.
        assert isinstance(report.return_stats.total_return_pct, float)
        # Check drawdown statistics are computed.
        assert isinstance(report.drawdown_stats.max_drawdown_pct, float)
        # Check that risk metrics include a sharpe ratio computed as a float.
        assert isinstance(report.risk_metrics.sharpe_ratio, float)
