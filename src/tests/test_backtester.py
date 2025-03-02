import pytest
from datetime import datetime
import pandas as pd
import numpy as np

from src.backtesting.backtester import Position, Portfolio, BacktestResult, Backtester
from src.utils.indicators.base_indicator import BaseIndicator


# ------------------------------------------------------------------------------
# MockIndicator for testing purposes
# ------------------------------------------------------------------------------
class MockIndicator(BaseIndicator):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Example mock signals - you can adjust this for different test scenarios.
        This always returns +1 (buy) except for rows 5–9 (sell) and rows 15+ (neutral).
        """
        signals = pd.Series(1, index=df.index)  # Always buy signal
        signals.iloc[5:10] = -1  # Sell signal for a period
        signals.iloc[15:] = 0    # Neutral signal for the rest
        return signals


# ------------------------------------------------------------------------------
# Tests for the Backtester, Positions, and Portfolio
# ------------------------------------------------------------------------------
class TestBacktester:
    def test_backtest_with_mock_indicator(self):
        """
        Test a full backtest run with a mock indicator to ensure
        the Backtester and BacktestResult classes handle signals and metrics.
        """
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(100, 110, 20),
            'high': np.linspace(102, 112, 20),
            'low': np.linspace(98, 108, 20),
            'close': np.linspace(101, 111, 20),
            'volume': np.random.randint(1000, 5000, 20)
        })

        # Initialize the backtester and indicator
        backtester = Backtester()
        indicator = MockIndicator()

        # Run backtest
        # NOTE: If your new Backtester code changed the signature of run_backtest,
        #       you may need to adapt these arguments.
        result = backtester.run_backtest(
            df=df,
            strategy=indicator,
            initial_capital=10000.0,
            position_size_pct=10.0
        )

        # Verify we got a valid BacktestResult
        assert isinstance(result, BacktestResult)

        # Ensure the portfolio has at least one position and an equity curve
        assert len(result.portfolio.positions) > 0
        assert len(result.portfolio.equity_curve) > 0

        # Check the metrics dictionary (if your code still uses result.metrics)
        assert result.metrics is not None
        # Confirm each key exists and is not None
        for key in [
            "total_return_pct", "annualized_return_pct", "max_drawdown", "win_rate",
            "total_trades", "profitable_trades", "avg_profit_trade", "avg_loss_trade",
            "largest_win", "largest_loss", "avg_trade_duration_hours", "profit_factor"
        ]:
            assert key in result.metrics, f"Missing metric: {key}"
            assert result.metrics[key] is not None, f"Metric {key} is None"

        # Example of further checks
        assert result.metrics["max_drawdown"] >= 0
        assert 0 <= result.metrics["win_rate"] <= 1
        assert result.metrics["largest_loss"] <= 0  # Largest loss typically negative
        assert result.metrics["profit_factor"] >= 0


    def test_position_initialization(self):
        """
        Test creation of a Position object to ensure fields are initialized properly.
        """
        entry_time = datetime(2023, 1, 1, 12, 0)
        pos = Position(
            entry_price=100.0,
            entry_time=entry_time,
            position_size=2.5,
            direction=1,
            stop_loss=95.0,
            take_profit=110.0
        )

        assert pos.entry_price == 100.0
        assert pos.entry_time == entry_time
        assert pos.position_size == 2.5
        assert pos.direction == 1  # 1 => LONG, -1 => SHORT
        assert pos.stop_loss == 95.0
        assert pos.take_profit == 110.0
        assert pos.status == "OPEN"
        assert pos.exit_price is None
        assert pos.exit_time is None
        assert pos.pnl == 0.0
        assert pos.pnl_pct == 0.0
        assert pos.exit_reason is None

    def test_position_close(self):
        """
        Test closing a Position (both long and short) and verify P&L calculations.
        """
        entry_time = datetime(2023, 1, 1, 12, 0)
        exit_time = datetime(2023, 1, 2, 12, 0)

        # Test long position
        long_pos = Position(
            entry_price=100.0,
            entry_time=entry_time,
            position_size=2.0,
            direction=1
        )
        long_pos.close(exit_price=110.0, exit_time=exit_time, reason="SIGNAL")

        assert long_pos.exit_price == 110.0
        assert long_pos.exit_time == exit_time
        assert long_pos.pnl == 20.0  # (110-100) * 2
        assert long_pos.pnl_pct == 10.0  # (110-100)/100 * 100
        assert long_pos.status == "CLOSED"
        assert long_pos.exit_reason == "SIGNAL"

        # Test short position
        short_pos = Position(
            entry_price=100.0,
            entry_time=entry_time,
            position_size=2.0,
            direction=-1
        )
        short_pos.close(exit_price=90.0, exit_time=exit_time, reason="TAKE_PROFIT")

        assert short_pos.exit_price == 90.0
        assert short_pos.pnl == 20.0  # (100-90) * 2 for a short
        assert short_pos.pnl_pct == 10.0
        assert short_pos.status == "CLOSED"
        assert short_pos.exit_reason == "TAKE_PROFIT"

    def test_position_to_dict(self):
        """
        Test converting a Position to a dictionary for logging or serialization.
        """
        entry_time = datetime(2023, 1, 1, 12, 0)
        exit_time = datetime(2023, 1, 2, 12, 0)

        pos = Position(
            entry_price=100.0,
            entry_time=entry_time,
            position_size=2.0,
            direction=1
        )
        pos.close(exit_price=105.0, exit_time=exit_time, reason="SIGNAL")

        pos_dict = pos.to_dict()

        # If your new code changed the direction string from "LONG" to something else,
        # adapt this assertion accordingly.
        assert pos_dict["entry_price"] == 100.0
        assert pos_dict["entry_time"] == entry_time
        assert pos_dict["direction"] == "LONG"
        assert pos_dict["exit_price"] == 105.0
        assert pos_dict["pnl"] == 10.0
        assert pos_dict["duration_hours"] == 24.0

    def test_portfolio_initialization(self):
        """
        Test portfolio creation to confirm initial capital is set
        and that no positions or equity curve data exist at creation.
        """
        portfolio = Portfolio(initial_capital=10000.0)

        assert portfolio.initial_capital == 10000.0
        assert portfolio.current_capital == 10000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.open_positions) == 0
        assert len(portfolio.closed_positions) == 0
        assert len(portfolio.equity_curve) == 0
        # If your new code changed the name from peak_capital to something else (e.g. peak_equity),
        # adapt this assertion accordingly:
        assert portfolio.peak_capital == 10000.0

    def test_open_position(self):
        """
        Test opening a new position through the Portfolio
        to ensure it is added to open_positions and positions.
        """
        portfolio = Portfolio(initial_capital=10000.0)
        entry_time = datetime(2023, 1, 1, 12, 0)

        position = portfolio.open_position(
            entry_price=100.0,
            entry_time=entry_time,
            position_size=2.0,
            direction=1,
            stop_loss=95.0,
            take_profit=110.0
        )

        assert len(portfolio.positions) == 1
        assert len(portfolio.open_positions) == 1
        assert position in portfolio.positions
        assert position in portfolio.open_positions
        assert position.entry_price == 100.0
        assert position.stop_loss == 95.0

    def test_close_position(self):
        """
        Test closing a position via the Portfolio, ensuring
        it moves from open_positions to closed_positions.
        """
        portfolio = Portfolio(initial_capital=10000.0)
        entry_time = datetime(2023, 1, 1, 12, 0)
        exit_time = datetime(2023, 1, 2, 12, 0)

        position = portfolio.open_position(
            entry_price=100.0,
            entry_time=entry_time,
            position_size=2.0,
            direction=1
        )
        portfolio.close_position(position, exit_price=110.0, exit_time=exit_time, reason="SIGNAL")

        assert len(portfolio.open_positions) == 0
        assert len(portfolio.closed_positions) == 1
        assert position.status == "CLOSED"
        assert portfolio.current_capital == 10020.0  # initial 10000 + 20 PnL

    def test_update_equity(self):
        """
        Test updating the equity curve for a portfolio with an open position
        and verify that the correct unrealized PnL is reflected.
        """
        portfolio = Portfolio(initial_capital=10000.0)
        entry_time = datetime(2023, 1, 1, 12, 0)
        update_time_1 = datetime(2023, 1, 1, 13, 0)
        update_time_2 = datetime(2023, 1, 1, 14, 0)

        position = portfolio.open_position(
            entry_price=100.0,
            entry_time=entry_time,
            position_size=5.0,
            direction=1
        )

        portfolio.update_equity(current_price=105.0, timestamp=update_time_1)
        portfolio.update_equity(current_price=110.0, timestamp=update_time_2)

        # If your new code logs equity differently, adjust these checks
        assert len(portfolio.equity_curve) == 2
        # Each share is up by (price - 100), times 5 shares => unrealized PnL
        # After first update: (105 - 100)*5 = 25 => total = 10025
        assert portfolio.equity_curve[0]['equity'] == 10025.0
        # Second update: (110 - 100)*5 = 50 => total = 10050
        assert portfolio.equity_curve[1]['equity'] == 10050.0
        # No drawdown in a strictly rising market
        assert portfolio.max_drawdown == 0.0
