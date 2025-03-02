# src/backtesting/backtester.py

"""
Core backtesting framework for the Ultimate Data Fetcher.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging
from pathlib import Path
import json
import os

from src.core.models import StandardizedCandle
from src.core.exceptions import DataFetcherError
from src.utils.indicators.base_indicator import BaseIndicator
from src.utils.strategy.base import BaseStrategy
from src.storage.processed import ProcessedDataStorage
from src.core.symbol_mapper import SymbolMapper

logger = logging.getLogger(__name__)

class Position:
    """Represents a trading position."""
    
    def __init__(self, entry_price: float, entry_time: datetime, 
                 position_size: float, direction: int, 
                 stop_loss: Optional[float] = None, 
                 take_profit: Optional[float] = None):
        """
        Initialize a new position.
        
        Args:
            entry_price: Entry price of the position
            entry_time: Entry timestamp
            position_size: Size of position in base currency
            direction: 1 for long, -1 for short
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        """
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_size = position_size
        self.direction = direction  # 1 for long, -1 for short
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.status = "OPEN"
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_reason = None
        
    def close(self, exit_price: float, exit_time: datetime, reason: str = "SIGNAL"):
        """
        Close the position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for closing the position
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        # Calculate PnL
        price_diff = (exit_price - self.entry_price) * self.direction
        self.pnl = price_diff * self.position_size
        self.pnl_pct = (price_diff / self.entry_price) * 100
        
        self.status = "CLOSED"
        self.exit_reason = reason
        
    def to_dict(self) -> Dict:
        """Convert position to dictionary for analysis."""
        return {
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "position_size": self.position_size,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "status": self.status,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_reason": self.exit_reason,
            "duration_hours": (self.exit_time - self.entry_time).total_seconds() / 3600 if self.exit_time else None
        }

class Portfolio:
    """Tracks portfolio performance during backtesting."""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity_curve = []
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
    def open_position(self, entry_price: float, entry_time: datetime, 
                      position_size: float, direction: int,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None) -> Position:
        """
        Open a new position.
        
        Args:
            entry_price: Entry price
            entry_time: Entry timestamp
            position_size: Size in base currency
            direction: 1 for long, -1 for short
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            New Position object
        """
        position = Position(
            entry_price=entry_price,
            entry_time=entry_time,
            position_size=position_size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions.append(position)
        self.open_positions.append(position)
        return position
    
    def close_position(self, position: Position, exit_price: float, 
                      exit_time: datetime, reason: str = "SIGNAL"):
        """
        Close a position.
        
        Args:
            position: Position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for closing
        """
        if position not in self.open_positions:
            logger.warning(f"Attempted to close a position that isn't open: {position}")
            return
        
        position.close(exit_price, exit_time, reason)
        self.current_capital += position.pnl
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        self.open_positions.remove(position)
        self.closed_positions.append(position)
        
    def update_equity(self, current_price: float, timestamp: datetime):
        """
        Update equity curve with mark-to-market of open positions.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for pos in self.open_positions:
            price_diff = (current_price - pos.entry_price) * pos.direction
            unrealized_pnl += price_diff * pos.position_size
            
        # Current equity includes unrealized PnL
        current_equity = self.current_capital + unrealized_pnl
        
        # Update drawdown
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity
        current_drawdown = (self.peak_capital - current_equity) / self.peak_capital * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        # Record equity point
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'unrealized_pnl': unrealized_pnl,
            'drawdown': current_drawdown
        })

class BacktestResult:
    """Contains backtest results and performance metrics."""
    
    def __init__(self, portfolio: Portfolio, test_params: Dict):
        """
        Initialize backtest result.
        
        Args:
            portfolio: Completed portfolio from backtest
            test_params: Parameters used for the backtest
        """
        self.portfolio = portfolio
        self.test_params = test_params
        self.metrics = self._calculate_metrics()
        
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results."""
        
        if not self.portfolio.closed_positions:
            return {
                "total_return_pct": 0,
                "annualized_return_pct": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "total_trades": 0,
                "profitable_trades": 0,
                "avg_profit_trade": 0,
                "avg_loss_trade": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "avg_trade_duration_hours": 0,
                "profit_factor": 0
            }
            
        closed_positions = self.portfolio.closed_positions
        total_trades = len(closed_positions)
        profitable_trades = sum(1 for p in closed_positions if p.pnl > 0)
        
        # Extract equity curve as Series for calculations
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        equity_series = None
        if not equity_df.empty:
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df.set_index('timestamp', inplace=True)
            equity_series = equity_df['equity']
        
        # Basic performance metrics
        total_pnl = sum(p.pnl for p in closed_positions)
        total_return_pct = (self.portfolio.current_capital / self.portfolio.initial_capital - 1) * 100
        
        # Calculate annualized return
        if equity_series is not None and len(equity_series) > 1:
            days = (equity_series.index[-1] - equity_series.index[0]).days
            if days > 0:
                annualized_return_pct = ((1 + total_return_pct/100) ** (365/days) - 1) * 100
            else:
                annualized_return_pct = 0
        else:
            annualized_return_pct = 0
            
        # Win rate and averages
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_profit_trade = np.mean([p.pnl for p in closed_positions if p.pnl > 0]) if profitable_trades > 0 else 0
        avg_loss_trade = np.mean([p.pnl for p in closed_positions if p.pnl < 0]) if total_trades - profitable_trades > 0 else 0
        
        # Largest win/loss
        largest_win = max([p.pnl for p in closed_positions if p.pnl > 0], default=0)
        largest_loss = min([p.pnl for p in closed_positions if p.pnl < 0], default=0)
        
        # Trade duration
        durations = [(p.exit_time - p.entry_time).total_seconds() / 3600 for p in closed_positions if p.exit_time]
        avg_trade_duration_hours = np.mean(durations) if durations else 0
        
        # Profit factor
        gross_profit = sum(p.pnl for p in closed_positions if p.pnl > 0)
        gross_loss = abs(sum(p.pnl for p in closed_positions if p.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (if we have equity curve)
        sharpe_ratio = 0
        if equity_series is not None and len(equity_series) > 1:
            # Calculate daily returns
            daily_returns = equity_series.resample('D').last().pct_change().dropna()
            if len(daily_returns) > 0:
                sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0
        
        return {
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "win_rate": win_rate,
            "max_drawdown": self.portfolio.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "avg_profit_trade": avg_profit_trade,
            "avg_loss_trade": avg_loss_trade,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "avg_trade_duration_hours": avg_trade_duration_hours,
            "profit_factor": profit_factor
        }
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "test_params": self.test_params,
            "metrics": self.metrics,
            "equity_curve": self.portfolio.equity_curve,
            "positions": [p.to_dict() for p in self.portfolio.positions]
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Optional file path to save plot
        """
        if not self.portfolio.equity_curve:
            logger.warning("No equity curve data to plot")
            return
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Create plot with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        equity_df['equity'].plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title(f'Backtest Results - Return: {self.metrics["total_return_pct"]:.2f}%')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        
        # Add entry/exit markers on the equity curve
        entry_times = [p.entry_time for p in self.portfolio.positions]
        entry_values = [equity_df.loc[equity_df.index >= t, 'equity'].iloc[0] if t in equity_df.index or any(i >= t for i in equity_df.index) else None for t in entry_times]
        
        exit_times = [p.exit_time for p in self.portfolio.positions if p.exit_time]
        exit_values = [equity_df.loc[equity_df.index >= t, 'equity'].iloc[0] if t in equity_df.index or any(i >= t for i in equity_df.index) else None for t in exit_times]
        
        # Filter out None values
        entry_points = [(t, v) for t, v in zip(entry_times, entry_values) if v is not None]
        exit_points = [(t, v) for t, v in zip(exit_times, exit_values) if v is not None]
        
        if entry_points:
            entry_t, entry_v = zip(*entry_points)
            ax1.scatter(entry_t, entry_v, color='green', marker='^', s=50, label='Entry')
            
        if exit_points:
            exit_t, exit_v = zip(*exit_points)
            ax1.scatter(exit_t, exit_v, color='red', marker='v', s=50, label='Exit')
            
        ax1.legend()
        
        # Plot drawdown
        equity_df['drawdown'].plot(ax=ax2, color='red', linewidth=1.5)
        ax2.set_ylabel('Drawdown %')
        ax2.set_title(f'Maximum Drawdown: {self.metrics["max_drawdown"]:.2f}%')
        ax2.grid(True)
        ax2.invert_yaxis()  # Invert so drawdowns go down
        
        # Plot trade P&L
        trade_df = pd.DataFrame([{
            'exit_time': p.exit_time,
            'pnl': p.pnl,
            'direction': p.direction
        } for p in self.portfolio.closed_positions if p.exit_time])
        
        if not trade_df.empty:
            trade_df.set_index('exit_time', inplace=True)
            trade_df = trade_df.sort_index()
            
            colors = ['green' if x > 0 else 'red' for x in trade_df['pnl']]
            ax3.bar(trade_df.index, trade_df['pnl'], color=colors, width=0.7)
            ax3.set_ylabel('Trade P&L')
            ax3.set_title(f'Win Rate: {self.metrics["win_rate"]*100:.1f}% ({self.metrics["profitable_trades"]}/{self.metrics["total_trades"]})')
            ax3.grid(True, axis='y')
        else:
            ax3.set_title('No closed trades')
            
        # Format x-axis for all subplots
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def save_results(self, file_path: str):
        """
        Save backtest results to a file.
        
        Args:
            file_path: File path to save results
        """
        result_dict = self.to_dict()
        
        # Convert datetime objects to strings
        result_dict_serializable = json.loads(
            pd.json_normalize(result_dict).to_json(date_format='iso')
        )
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(result_dict_serializable, f, indent=2)
        
        logger.info(f"Backtest results saved to {file_path}")

class Backtester:
    """Core backtesting engine."""
    
    def __init__(self, data_storage: ProcessedDataStorage = None):
        """
        Initialize backtester.
        
        Args:
            data_storage: Optional ProcessedDataStorage instance for loading data
        """
        self.data_storage = data_storage
        self.symbol_mapper = SymbolMapper()
        
    async def load_data(self, exchange: str, market: str, 
                       resolution: str, start_time: datetime, 
                       end_time: datetime) -> pd.DataFrame:
        """
        Load price data for backtesting.
        
        Args:
            exchange: Exchange name
            market: Market symbol
            resolution: Candle resolution
            start_time: Start time
            end_time: End time
            
        Returns:
            DataFrame with price data
        """
        if self.data_storage is None:
            raise ValueError("No data storage configured. Please provide ProcessedDataStorage instance.")
        
        try:
            # Try to convert market to exchange-specific format if needed
            exchange_market = self.symbol_mapper.to_exchange_symbol(exchange, market)
            logger.info(f"Loading data for {exchange}:{exchange_market} ({resolution}) from {start_time} to {end_time}")
            
            df = await self.data_storage.load_candles(
                exchange=exchange,
                market=exchange_market,
                resolution=resolution,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            # If symbol mapping fails, try with the original market string
            logger.warning(f"Symbol mapping failed, trying with original market: {str(e)}")
            df = await self.data_storage.load_candles(
                exchange=exchange,
                market=market,
                resolution=resolution,
                start_time=start_time,
                end_time=end_time
            )
        
        if df.empty:
            raise DataFetcherError(f"No data found for {exchange} {market} {resolution} between {start_time} and {end_time}")
            
        return df
    
    def _check_stop_loss_take_profit(self, position: Position, 
                                    candle: pd.Series) -> Optional[str]:
        """
        Check if stop loss or take profit is hit.
        
        Args:
            position: Position to check
            candle: Current price candle
            
        Returns:
            Exit reason if SL/TP hit, otherwise None
        """
        if position.direction == 1:  # Long position
            # Check stop loss (assume we get filled at the low)
            if position.stop_loss and candle['low'] <= position.stop_loss:
                return "STOP_LOSS"
            # Check take profit (assume we get filled at the high)  
            if position.take_profit and candle['high'] >= position.take_profit:
                return "TAKE_PROFIT"
                
        else:  # Short position
            # Check stop loss (assume we get filled at the high)
            if position.stop_loss and candle['high'] >= position.stop_loss:
                return "STOP_LOSS"
            # Check take profit (assume we get filled at the low)
            if position.take_profit and candle['low'] <= position.take_profit:
                return "TAKE_PROFIT"
                
        return None
    
    def run_backtest_with_args(self, args: Any) -> BacktestResult:
        """
        Run a backtest using CLI arguments.
        
        Args:
            args: Arguments from CLI parser
            
        Returns:
            BacktestResult object with test results
        """
        from datetime import timedelta
        
        # Parse required arguments
        if not hasattr(args, 'market') or not args.market:
            raise ValueError("Market symbol is required")
            
        # Determine date range
        end_time = datetime.now()
        if hasattr(args, 'end_date') and args.end_date:
            end_time = args.end_date
            
        start_time = None
        if hasattr(args, 'start_date') and args.start_date:
            start_time = args.start_date
        elif hasattr(args, 'days') and args.days:
            start_time = end_time - timedelta(days=args.days)
        else:
            # Default to 90 days
            start_time = end_time - timedelta(days=90)
            
        # Get other parameters
        exchange = getattr(args, 'exchange', 'binance')
        resolution = getattr(args, 'resolution', '1D')
        initial_capital = getattr(args, 'capital', 10000.0)
        position_size_pct = getattr(args, 'position_size', 10.0)
        stop_loss_pct = getattr(args, 'stop_loss', None) 
        take_profit_pct = getattr(args, 'take_profit', None)
        commission_pct = getattr(args, 'commission', 0.1)
        slippage_pct = getattr(args, 'slippage', 0.05)
        max_positions = getattr(args, 'max_positions', 1)
        
        # Load data async
        import asyncio
        df = asyncio.run(self.load_data(
            exchange=exchange,
            market=args.market,
            resolution=resolution,
            start_time=start_time,
            end_time=end_time
        ))
        
        # Load strategy
        strategy = args.strategy_instance
        
        # Run backtest
        return self.run_backtest(
            df=df,
            strategy=strategy,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_positions=max_positions,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct
        )
    
    def run_backtest(self, df: pd.DataFrame, strategy: Union[BaseStrategy, BaseIndicator],
                    initial_capital: float = 10000.0,
                    position_size_pct: float = 10.0,
                    stop_loss_pct: Optional[float] = None,
                    take_profit_pct: Optional[float] = None,
                    max_positions: int = 1,
                    commission_pct: float = 0.1,
                    slippage_pct: float = 0.05) -> BacktestResult:
        """
        Run a backtest.
        
        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy or indicator that generates signals
            initial_capital: Starting capital
            position_size_pct: Position size as percentage of capital
            stop_loss_pct: Optional stop loss percentage
            take_profit_pct: Optional take profit percentage
            max_positions: Maximum number of concurrent positions
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
            
        Returns:
            BacktestResult object with test results
        """

        # Ensure we have all required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing required columns. Has: {df.columns}, needs: {required_cols}")
            
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # If indicator is provided, generate signals
        if isinstance(strategy, BaseIndicator):
            signals = strategy.generate_signals(df_copy)
        else:
            signals = strategy.generate_signals(df_copy)
            
        # Reset index if it's a DatetimeIndex to avoid potential issues
        has_datetime_index = isinstance(df_copy.index, pd.DatetimeIndex)
        if has_datetime_index:
            df_copy = df_copy.reset_index()
            # Find timestamp column (could be 'timestamp', 'index', or 'date')
            timestamp_col = None
            for col in ['timestamp', 'index', 'date']:
                if col in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    timestamp_col = col
                    break
                    
            if timestamp_col is None:
                raise ValueError("DataFrame has DatetimeIndex but no timestamp column found after reset_index()")
        else:
            if 'timestamp' not in df_copy.columns:
                raise ValueError("DataFrame must have a 'timestamp' column")
            timestamp_col = 'timestamp'
            
        # Initialize portfolio
        portfolio = Portfolio(initial_capital=initial_capital)
        
        # Loop through each candle
        for i, candle in df_copy.iterrows():
            if i == 0:
                # Skip first candle as we need previous signal for comparison
                continue
                
            current_price = candle['close']
            current_time = pd.to_datetime(candle[timestamp_col])
            current_signal = signals.iloc[i] if hasattr(signals, 'iloc') else signals[i]
            previous_signal = signals.iloc[i-1] if hasattr(signals, 'iloc') else signals[i-1]
            
            # Update mark-to-market for portfolio
            portfolio.update_equity(current_price, current_time)
            
            # Process existing positions
            for position in list(portfolio.open_positions):
                # Check stop loss / take profit
                exit_reason = self._check_stop_loss_take_profit(position, candle)
                
                if exit_reason:
                    # Use appropriate price based on exit reason
                    if exit_reason == "STOP_LOSS":
                        exit_price = position.stop_loss
                    elif exit_reason == "TAKE_PROFIT":
                        exit_price = position.take_profit
                    else:
                        exit_price = current_price
                        
                    # Apply slippage
                    if exit_reason != "TAKE_PROFIT":  # No slippage for TP (conservative)
                        exit_price = exit_price * (1 - slippage_pct/100 * position.direction) 
                        
                    # Apply commission
                    exit_cost = exit_price * position.position_size * (commission_pct/100)
                    final_exit_price = exit_price - (exit_cost / position.position_size) * position.direction
                    
                    portfolio.close_position(position, final_exit_price, current_time, exit_reason)
                    continue
                    
                # Exit position based on signal
                if (position.direction == 1 and current_signal == -1) or \
                   (position.direction == -1 and current_signal == 1) or \
                   (current_signal == 0 and previous_signal != 0):
                    
                    # Apply slippage
                    exit_price = current_price * (1 - slippage_pct/100 * position.direction)
                    
                    # Apply commission
                    exit_cost = exit_price * position.position_size * (commission_pct/100)
                    final_exit_price = exit_price - (exit_cost / position.position_size) * position.direction
                    
                    portfolio.close_position(position, final_exit_price, current_time, "SIGNAL")
            
            # Open new position if signaled and we have fewer than max_positions
            if current_signal != 0 and len(portfolio.open_positions) < max_positions and previous_signal != current_signal:
                # Calculate position size
                position_value = portfolio.current_capital * (position_size_pct/100)
                
                # Apply slippage
                entry_price = current_price * (1 + slippage_pct/100 * current_signal)
                
                # Apply commission
                position_size = position_value / entry_price
                entry_cost = entry_price * position_size * (commission_pct/100)
                effective_entry_price = entry_price + (entry_cost / position_size) * current_signal
                
                # Calculate stop loss and take profit prices if specified
                stop_loss = None
                take_profit = None
                
                if stop_loss_pct:
                    if current_signal == 1:  # Long
                        stop_loss = effective_entry_price * (1 - stop_loss_pct/100)
                    else:  # Short
                        stop_loss = effective_entry_price * (1 + stop_loss_pct/100)
                        
                if take_profit_pct:
                    if current_signal == 1:  # Long
                        take_profit = effective_entry_price * (1 + take_profit_pct/100)
                    else:  # Short
                        take_profit = effective_entry_price * (1 - take_profit_pct/100)
                
                # Open the position
                portfolio.open_position(
                    entry_price=effective_entry_price,
                    entry_time=current_time,
                    position_size=position_size,
                    direction=current_signal,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
        # Close any remaining open positions at the last price
        last_candle = df_copy.iloc[-1]
        last_price = last_candle['close']
        last_time = pd.to_datetime(last_candle[timestamp_col])
        
        for position in list(portfolio.open_positions):
            portfolio.close_position(position, last_price, last_time, "END_OF_BACKTEST")
            
        # Create result object
        test_params = {
            "start_time": pd.to_datetime(df_copy[timestamp_col].iloc[0]),
            "end_time": pd.to_datetime(df_copy[timestamp_col].iloc[-1]),
            "initial_capital": initial_capital,
            "position_size_pct": position_size_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "max_positions": max_positions,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "strategy": strategy.__class__.__name__
        }
        
        return BacktestResult(portfolio, test_params)
    
    def save_backtest(self, result: BacktestResult, output_dir: str = "backtest_results", 
                     prefix: str = None, include_plot: bool = True) -> str:
        """
        Save backtest results and plots to a directory.
        
        Args:
            result: BacktestResult object
            output_dir: Directory to save results
            prefix: Optional prefix for filenames
            include_plot: Whether to include plots
            
        Returns:
            Path to the saved results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        if prefix:
            base_filename = f"{prefix}_{timestamp}"
        else:
            strategy_name = result.test_params.get("strategy", "strategy")
            base_filename = f"{strategy_name}_{timestamp}"
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        result.save_results(json_path)
        
        # Save plot if requested
        if include_plot:
            plot_path = os.path.join(output_dir, f"{base_filename}.png")
            result.plot_results(plot_path)
        
        return json_path
    
    def print_results_summary(self, result: BacktestResult) -> None:
        """
        Print a summary of backtest results to the console.
        
        Args:
            result: BacktestResult object
        """
        metrics = result.metrics
        test_params = result.test_params
        
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {test_params.get('strategy', 'Strategy')}")
        print("="*60)
        
        # Date range
        start_time = test_params.get("start_time")
        end_time = test_params.get("end_time")
        if start_time and end_time:
            print(f"Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # Key metrics
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Trade statistics
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Avg Profit Trade: ${metrics['avg_profit_trade']:.2f}")
        print(f"  Avg Loss Trade: ${metrics['avg_loss_trade']:.2f}")
        print(f"  Avg Trade Duration: {metrics['avg_trade_duration_hours']:.2f} hours")
        
        # Strategy parameters
        print(f"\nStrategy Parameters:")
        print(f"  Initial Capital: ${test_params['initial_capital']:.2f}")
        print(f"  Position Size: {test_params['position_size_pct']:.2f}%")
        
        if test_params['stop_loss_pct']:
            print(f"  Stop Loss: {test_params['stop_loss_pct']:.2f}%")
        else:
            print(f"  Stop Loss: None")
            
        if test_params['take_profit_pct']:
            print(f"  Take Profit: {test_params['take_profit_pct']:.2f}%")
        else:
            print(f"  Take Profit: None")
            
        print(f"  Commission: {test_params['commission_pct']:.2f}%")
        print(f"  Slippage: {test_params['slippage_pct']:.2f}%")
        print("="*60)