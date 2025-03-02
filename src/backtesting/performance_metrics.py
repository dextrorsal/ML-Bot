 #src/backtesting/performance_metrics.py

"""
Performance metrics for evaluating trading strategies.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TradeStats:
    """Statistics about completed trades."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    largest_profit: float = 0.0
    largest_loss: float = 0.0
    avg_holding_bars: float = 0.0
    avg_profit_bars: float = 0.0
    avg_loss_bars: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0
    avg_mae: float = 0.0  # Maximum Adverse Excursion
    avg_mfe: float = 0.0  # Maximum Favorable Excursion
    profit_per_trade: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    trades_by_exit: Dict[str, int] = field(default_factory=dict)
    trades_by_day: Dict[str, int] = field(default_factory=dict)  # e.g., 'Monday': 5
    trades_by_hour: Dict[int, int] = field(default_factory=dict)  # e.g., 9: 10
    
    def to_dict(self) -> Dict[str, Union[int, float, Dict]]:
        """Convert TradeStats to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "largest_profit": self.largest_profit,
            "largest_loss": self.largest_loss,
            "avg_holding_bars": self.avg_holding_bars,
            "avg_profit_bars": self.avg_profit_bars,
            "avg_loss_bars": self.avg_loss_bars,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "expectancy": self.expectancy,
            "avg_mae": self.avg_mae,
            "avg_mfe": self.avg_mfe,
            "profit_per_trade": self.profit_per_trade,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "long_win_rate": self.long_win_rate,
            "short_win_rate": self.short_win_rate,
            "trades_by_exit": self.trades_by_exit,
            "trades_by_day": self.trades_by_day,
            "trades_by_hour": self.trades_by_hour
        }

@dataclass
class ReturnStats:
    """Statistics about returns."""
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    daily_return_pct: float = 0.0
    monthly_return_pct: float = 0.0
    volatility_annual_pct: float = 0.0
    volatility_daily_pct: float = 0.0
    downside_volatility_pct: float = 0.0
    best_month_pct: float = 0.0
    worst_month_pct: float = 0.0
    best_day_pct: float = 0.0
    worst_day_pct: float = 0.0
    monthly_returns: Dict[str, float] = field(default_factory=dict)  # e.g., '2023-01': 2.5
    avg_up_month_pct: float = 0.0
    avg_down_month_pct: float = 0.0
    pct_profitable_months: float = 0.0
    pct_profitable_days: float = 0.0
    
    def to_dict(self) -> Dict[str, Union[float, Dict]]:
        """Convert ReturnStats to dictionary."""
        return {
            "total_return_pct": self.total_return_pct,
            "annual_return_pct": self.annual_return_pct,
            "daily_return_pct": self.daily_return_pct,
            "monthly_return_pct": self.monthly_return_pct,
            "volatility_annual_pct": self.volatility_annual_pct,
            "volatility_daily_pct": self.volatility_daily_pct,
            "downside_volatility_pct": self.downside_volatility_pct,
            "best_month_pct": self.best_month_pct,
            "worst_month_pct": self.worst_month_pct,
            "best_day_pct": self.best_day_pct,
            "worst_day_pct": self.worst_day_pct,
            "monthly_returns": self.monthly_returns,
            "avg_up_month_pct": self.avg_up_month_pct,
            "avg_down_month_pct": self.avg_down_month_pct,
            "pct_profitable_months": self.pct_profitable_months,
            "pct_profitable_days": self.pct_profitable_days
        }

@dataclass
class DrawdownStats:
    """Statistics about drawdowns."""
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0  # In days
    avg_drawdown_pct: float = 0.0
    avg_drawdown_duration: int = 0
    current_drawdown_pct: float = 0.0
    current_drawdown_duration: int = 0
    drawdowns: List[Dict[str, Union[float, int, datetime]]] = field(default_factory=list)
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    pain_index: float = 0.0
    calmar_ratio: float = 0.0  # Annual return / Max drawdown
    sterling_ratio: float = 0.0  # Annual return / Avg of 3 worst drawdowns
    burke_ratio: float = 0.0
    time_to_recovery: int = 0  # Days to recover from max drawdown
    
    def to_dict(self) -> Dict[str, Union[float, int, List]]:
        """Convert DrawdownStats to dictionary."""
        return {
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_drawdown_pct": self.avg_drawdown_pct,
            "avg_drawdown_duration": self.avg_drawdown_duration,
            "current_drawdown_pct": self.current_drawdown_pct,
            "current_drawdown_duration": self.current_drawdown_duration,
            "drawdowns": self.drawdowns,
            "recovery_factor": self.recovery_factor,
            "ulcer_index": self.ulcer_index,
            "pain_index": self.pain_index,
            "calmar_ratio": self.calmar_ratio,
            "sterling_ratio": self.sterling_ratio,
            "burke_ratio": self.burke_ratio,
            "time_to_recovery": self.time_to_recovery
        }

@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    modigliani_ratio: float = 0.0
    omega_ratio: float = 0.0
    capture_ratio: float = 0.0
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    r_squared: float = 0.0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    tail_ratio: float = 0.0
    stability_of_timeseries: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert RiskMetrics to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "treynor_ratio": self.treynor_ratio,
            "information_ratio": self.information_ratio,
            "modigliani_ratio": self.modigliani_ratio,
            "omega_ratio": self.omega_ratio,
            "capture_ratio": self.capture_ratio,
            "upside_capture": self.upside_capture,
            "downside_capture": self.downside_capture,
            "beta": self.beta,
            "alpha": self.alpha,
            "r_squared": self.r_squared,
            "value_at_risk_95": self.value_at_risk_95,
            "conditional_var_95": self.conditional_var_95,
            "tail_ratio": self.tail_ratio,
            "stability_of_timeseries": self.stability_of_timeseries,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis
        }

@dataclass
class PerformanceReport:
    """Comprehensive performance report combining all metrics."""
    trade_stats: TradeStats = field(default_factory=TradeStats)
    return_stats: ReturnStats = field(default_factory=ReturnStats)
    drawdown_stats: DrawdownStats = field(default_factory=DrawdownStats)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    backtest_period: Tuple[datetime, datetime] = None
    initial_capital: float = 0.0
    final_capital: float = 0.0
    benchmark_return_pct: float = 0.0
    benchmark_volatility_pct: float = 0.0
    benchmark_max_drawdown_pct: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert the entire performance report to a dictionary."""
        result = {
            "trade_stats": self.trade_stats.to_dict(),
            "return_stats": self.return_stats.to_dict(),
            "drawdown_stats": self.drawdown_stats.to_dict(),
            "risk_metrics": self.risk_metrics.to_dict(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "benchmark_return_pct": self.benchmark_return_pct,
            "benchmark_volatility_pct": self.benchmark_volatility_pct,
            "benchmark_max_drawdown_pct": self.benchmark_max_drawdown_pct,
            "additional_metrics": self.additional_metrics
        }
        
        if self.backtest_period:
            result["backtest_start"] = self.backtest_period[0].isoformat()
            result["backtest_end"] = self.backtest_period[1].isoformat()
            
        return result
    
    def summary(self) -> str:
        """Generate a text summary of key performance metrics."""
        return f"""
PERFORMANCE SUMMARY
------------------
Period: {self.backtest_period[0].strftime('%Y-%m-%d')} to {self.backtest_period[1].strftime('%Y-%m-%d')}
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${self.final_capital:,.2f}

RETURNS
-------
Total Return: {self.return_stats.total_return_pct:.2f}%
Annualized Return: {self.return_stats.annual_return_pct:.2f}%
Volatility (Annual): {self.return_stats.volatility_annual_pct:.2f}%

RISK METRICS
-----------
Sharpe Ratio: {self.risk_metrics.sharpe_ratio:.2f}
Sortino Ratio: {self.risk_metrics.sortino_ratio:.2f}
Max Drawdown: {self.drawdown_stats.max_drawdown_pct:.2f}%
Calmar Ratio: {self.drawdown_stats.calmar_ratio:.2f}

TRADE STATISTICS
---------------
Total Trades: {self.trade_stats.total_trades}
Win Rate: {self.trade_stats.win_rate:.2f}%
Profit Factor: {self.trade_stats.profit_factor:.2f}
Expectancy: ${self.trade_stats.expectancy:.2f}

VS BENCHMARK
-----------
Benchmark Return: {self.benchmark_return_pct:.2f}%
Alpha: {self.risk_metrics.alpha:.2f}%
Beta: {self.risk_metrics.beta:.2f}
Information Ratio: {self.risk_metrics.information_ratio:.2f}
"""


class PerformanceAnalyzer:
    """
    Analyzer for calculating trading performance metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate as decimal (e.g., 0.02 for 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        
    def _calculate_drawdowns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdowns from equity curve.
        
        Args:
            equity_curve: Series of portfolio equity values
            
        Returns:
            DataFrame with drawdown metrics
        """
        # Ensure equity is a Series with DatetimeIndex
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve = pd.Series(equity_curve.values, 
                                     index=pd.to_datetime(equity_curve.index))
        
        # Calculate expanding maximum
        expanding_max = equity_curve.expanding().max()
        
        # Calculate drawdown in dollars
        drawdown = equity_curve - expanding_max
        
        # Calculate drawdown in percentage
        drawdown_pct = drawdown / expanding_max * 100
        
        # Find drawdown periods
        is_drawdown = drawdown_pct < 0
        
        # Setup storage for drawdown periods
        drawdown_periods = []
        current_drawdown_start = None
        
        # Iterate through the drawdown Series
        for date, value in is_drawdown.items():
            if value and current_drawdown_start is None:
                current_drawdown_start = date
            elif not value and current_drawdown_start is not None:
                # We've found the end of a drawdown period
                drawdown_end = date
                
                # Find the maximum drawdown in this period
                period_mask = (drawdown_pct.index >= current_drawdown_start) & \
                              (drawdown_pct.index <= drawdown_end)
                period_dd = drawdown_pct[period_mask]
                max_dd = period_dd.min()
                max_dd_date = period_dd.idxmin()
                
                # Calculate recovery time in days
                recovery_days = (drawdown_end - max_dd_date).days
                drawdown_duration = (drawdown_end - current_drawdown_start).days
                
                # Only record significant drawdowns (e.g., > 1%)
                if max_dd < -1:
                    drawdown_periods.append({
                        'start_date': current_drawdown_start,
                        'end_date': drawdown_end,
                        'max_drawdown_date': max_dd_date,
                        'max_drawdown_pct': max_dd,
                        'duration_days': drawdown_duration,
                        'recovery_days': recovery_days
                    })
                
                current_drawdown_start = None
        
        # Check if we're still in a drawdown at the end of the series
        if current_drawdown_start is not None:
            last_date = drawdown_pct.index[-1]
            period_mask = drawdown_pct.index >= current_drawdown_start
            period_dd = drawdown_pct[period_mask]
            max_dd = period_dd.min()
            max_dd_date = period_dd.idxmin()
            drawdown_duration = (last_date - current_drawdown_start).days
            
            # Only record significant drawdowns
            if max_dd < -1:
                drawdown_periods.append({
                    'start_date': current_drawdown_start,
                    'end_date': None,  # Still ongoing
                    'max_drawdown_date': max_dd_date,
                    'max_drawdown_pct': max_dd,
                    'duration_days': drawdown_duration,
                    'recovery_days': None  # Still recovering
                })
        
        # Calculate Ulcer Index (UI)
        squared_dd = np.sqrt(np.mean(np.square(drawdown_pct / 100)))
        ulcer_index = squared_dd * 100  # Convert back to percentage
        
        # Calculate Pain Index (average absolute drawdown)
        pain_index = np.abs(drawdown_pct[drawdown_pct < 0]).mean() if len(drawdown_pct[drawdown_pct < 0]) > 0 else 0
        
        return {
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct,
            'drawdown_periods': drawdown_periods,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index
        }
    
    def _calculate_returns(self, equity_curve: pd.Series) -> Dict:
        """
        Calculate return metrics from equity curve.
        
        Args:
            equity_curve: Series of portfolio equity values
            
        Returns:
            Dictionary with return metrics
        """
        # Ensure equity is a Series with DatetimeIndex
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve = pd.Series(equity_curve.values, 
                                     index=pd.to_datetime(equity_curve.index))
            
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate log returns for accurate compounding
        log_returns = np.log(equity_curve / equity_curve.shift(1)).dropna()
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        # Calculate period lengths
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        trading_days = len(returns)
        calendar_days = (end_date - start_date).days
        years = calendar_days / 365
        
        # Annualized return using CAGR formula
        annual_return = (((1 + total_return / 100) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        # Daily/monthly metrics
        returns_daily = returns.copy()
        returns_monthly = returns.resample('ME').sum()
        
        daily_return_mean = returns_daily.mean() * 100
        monthly_return_mean = returns_monthly.mean() * 100
        
        # Volatility calculations
        daily_volatility = returns_daily.std() * 100
        annual_volatility = daily_volatility * np.sqrt(252)
        monthly_volatility = returns_monthly.std() * 100
        
        # Downside volatility (negative returns only)
        downside_returns = returns_daily[returns_daily < 0]
        downside_volatility = downside_returns.std() * 100 * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Best/worst periods
        best_day = returns_daily.max() * 100
        worst_day = returns_daily.min() * 100
        best_month = returns_monthly.max() * 100
        worst_month = returns_monthly.min() * 100
        
        # Percentage of profitable periods
        pct_profitable_days = (returns_daily > 0).mean() * 100
        pct_profitable_months = (returns_monthly > 0).mean() * 100
        
        # Average up/down months
        up_months = returns_monthly[returns_monthly > 0]
        down_months = returns_monthly[returns_monthly < 0]
        avg_up_month = up_months.mean() * 100 if len(up_months) > 0 else 0
        avg_down_month = down_months.mean() * 100 if len(down_months) > 0 else 0
        
        # Monthly returns by calendar month
        monthly_returns_dict = {}
        for date, value in returns_monthly.items():
            month_key = date.strftime("%Y-%m")
            monthly_returns_dict[month_key] = value * 100
            
        return {
            'equity_curve': equity_curve,
            'returns': returns,
            'log_returns': log_returns,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'daily_return_pct': daily_return_mean,
            'monthly_return_pct': monthly_return_mean,
            'volatility_daily_pct': daily_volatility,
            'volatility_annual_pct': annual_volatility,
            'volatility_monthly_pct': monthly_volatility,
            'downside_volatility_pct': downside_volatility,
            'best_day_pct': best_day,
            'worst_day_pct': worst_day,
            'best_month_pct': best_month,
            'worst_month_pct': worst_month,
            'pct_profitable_days': pct_profitable_days,
            'pct_profitable_months': pct_profitable_months,
            'avg_up_month_pct': avg_up_month,
            'avg_down_month_pct': avg_down_month,
            'monthly_returns': monthly_returns_dict,
            'trading_days': trading_days,
            'calendar_days': calendar_days,
            'years': years
        }
            
    def _calculate_trade_statistics(self, trades: List[Dict]) -> TradeStats:
        """
        Calculate trade statistics from a list of trade dictionaries.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            TradeStats object
        """
        if not trades:
            return TradeStats()
        
        # Initialize counters
        total_trades = len(trades)
        winning_trades = 0
        losing_trades = 0
        breakeven_trades = 0
        gross_profits = 0.0
        gross_losses = 0.0
        total_profit = 0.0
        holding_periods = []
        profit_holding_periods = []
        loss_holding_periods = []
        recent_trade_results = []  # For tracking consecutive wins/losses
        max_profit = 0.0
        max_loss = 0.0
        total_mae = 0.0  # Maximum Adverse Excursion
        total_mfe = 0.0  # Maximum Favorable Excursion
        
        # Direction stats
        long_trades = 0
        long_wins = 0
        short_trades = 0
        short_wins = 0
        
        # Exit reason stats
        exit_reasons = {}
        
        # Time-of-day and day-of-week stats
        trades_by_day = {}
        trades_by_hour = {}
        
        for trade in trades:
            # Basic categorization
            pnl = trade.get('pnl', 0)
            total_profit += pnl
            
            if pnl > 0:
                winning_trades += 1
                gross_profits += pnl
                if pnl > max_profit:
                    max_profit = pnl
                recent_trade_results.append(1)
            elif pnl < 0:
                losing_trades += 1
                gross_losses += abs(pnl)
                if abs(pnl) > max_loss:
                    max_loss = abs(pnl)
                recent_trade_results.append(-1)
            else:
                breakeven_trades += 1
                recent_trade_results.append(0)
                
            # Track direction performance
            direction = trade.get('direction', 0)
            if direction == 1:  # Long
                long_trades += 1
                if pnl > 0:
                    long_wins += 1
            elif direction == -1:  # Short
                short_trades += 1
                if pnl > 0:
                    short_wins += 1
                    
            # Track exit reasons
            exit_reason = trade.get('exit_reason', 'Unknown')
            exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
            
            # Track MAE and MFE if available
            mae = trade.get('max_adverse_excursion', 0)
            mfe = trade.get('max_favorable_excursion', 0)
            total_mae += abs(mae)
            total_mfe += mfe
            
            # Holding period statistics
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                holding_period = (exit_time - entry_time).total_seconds() / 3600  # in hours
                holding_periods.append(holding_period)
                
                if pnl > 0:
                    profit_holding_periods.append(holding_period)
                elif pnl < 0:
                    loss_holding_periods.append(holding_period)
                    
                # Day of week stats
                day_of_week = entry_time.strftime('%A')
                trades_by_day[day_of_week] = trades_by_day.get(day_of_week, 0) + 1
                
                # Hour of day stats
                hour_of_day = entry_time.hour
                trades_by_hour[hour_of_day] = trades_by_hour.get(hour_of_day, 0) + 1
        
        # Calculate win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate profit factor
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Calculate average profit/loss
        avg_profit = gross_profits / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_losses / losing_trades if losing_trades > 0 else 0
        
        # Calculate average holding periods
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        avg_profit_period = np.mean(profit_holding_periods) if profit_holding_periods else 0
        avg_loss_period = np.mean(loss_holding_periods) if loss_holding_periods else 0
        
        # Calculate consecutive win/loss streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        last_result = 0
        
        for result in recent_trade_results:
            if result == last_result and result != 0:
                current_streak += 1
            else:
                current_streak = 1 if result != 0 else 0
                
            if result == 1 and current_streak > max_win_streak:
                max_win_streak = current_streak
            elif result == -1 and current_streak > max_loss_streak:
                max_loss_streak = current_streak
                
            last_result = result
            
        current_streak = current_streak if last_result != 0 else 0
        current_consecutive_wins = current_streak if last_result == 1 else 0
        current_consecutive_losses = current_streak if last_result == -1 else 0
        
        # Calculate expectancy
        expectancy = (win_rate/100 * avg_profit) - ((100-win_rate)/100 * avg_loss)
        
        # Direction win rates
        long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
        short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0
        
        # Average MAE and MFE
        avg_mae = total_mae / total_trades if total_trades > 0 else 0
        avg_mfe = total_mfe / total_trades if total_trades > 0 else 0
        
        return TradeStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            largest_profit=max_profit,
            largest_loss=max_loss,
            avg_holding_bars=avg_holding_period,
            avg_profit_bars=avg_profit_period,
            avg_loss_bars=avg_loss_period,
            consecutive_wins=current_consecutive_wins,
            consecutive_losses=current_consecutive_losses,
            max_consecutive_wins=max_win_streak,
            max_consecutive_losses=max_loss_streak,
            expectancy=expectancy,
            avg_mae=avg_mae,
            avg_mfe=avg_mfe,
            profit_per_trade=total_profit / total_trades,
            long_trades=long_trades,
            short_trades=short_trades,
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            trades_by_exit=exit_reasons,
            trades_by_day=trades_by_day,
            trades_by_hour=trades_by_hour
        )
    
    def _calculate_risk_metrics(self, 
                                returns: pd.Series, 
                                benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
            """
            Calculate risk-adjusted performance metrics.
            
            Args:
                returns: Series of strategy returns
                benchmark_returns: Optional series of benchmark returns
                
            Returns:
                RiskMetrics object
            """
            if len(returns) < 5:
                logger.warning("Not enough return data for accurate risk metrics")
                return RiskMetrics()
            
            # Calculate mean returns
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Calculate annualized measures (assuming daily returns)
            annual_factor = np.sqrt(252)
            annual_return = mean_return * 252
            annual_std = std_return * annual_factor
            
            # Sharpe Ratio
            excess_return = mean_return - self.daily_risk_free
            sharpe = excess_return / std_return * annual_factor if std_return > 0 else 0
            
            # Sortino Ratio (downside risk only)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino = excess_return / downside_std * annual_factor if downside_std > 0 else 0
            
            # Value at Risk (VaR)
            returns_sorted = sorted(returns)
            var_95 = np.percentile(returns_sorted, 5)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            
            # Skewness and Kurtosis
            skew = returns.skew() if len(returns) >= 3 else 0
            kurt = returns.kurtosis() if len(returns) >= 4 else 0
            
            # Omega Ratio
            threshold = self.daily_risk_free
            omega = returns[returns > threshold].sum() / abs(returns[returns < threshold].sum()) if len(returns[returns < threshold]) > 0 else float('inf')
            
            # Benchmark-related metrics (if provided)
            beta = 0.0
            alpha = 0.0
            treynor = 0.0
            information_ratio = 0.0
            r_squared = 0.0
            capture_ratio = 0.0
            upside_capture = 0.0
            downside_capture = 0.0
            
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                # Align data
                common_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
                if len(common_data) < 3:
                    logger.warning("Not enough aligned return data for benchmark comparison")
                else:
                    strat_returns = common_data.iloc[:, 0]
                    bench_returns = common_data.iloc[:, 1]
                    
                    # Beta calculation
                    cov_matrix = common_data.cov()
                    beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1] if cov_matrix.iloc[1, 1] > 0 else 0
                    
                    # Alpha calculation (Jensen's Alpha)
                    bench_return_mean = bench_returns.mean()
                    bench_excess = bench_return_mean - self.daily_risk_free
                    expected_return = self.daily_risk_free + beta * bench_excess
                    alpha = (strat_returns.mean() - expected_return) * 252  # Annualized
                    
                    # Treynor Ratio
                    treynor = excess_return / beta * 252 if beta > 0 else 0
                    
                    # Information Ratio
                    return_diff = strat_returns - bench_returns
                    tracking_error = return_diff.std() * annual_factor
                    information_ratio = (annual_return - bench_return_mean * 252) / tracking_error if tracking_error > 0 else 0
                    
                    # R-squared
                    correlation = strat_returns.corr(bench_returns)
                    r_squared = correlation ** 2
                    
                    # Capture ratios
                    up_market = bench_returns[bench_returns > 0]
                    down_market = bench_returns[bench_returns < 0]
                    
                    if len(up_market) > 0:
                        strat_up = strat_returns[bench_returns > 0]
                        upside_capture = strat_up.mean() / up_market.mean() * 100 if up_market.mean() != 0 else 0
                    
                    if len(down_market) > 0:
                        strat_down = strat_returns[bench_returns < 0]
                        downside_capture = strat_down.mean() / down_market.mean() * 100 if down_market.mean() != 0 else 0
                    
                    capture_ratio = upside_capture / downside_capture if downside_capture != 0 else float('inf')
            
            # Stability of timeseries (R-squared of linear regression of equity curve)
            equity_curve = (1 + returns).cumprod()
            x = np.arange(len(equity_curve))
            slope, _, r_value, _, _ = np.polyfit(x, equity_curve, 1, full=True)[0:5]
            stability = r_value ** 2
            
            # Tail ratio (absolute ratio of 95th/5th percentiles)
            tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
            
            return RiskMetrics(
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                treynor_ratio=treynor,
                information_ratio=information_ratio,
                omega_ratio=omega,
                capture_ratio=capture_ratio,
                upside_capture=upside_capture,
                downside_capture=downside_capture,
                beta=beta,
                alpha=alpha,
                r_squared=r_squared,
                value_at_risk_95=var_95,
                conditional_var_95=cvar_95,
                tail_ratio=tail_ratio,
                stability_of_timeseries=stability,
                skewness=skew,
                kurtosis=kurt
            )
    
    def analyze_equity_curve(self, 
                           equity_curve: pd.Series,
                           trades: Optional[List[Dict]] = None,
                           benchmark_equity: Optional[pd.Series] = None) -> PerformanceReport:
        """
        Analyze equity curve and generate comprehensive performance report.
        
        Args:
            equity_curve: Series of portfolio equity values
            trades: Optional list of completed trades
            benchmark_equity: Optional series of benchmark equity values
            
        Returns:
            PerformanceReport object
        """
        # Ensure equity is a Series with DatetimeIndex
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve = pd.Series(equity_curve.values, 
                                    index=pd.to_datetime(equity_curve.index))
                                    
        # Calculate return series
        returns = equity_curve.pct_change().dropna()
        
        # Calculate benchmark returns if provided
        benchmark_returns = None
        benchmark_return_pct = 0.0
        benchmark_volatility_pct = 0.0
        benchmark_max_drawdown_pct = 0.0
        
        if benchmark_equity is not None:
            if not isinstance(benchmark_equity.index, pd.DatetimeIndex):
                benchmark_equity = pd.Series(benchmark_equity.values,
                                          index=pd.to_datetime(benchmark_equity.index))
            
            # Calculate benchmark return series
            benchmark_returns = benchmark_equity.pct_change().dropna()
            
            # Calculate basic benchmark metrics
            benchmark_return_pct = (benchmark_equity.iloc[-1] / 
                                 benchmark_equity.iloc[0] - 1) * 100
            benchmark_volatility_pct = benchmark_returns.std() * np.sqrt(252) * 100
            
            # Calculate benchmark drawdown
            benchmark_dd = self._calculate_drawdowns(benchmark_equity)
            benchmark_max_drawdown_pct = abs(benchmark_dd['drawdown_pct'].min())
            
        # Calculate return statistics
        return_stats_dict = self._calculate_returns(equity_curve)
        
        # Create ReturnStats object
        return_stats = ReturnStats(
            total_return_pct=return_stats_dict['total_return_pct'],
            annual_return_pct=return_stats_dict['annual_return_pct'],
            daily_return_pct=return_stats_dict['daily_return_pct'],
            monthly_return_pct=return_stats_dict['monthly_return_pct'],
            volatility_annual_pct=return_stats_dict['volatility_annual_pct'],
            volatility_daily_pct=return_stats_dict['volatility_daily_pct'],
            downside_volatility_pct=return_stats_dict['downside_volatility_pct'],
            best_month_pct=return_stats_dict['best_month_pct'],
            worst_month_pct=return_stats_dict['worst_month_pct'],
            best_day_pct=return_stats_dict['best_day_pct'],
            worst_day_pct=return_stats_dict['worst_day_pct'],
            monthly_returns=return_stats_dict['monthly_returns'],
            avg_up_month_pct=return_stats_dict['avg_up_month_pct'],
            avg_down_month_pct=return_stats_dict['avg_down_month_pct'],
            pct_profitable_months=return_stats_dict['pct_profitable_months'],
            pct_profitable_days=return_stats_dict['pct_profitable_days']
        )
        
        # Calculate drawdown statistics
        dd_stats = self._calculate_drawdowns(equity_curve)
        
        # Get max drawdown
        max_dd_pct = abs(dd_stats['drawdown_pct'].min())
        
        # Calculate average drawdown stats
        drawdown_periods = dd_stats['drawdown_periods']
        avg_dd_pct = np.mean([d['max_drawdown_pct'] for d in drawdown_periods]) if drawdown_periods else 0
        avg_dd_duration = np.mean([d['duration_days'] for d in drawdown_periods]) if drawdown_periods else 0
        
        # Current drawdown
        current_dd_pct = abs(dd_stats['drawdown_pct'].iloc[-1]) if dd_stats['drawdown_pct'].iloc[-1] < 0 else 0
        
        # Current drawdown duration
        current_dd_duration = 0
        if current_dd_pct > 0:
            # Find start of current drawdown
            current_peak_idx = dd_stats['drawdown'].iloc[:-1].idxmax()
            current_dd_duration = (equity_curve.index[-1] - current_peak_idx).days
            
        # Calculate recovery factor and time to recovery
        time_to_recovery = 0
        recovery_factor = 0
        
        if max_dd_pct > 0 and len(drawdown_periods) > 0:
            # Find the maximum drawdown period
            max_dd_period = max(drawdown_periods, key=lambda x: abs(x['max_drawdown_pct']))
            
            if max_dd_period['end_date'] is not None:
                time_to_recovery = max_dd_period['recovery_days']
                trading_days = return_stats_dict['trading_days']
                recovery_factor = return_stats_dict['annual_return_pct'] / max_dd_pct if max_dd_pct > 0 else 0
        
        # Sterling ratio (uses average of 3 worst drawdowns)
        worst_drawdowns = sorted(drawdown_periods, key=lambda x: x['max_drawdown_pct'])[:3]
        avg_worst_dd = np.mean([d['max_drawdown_pct'] for d in worst_drawdowns]) if worst_drawdowns else max_dd_pct
        sterling_ratio = return_stats_dict['annual_return_pct'] / abs(avg_worst_dd) if abs(avg_worst_dd) > 0 else 0
        
        # Burke ratio (uses sum of squares of drawdowns)
        squared_dd_sum = np.sum([d['max_drawdown_pct']**2 for d in drawdown_periods]) if drawdown_periods else 0
        burke_ratio = return_stats_dict['annual_return_pct'] / np.sqrt(squared_dd_sum) if squared_dd_sum > 0 else 0
        
        # Create DrawdownStats object
        drawdown_stats = DrawdownStats(
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration=max([d['duration_days'] for d in drawdown_periods]) if drawdown_periods else 0,
            avg_drawdown_pct=avg_dd_pct,
            avg_drawdown_duration=avg_dd_duration,
            current_drawdown_pct=current_dd_pct,
            current_drawdown_duration=current_dd_duration,
            drawdowns=drawdown_periods,
            recovery_factor=recovery_factor,
            ulcer_index=dd_stats['ulcer_index'],
            pain_index=dd_stats['pain_index'],
            calmar_ratio=return_stats_dict['annual_return_pct'] / max_dd_pct if max_dd_pct > 0 else 0,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            time_to_recovery=time_to_recovery
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(returns, benchmark_returns)
        
        # Calculate trade statistics if trades provided
        trade_stats = self._calculate_trade_statistics(trades) if trades else TradeStats()
        
        # Create the full performance report
        report = PerformanceReport(
            trade_stats=trade_stats,
            return_stats=return_stats,
            drawdown_stats=drawdown_stats,
            risk_metrics=risk_metrics,
            backtest_period=(equity_curve.index[0], equity_curve.index[-1]),
            initial_capital=equity_curve.iloc[0],
            final_capital=equity_curve.iloc[-1],
            benchmark_return_pct=benchmark_return_pct,
            benchmark_volatility_pct=benchmark_volatility_pct,
            benchmark_max_drawdown_pct=benchmark_max_drawdown_pct
        )
        
        return report

    def compare_strategies(self, 
                         equity_curves: Dict[str, pd.Series],
                         benchmark_equity: Optional[pd.Series] = None) -> Dict[str, Dict]:
        """
        Compare multiple trading strategies.
        
        Args:
            equity_curves: Dictionary mapping strategy names to equity curves
            benchmark_equity: Optional benchmark equity curve
            
        Returns:
            Dictionary of performance metrics by strategy
        """
        results = {}
        
        for name, equity in equity_curves.items():
            report = self.analyze_equity_curve(equity, None, benchmark_equity)
            results[name] = report.to_dict()
            
        return results
    
    def calculate_correlation_matrix(self, 
                                   return_series: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix between different return series.
        
        Args:
            return_series: Dictionary mapping names to return series
            
        Returns:
            DataFrame with correlation matrix
        """
        # Create a DataFrame with all return series
        returns_df = pd.DataFrame(return_series)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def calculate_rolling_metrics(self,
                               equity_curve: pd.Series,
                               window: int = 60,
                               metrics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            equity_curve: Series of portfolio equity values
            window: Rolling window size in bars
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ['returns', 'volatility', 'sharpe', 'drawdown']
            
        if len(equity_curve) <= window:
            logger.warning(f"Equity curve length ({len(equity_curve)}) is shorter than window ({window})")
            return pd.DataFrame()
            
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Initialize result DataFrame
        rolling_metrics = pd.DataFrame(index=returns.index[window-1:])
        
        for i in range(window-1, len(returns)):
            window_returns = returns.iloc[i-window+1:i+1]
            window_equity = equity_curve.iloc[i-window+1:i+1]
            
            if 'returns' in metrics:
                rolling_metrics.loc[returns.index[i], 'return'] = \
                    (window_equity.iloc[-1] / window_equity.iloc[0] - 1) * 100
                    
            if 'volatility' in metrics:
                rolling_metrics.loc[returns.index[i], 'volatility'] = \
                    window_returns.std() * np.sqrt(252) * 100
                    
            if 'sharpe' in metrics:
                mean_ret = window_returns.mean()
                std_ret = window_returns.std()
                sharpe = (mean_ret - self.daily_risk_free) / std_ret * np.sqrt(252) if std_ret > 0 else 0
                rolling_metrics.loc[returns.index[i], 'sharpe'] = sharpe
                
            if 'drawdown' in metrics:
                window_dd = self._calculate_drawdowns(window_equity)
                rolling_metrics.loc[returns.index[i], 'drawdown'] = abs(window_dd['drawdown_pct'].min())
        
        return rolling_metrics