# src/backtesting/risk_analysis.py

"""
Monte Carlo simulation and risk analysis tools for the backtesting framework.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import seaborn as sns
import random
from scipy import stats

from src.backtesting.backtester import BacktestResult, Portfolio, Position

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """Monte Carlo simulation for trading strategy risk analysis."""
    
    def __init__(self, results_dir: str = "monte_carlo_results"):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            results_dir: Directory to save simulation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _bootstrap_trades(self, 
                         positions: List[Position], 
                         num_samples: int,
                         sample_pct: float = 0.8) -> List[List[Position]]:
        """
        Generate bootstrapped trade samples for Monte Carlo simulation.
        
        Args:
            positions: List of closed positions
            num_samples: Number of bootstrap samples to generate
            sample_pct: Percentage of trades to sample in each iteration
            
        Returns:
            List of position samples
        """
        closed_positions = [p for p in positions if p.status == "CLOSED"]
        if not closed_positions:
            logger.warning("No closed positions for bootstrapping")
            return []
            
        num_positions = len(closed_positions)
        sample_size = max(1, int(num_positions * sample_pct))
        
        samples = []
        for _ in range(num_samples):
            # Sample with replacement
            sample_positions = random.choices(closed_positions, k=sample_size)
            samples.append(sample_positions)
            
        return samples
    
    def _resample_trade_sequence(self,
                               positions: List[Position],
                               num_samples: int) -> List[List[Position]]:
        """
        Generate resampled trade sequences by shuffling order.
        
        Args:
            positions: List of closed positions
            num_samples: Number of resampled sequences to generate
            
        Returns:
            List of resampled position sequences
        """
        closed_positions = [p for p in positions if p.status == "CLOSED"]
        if not closed_positions:
            logger.warning("No closed positions for resampling")
            return []
            
        samples = []
        for _ in range(num_samples):
            # Create a copy and shuffle
            sample = closed_positions.copy()
            random.shuffle(sample)
            samples.append(sample)
            
        return samples
    
    def _reconstruct_equity_curve(self,
                               positions: List[Position],
                               initial_capital: float) -> pd.Series:
        """
        Reconstruct equity curve from a sequence of positions.
        
        Args:
            positions: List of positions in sequence
            initial_capital: Starting capital
            
        Returns:
            Series with equity curve
        """
        equity = [initial_capital]
        current_capital = initial_capital
        
        for pos in positions:
            if pos.status == "CLOSED":
                current_capital += pos.pnl
                equity.append(current_capital)
                
        return pd.Series(equity)
    
    def _simulate_returns(self,
                        returns: pd.Series,
                        num_simulations: int,
                        periods: int) -> np.ndarray:
        """
        Simulate future returns using bootstrap sampling.
        
        Args:
            returns: Historical return series
            num_simulations: Number of simulations to run
            periods: Number of periods to simulate
            
        Returns:
            Array of simulated equity curves
        """
        # Convert to numpy array for speed
        returns_array = returns.values
        
        # Initialize result array
        result = np.zeros((num_simulations, periods + 1))
        result[:, 0] = 1.0  # Start with $1
        
        # Run simulations
        for i in range(num_simulations):
            # Sample returns with replacement
            sampled_returns = np.random.choice(
                returns_array, size=periods, replace=True
            )
            
            # Calculate cumulative return
            result[i, 1:] = np.cumprod(1 + sampled_returns)
            
        return result
    
    def run_trade_simulation(self,
                           backtest_result: BacktestResult,
                           num_simulations: int = 1000,
                           sample_pct: float = 0.8,
                           method: str = 'bootstrap') -> Dict:
        """
        Run Monte Carlo simulation based on trade results.
        
        Args:
            backtest_result: Backtest result containing trades
            num_simulations: Number of simulations to run
            sample_pct: Percentage of trades to include in each sample
            method: Simulation method ('bootstrap' or 'shuffle')
            
        Returns:
            Dictionary with simulation results
        """
        # Extract positions and initial capital
        positions = backtest_result.portfolio.positions
        initial_capital = backtest_result.portfolio.initial_capital
        
        # Generate samples based on method
        if method == 'bootstrap':
            samples = self._bootstrap_trades(positions, num_simulations, sample_pct)
        elif method == 'shuffle':
            samples = self._resample_trade_sequence(positions, num_simulations)
        else:
            raise ValueError(f"Unknown simulation method: {method}")
            
        if not samples:
            logger.error("No samples generated for simulation")
            return {}
            
        # Reconstruct equity curves
        equity_curves = []
        final_capitals = []
        drawdowns = []
        
        for sample in samples:
            equity = self._reconstruct_equity_curve(sample, initial_capital)
            equity_curves.append(equity)
            final_capitals.append(equity.iloc[-1])
            
            # Calculate drawdown
            peak = equity.expanding().max()
            drawdown = ((equity - peak) / peak).min() * 100
            drawdowns.append(drawdown)
            
        # Calculate statistics
        final_capitals = np.array(final_capitals)
        drawdowns = np.array(drawdowns)
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        capital_percentiles = np.percentile(final_capitals, percentiles)
        drawdown_percentiles = np.percentile(drawdowns, percentiles)
        
        # Construct result dictionary
        result = {
            'method': method,
            'num_simulations': num_simulations,
            'sample_percentage': sample_pct,
            'initial_capital': initial_capital,
            'final_capital_mean': np.mean(final_capitals),
            'final_capital_median': np.median(final_capitals),
            'final_capital_std': np.std(final_capitals),
            'final_capital_percentiles': dict(zip(percentiles, capital_percentiles)),
            'drawdown_mean': np.mean(drawdowns),
            'drawdown_median': np.median(drawdowns),
            'drawdown_std': np.std(drawdowns),
            'drawdown_percentiles': dict(zip(percentiles, drawdown_percentiles)),
            'win_probability': np.mean(final_capitals > initial_capital)
        }
        
        # Plot results
        self._plot_simulation_results(
            equity_curves, 
            final_capitals, 
            drawdowns, 
            method,
            backtest_result
        )
        
        return result
    
    def run_returns_simulation(self,
                             backtest_result: BacktestResult,
                             num_simulations: int = 1000,
                             future_periods: int = 252,
                             confidence_interval: float = 0.95) -> Dict:
        """
        Run Monte Carlo simulation based on historical returns.
        
        Args:
            backtest_result: Backtest result with equity curve
            num_simulations: Number of simulations to run
            future_periods: Number of future periods to simulate
            confidence_interval: Confidence interval for projections
            
        Returns:
            Dictionary with simulation results
        """
        # Extract equity curve and calculate returns
        equity_data = pd.DataFrame(backtest_result.portfolio.equity_curve)
        if equity_data.empty:
            logger.error("No equity curve data in backtest result")
            return {}
            
        # Convert to Series and calculate returns
        equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'])
        equity_data.set_index('timestamp', inplace=True)
        equity = equity_data['equity']
        returns = equity.pct_change().dropna()
        
        if len(returns) < 20:
            logger.warning("Not enough return data for reliable simulation")
            
        # Run simulation
        simulation_data = self._simulate_returns(
            returns, num_simulations, future_periods
        )
        
        # Scale simulation data to match last equity value
        current_equity = equity.iloc[-1]
        simulation_data = simulation_data * current_equity
        
        # Calculate statistics
        final_values = simulation_data[:, -1]
        
        # Percentiles for confidence intervals
        lower_percentile = (1 - confidence_interval) / 2 * 100
        upper_percentile = (1 + confidence_interval) / 2 * 100
        percentiles = [1, 5, 10, lower_percentile, 50, upper_percentile, 90, 95, 99]
        final_percentiles = np.percentile(final_values, percentiles)
        
        # Calculate max drawdowns for each path
        drawdowns = np.zeros(num_simulations)
        for i in range(num_simulations):
            path = simulation_data[i, :]
            peak = np.maximum.accumulate(path)
            drawdown = ((path - peak) / peak).min() * 100
            drawdowns[i] = drawdown
            
        drawdown_percentiles = np.percentile(drawdowns, percentiles)
        
        # Calculate CAGR for each path
        years = future_periods / 252  # Assuming daily data
        cagrs = ((final_values / current_equity) ** (1 / years) - 1) * 100
        cagr_percentiles = np.percentile(cagrs, percentiles)
        
        # Construct result dictionary
        result = {
            'method': 'returns',
            'num_simulations': num_simulations,
            'future_periods': future_periods,
            'confidence_interval': confidence_interval,
            'current_equity': current_equity,
            'simulation_paths': simulation_data.tolist(),
            'final_value_mean': np.mean(final_values),
            'final_value_median': np.median(final_values),
            'final_value_std': np.std(final_values),
            'final_percentiles': dict(zip(percentiles, final_percentiles)),
            'cagr_mean': np.mean(cagrs),
            'cagr_median': np.median(cagrs),
            'cagr_std': np.std(cagrs),
            'cagr_percentiles': dict(zip(percentiles, cagr_percentiles)),
            'drawdown_mean': np.mean(drawdowns),
            'drawdown_median': np.median(drawdowns),
            'drawdown_percentiles': dict(zip(percentiles, drawdown_percentiles)),
            'profit_probability': np.mean(final_values > current_equity)
        }
        
        # Plot results
        self._plot_return_simulation(
            simulation_data, 
            final_values,
            drawdowns, 
            cagrs,
            equity,
            confidence_interval
        )
        
        return result
    
    def _plot_simulation_results(self,
                               equity_curves: List[pd.Series],
                               final_capitals: np.ndarray,
                               drawdowns: np.ndarray,
                               method: str,
                               backtest_result: BacktestResult,
                               save_path: Optional[str] = None):
        """
        Plot Monte Carlo simulation results.
        
        Args:
            equity_curves: List of simulated equity curves
            final_capitals: Array of final capital values
            drawdowns: Array of maximum drawdowns
            method: Simulation method used
            backtest_result: Original backtest result
            save_path: Optional path to save plot
        """
        # Set up plot
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)
        
        # Plot original equity curve
        original_equity = pd.DataFrame(backtest_result.portfolio.equity_curve)
        original_equity['timestamp'] = pd.to_datetime(original_equity['timestamp'])
        original_equity.set_index('timestamp', inplace=True)
        
        # Plot 1: Original equity curve
        ax1 = fig.add_subplot(gs[0, 0])
        original_equity['equity'].plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('Original Backtest Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        
        # Plot 2: Histogram of final capitals
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(final_capitals, bins=30, kde=True, ax=ax2)
        ax2.axvline(x=backtest_result.portfolio.initial_capital, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('Distribution of Final Capital')
        ax2.set_xlabel('Final Capital')
        ax2.set_ylabel('Frequency')
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        percs = np.percentile(final_capitals, percentiles)
        for i, p in enumerate(percs):
            ax2.axvline(x=p, color='g', alpha=0.5, linestyle=':')
            ax2.text(p, 0, f"{percentiles[i]}%", rotation=90, verticalalignment='bottom')
            
        # Plot 3: Histogram of drawdowns
        ax3 = fig.add_subplot(gs[1, 0])
        sns.histplot(drawdowns, bins=30, kde=True, ax=ax3)
        ax3.axvline(x=backtest_result.metrics['max_drawdown'], color='r', linestyle='--', alpha=0.7)
        ax3.set_title('Distribution of Maximum Drawdowns')
        ax3.set_xlabel('Maximum Drawdown (%)')
        ax3.set_ylabel('Frequency')
        
        # Plot 4: Random sample of equity curves
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot a sample of curves (e.g., 100 random ones)
        num_to_plot = min(100, len(equity_curves))
        sample_indices = random.sample(range(len(equity_curves)), num_to_plot)
        
        for i in sample_indices:
            ax4.plot(equity_curves[i], alpha=0.1, color='blue')
            
        ax4.set_title(f'Sample of {num_to_plot} Simulated Equity Curves')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Portfolio Value')
        ax4.grid(True)
        
        # Plot 5: Return vs Drawdown scatter
        ax5 = fig.add_subplot(gs[2, 0])
        returns = (final_capitals / backtest_result.portfolio.initial_capital - 1) * 100
        ax5.scatter(drawdowns, returns, alpha=0.5)
        ax5.set_title('Risk vs Return')
        ax5.set_xlabel('Maximum Drawdown (%)')
        ax5.set_ylabel('Total Return (%)')
        ax5.grid(True)
        
        # Add labels for quadrants
        x_mid = np.median(drawdowns)
        y_mid = np.median(returns)
        ax5.axhline(y=y_mid, color='r', linestyle='--', alpha=0.3)
        ax5.axvline(x=x_mid, color='r', linestyle='--', alpha=0.3)
        
        # Plot 6: Cumulative distribution of returns
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_returns = np.sort(returns)
        cumulative = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        ax6.plot(sorted_returns, cumulative)
        ax6.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        ax6.set_title('Cumulative Distribution of Returns')
        ax6.set_xlabel('Total Return (%)')
        ax6.set_ylabel('Cumulative Probability')
        ax6.grid(True)
        
        # Add annotation with statistics
        fig.subplots_adjust(bottom=0.15)
        plt.figtext(0.5, 0.02, 
                   f"Monte Carlo Simulation ({method}): {len(equity_curves)} runs\n"
                   f"Mean Final Capital: ${np.mean(final_capitals):,.2f} | "
                   f"Median: ${np.median(final_capitals):,.2f} | "
                   f"5% Worst Case: ${np.percentile(final_capitals, 5):,.2f}\n"
                   f"Mean Max Drawdown: {np.mean(drawdowns):.2f}% | "
                   f"Median: {np.median(drawdowns):.2f}% | "
                   f"5% Worst Case: {np.percentile(drawdowns, 95):.2f}%",
                   ha='center',
                   fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = self.results_dir / f"monte_carlo_{method}_{timestamp}.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def _plot_return_simulation(self,
                              simulation_data: np.ndarray,
                              final_values: np.ndarray,
                              drawdowns: np.ndarray,
                              cagrs: np.ndarray,
                              historical_equity: pd.Series,
                              confidence_interval: float,
                              save_path: Optional[str] = None):
        """
        Plot return-based Monte Carlo simulation.
        
        Args:
            simulation_data: Array of simulated equity paths
            final_values: Array of final equity values
            drawdowns: Array of maximum drawdowns
            cagrs: Array of compound annual growth rates
            historical_equity: Historical equity curve
            confidence_interval: Confidence interval used
            save_path: Optional path to save plot
        """
        # Calculate confidence bounds
        lower_pct = (1 - confidence_interval) / 2 * 100
        upper_pct = 100 - lower_pct
        
        # Create subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)
        
        # Plot 1: Future paths with confidence interval
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot historical data
        hist_x = np.arange(len(historical_equity))
        ax1.plot(hist_x, historical_equity.values, 'b-', linewidth=2, label='Historical')
        
        # Create future dates array
        future_x = np.arange(len(historical_equity), len(historical_equity) + simulation_data.shape[1])
        
        # Plot a sample of simulation paths
        num_to_plot = min(100, simulation_data.shape[0])
        random_indices = np.random.choice(simulation_data.shape[0], num_to_plot, replace=False)
        
        for i in random_indices:
            ax1.plot(future_x, simulation_data[i, :], 'b-', alpha=0.05)
            
        # Plot confidence intervals
        lower_bound = np.percentile(simulation_data, lower_pct, axis=0)
        median = np.percentile(simulation_data, 50, axis=0)
        upper_bound = np.percentile(simulation_data, upper_pct, axis=0)
        
        ax1.plot(future_x, median, 'g-', linewidth=2, label='Median')
        ax1.plot(future_x, lower_bound, 'r--', linewidth=1.5, 
                label=f'{lower_pct:.1f}th Percentile')
        ax1.plot(future_x, upper_bound, 'r--', linewidth=1.5,
                label=f'{upper_pct:.1f}th Percentile')
        
        ax1.axvline(x=len(historical_equity) - 1, color='k', linestyle='--')
        ax1.set_title(f'Monte Carlo Simulation: Projected Equity Paths ({len(simulation_data)} simulations)')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Histogram of final values
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(final_values, bins=30, kde=True, ax=ax2)
        ax2.axvline(x=historical_equity.iloc[-1], color='r', linestyle='--', alpha=0.7)
        
        # Add percentile lines
        percs = [5, 25, 50, 75, 95]
        perc_values = np.percentile(final_values, percs)
        for p, val in zip(percs, perc_values):
            ax2.axvline(x=val, color='g', alpha=0.5, linestyle=':')
            ax2.text(val, 0, f"{p}%", rotation=90, verticalalignment='bottom')
            
        ax2.set_title('Distribution of Final Portfolio Values')
        ax2.set_xlabel('Final Value')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Histogram of CAGRs
        ax3 = fig.add_subplot(gs[1, 1])
        sns.histplot(cagrs, bins=30, kde=True, ax=ax3)
        
        # Add mean and median lines
        mean_cagr = np.mean(cagrs)
        median_cagr = np.median(cagrs)
        ax3.axvline(x=mean_cagr, color='b', linestyle='-', alpha=0.7, label=f'Mean: {mean_cagr:.2f}%')
        ax3.axvline(x=median_cagr, color='g', linestyle='--', alpha=0.7, label=f'Median: {median_cagr:.2f}%')
        
        ax3.set_title('Distribution of Annual Growth Rates (CAGR)')
        ax3.set_xlabel('CAGR (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Plot 4: Histogram of Drawdowns
        ax4 = fig.add_subplot(gs[2, 0])
        sns.histplot(drawdowns, bins=30, kde=True, ax=ax4)
        
        # Add percentile lines
        dd_percs = np.percentile(drawdowns, percs)
        for p, val in zip(percs, dd_percs):
            ax4.axvline(x=val, color='g', alpha=0.5, linestyle=':')
            ax4.text(val, 0, f"{p}%", rotation=90, verticalalignment='bottom')
            
        ax4.set_title('Distribution of Maximum Drawdowns')
        ax4.set_xlabel('Maximum Drawdown (%)')
        ax4.set_ylabel('Frequency')
        
        # Plot 5: CAGR vs Drawdown scatter
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(drawdowns, cagrs, alpha=0.5)
        
        # Add quadrants
        x_mid = np.median(drawdowns)
        y_mid = np.median(cagrs)
        ax5.axhline(y=y_mid, color='r', linestyle='--', alpha=0.3)
        ax5.axvline(x=x_mid, color='r', linestyle='--', alpha=0.3)
        
        # Annotate quadrants
        ax5.text(min(drawdowns), y_mid, 'Better', ha='left', va='bottom')
        ax5.text(x_mid, max(cagrs), 'Better', ha='center', va='top')
        ax5.text(max(drawdowns), y_mid, 'Worse', ha='right', va='bottom')
        ax5.text(x_mid, min(cagrs), 'Worse', ha='center', va='bottom')
        
        ax5.set_title('Risk vs Return Trade-off')
        ax5.set_xlabel('Maximum Drawdown (%)')
        ax5.set_ylabel('CAGR (%)')
        ax5.grid(True)
        
        # Add summary statistics
        fig.subplots_adjust(bottom=0.15)
        plt.figtext(0.5, 0.02, 
                   f"Return-Based Monte Carlo: {len(simulation_data)} simulations\n"
                   f"Current Portfolio Value: ${historical_equity.iloc[-1]:,.2f} | "
                   f"Median Final Value: ${np.median(final_values):,.2f}\n"
                   f"CAGR - Median: {median_cagr:.2f}% | "
                   f"5% Worst: {np.percentile(cagrs, 5):.2f}% | "
                   f"5% Best: {np.percentile(cagrs, 95):.2f}%\n"
                   f"Probability of Profit: {100 * np.mean(final_values > historical_equity.iloc[-1]):.1f}%",
                   ha='center',
                   fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = self.results_dir / f"monte_carlo_returns_{timestamp}.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
            
        plt.show()

class RiskAnalyzer:
    """Risk analysis tools for trading strategies."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize risk analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    
    def calculate_var(self, 
                     returns: pd.Series,
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for VaR calculation
            method: Method for calculating VaR ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Value at Risk as a positive percentage
        """
        if len(returns) < 3:
            return 0.0
            
        if method == 'historical':
            # Historical VaR is simply the relevant percentile of historical returns
            var = np.percentile(returns, 100 * (1 - confidence_level))
            return abs(var) * 100  # Convert to positive percentage
            
        elif method == 'parametric':
            # Parametric VaR assumes normally distributed returns
            mean = returns.mean()
            std = returns.std()
            var = stats.norm.ppf(1 - confidence_level, mean, std)
            return abs(var) * 100
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR uses bootstrapped returns
            n_samples = 10000
            bootstrapped_returns = np.random.choice(returns, size=n_samples, replace=True)
            var = np.percentile(bootstrapped_returns, 100 * (1 - confidence_level))
            return abs(var) * 100
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def calculate_cvar(self,
                      returns: pd.Series,
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            Conditional Value at Risk as a positive percentage
        """
        if len(returns) < 3:
            return 0.0
            
        # Calculate VaR
        var_cutoff = np.percentile(returns, 100 * (1 - confidence_level))
        
        # CVaR is the mean of returns below VaR
        cvar = returns[returns <= var_cutoff].mean()
        return abs(cvar) * 100  # Convert to positive percentage
    
    def calculate_drawdowns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdowns from equity curve.
        
       Args:
            equity_curve: Series of portfolio equity values
            
        Returns:
            DataFrame with drawdown details
        """
        # Ensure equity is a Series with DatetimeIndex
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve = pd.Series(equity_curve.values, 
                                     index=pd.to_datetime(equity_curve.index))
        
        # Calculate expanding maximum
        peak = equity_curve.expanding().max()
        
        # Calculate drawdown in dollars
        drawdown = equity_curve - peak
        
        # Calculate drawdown in percentage
        drawdown_pct = drawdown / peak * 100
        
        # Find drawdown periods
        is_drawdown = drawdown_pct < 0
        
        # Identify drawdown periods
        drawdown_periods = []
        current_dd_start = None
        
        for date, value in is_drawdown.items():
            if value and current_dd_start is None:
                current_dd_start = date
            elif not value and current_dd_start is not None:
                # End of drawdown period
                drawdown_end = date
                
                # Find max drawdown in this period
                period_mask = (drawdown_pct.index >= current_dd_start) & \
                              (drawdown_pct.index <= drawdown_end)
                period_dd = drawdown_pct[period_mask]
                max_dd = period_dd.min()
                max_dd_date = period_dd.idxmin()
                
                # Calculate duration
                duration_days = (drawdown_end - current_dd_start).days
                recovery_days = (drawdown_end - max_dd_date).days
                
                drawdown_periods.append({
                    'start_date': current_dd_start,
                    'end_date': drawdown_end,
                    'max_drawdown_date': max_dd_date,
                    'max_drawdown_pct': max_dd,
                    'duration_days': duration_days,
                    'recovery_days': recovery_days
                })
                
                current_dd_start = None
                
        # Check if still in drawdown at the end
        if current_dd_start is not None:
            last_date = drawdown_pct.index[-1]
            period_mask = drawdown_pct.index >= current_dd_start
            period_dd = drawdown_pct[period_mask]
            max_dd = period_dd.min()
            max_dd_date = period_dd.idxmin()
            duration_days = (last_date - current_dd_start).days
            
            drawdown_periods.append({
                'start_date': current_dd_start,
                'end_date': None,  # Still ongoing
                'max_drawdown_date': max_dd_date,
                'max_drawdown_pct': max_dd,
                'duration_days': duration_days,
                'recovery_days': None
            })
            
        # Create a DataFrame from drawdown periods
        if drawdown_periods:
            drawdowns_df = pd.DataFrame(drawdown_periods)
            
            # Add some aggregate statistics
            drawdowns_df = drawdowns_df.sort_values('max_drawdown_pct')
            
            # Calculate running underwater periods
            underwater_days = sum(1 for x in drawdown_pct if x < 0)
            underwater_pct = (underwater_days / len(drawdown_pct)) * 100 if len(drawdown_pct) > 0 else 0
            
            # Add metadata
            metadata = {
                'total_drawdowns': len(drawdown_periods),
                'max_drawdown_pct': drawdowns_df['max_drawdown_pct'].min() if not drawdowns_df.empty else 0,
                'avg_drawdown_pct': drawdowns_df['max_drawdown_pct'].mean() if not drawdowns_df.empty else 0,
                'median_drawdown_pct': drawdowns_df['max_drawdown_pct'].median() if not drawdowns_df.empty else 0,
                'avg_duration_days': drawdowns_df['duration_days'].mean() if not drawdowns_df.empty else 0,
                'max_duration_days': drawdowns_df['duration_days'].max() if not drawdowns_df.empty else 0,
                'underwater_days': underwater_days,
                'underwater_pct': underwater_pct,
                'drawdown': drawdown,
                'drawdown_pct': drawdown_pct
            }
            
            return {
                'drawdowns': drawdowns_df,
                'metadata': metadata
            }
        else:
            # No drawdowns found
            return {
                'drawdowns': pd.DataFrame(),
                'metadata': {
                    'total_drawdowns': 0,
                    'max_drawdown_pct': 0,
                    'avg_drawdown_pct': 0,
                    'median_drawdown_pct': 0,
                    'avg_duration_days': 0,
                    'max_duration_days': 0,
                    'underwater_days': 0,
                    'underwater_pct': 0,
                    'drawdown': drawdown,
                    'drawdown_pct': drawdown_pct
                }
            }
            
    def calculate_stress_test(self,
                            returns: pd.Series,
                            scenarios: Dict[str, float] = None) -> Dict[str, float]:
        """
        Perform stress testing by simulating extreme market events.
        
        Args:
            returns: Series of historical returns
            scenarios: Dictionary mapping scenario names to return drops
            
        Returns:
            Dictionary with stress test results
        """
        if scenarios is None:
            # Default stress scenarios based on historical market crashes
            scenarios = {
                'market_crash_2008': -0.40,    # 2008 financial crisis
                'dot_com_crash': -0.30,       # 2000-2002 dot-com crash
                'covid_crash': -0.35,         # March 2020 COVID crash
                'black_monday': -0.20,        # 1987 Black Monday
                'moderate_correction': -0.15,  # Typical market correction
                'mild_correction': -0.10      # Mild market pullback
            }
        
        current_value = 1.0  # Start with $1
        
        # Calculate portfolio value after each scenario
        results = {}
        for scenario, drop in scenarios.items():
            # Calculate new value after drop
            new_value = current_value * (1 + drop)
            
            # Estimate recovery time based on historical returns
            if len(returns) > 20:
                positive_returns = returns[returns > 0]
                avg_daily_gain = positive_returns.mean() if len(positive_returns) > 0 else 0.001
                
                # Estimate days to recover
                if avg_daily_gain > 0:
                    recovery_pct_needed = -drop / (1 + drop)  # Percentage gain needed to recover
                    estimated_days = recovery_pct_needed / avg_daily_gain
                else:
                    estimated_days = float('inf')
                    
                # Convert to trading days (approx. 252 per year)
                estimated_years = estimated_days / 252
                recovery_estimate = estimated_years
            else:
                recovery_estimate = None
                
            # Store results
            results[scenario] = {
                'impact_pct': drop * 100,
                'new_value': new_value,
                'drawdown_pct': drop * 100,
                'estimated_recovery_years': recovery_estimate
            }
            
        return results
        
    def kelly_criterion(self,
                      win_rate: float,
                      win_loss_ratio: float,
                      adjust_factor: float = 0.5) -> float:
        """
        Calculate the optimal position size using the Kelly Criterion.
        
        Args:
            win_rate: Probability of winning (0.0-1.0)
            win_loss_ratio: Ratio of average win to average loss
            adjust_factor: Conservative adjustment (usually 0.5 for "half-Kelly")
            
        Returns:
            Recommended position size as fraction of capital
        """
        if win_loss_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
            
        # Kelly formula: f = (p*b - q)/b where p=win_rate, q=loss_rate, b=win_loss_ratio
        loss_rate = 1 - win_rate
        kelly_percentage = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Apply adjustment factor for more conservative sizing
        adjusted_kelly = kelly_percentage * adjust_factor
        
        # Ensure result is non-negative
        return max(0, adjusted_kelly)
        
    def analyze_position_sizing(self,
                              trades: List[Dict],
                              capital: float,
                              risk_per_trade_pct: float = 2.0,
                              method: str = 'fixed_risk') -> Dict:
        """
        Analyze position sizing strategies based on historical trades.
        
        Args:
            trades: List of trade dictionaries
            capital: Starting capital
            risk_per_trade_pct: Percentage of capital to risk per trade
            method: Position sizing method ('fixed_risk', 'fixed_fractional', 'kelly')
            
        Returns:
            Dictionary with position sizing analysis
        """
        if not trades:
            return {
                'method': method,
                'risk_per_trade_pct': risk_per_trade_pct,
                'average_position_size': 0,
                'max_position_size': 0,
                'min_position_size': 0,
                'position_size_as_pct_of_capital': 0,
                'max_drawdown_pct': 0,
                'final_capital': capital,
                'roi_pct': 0
            }
            
        # Calculate win rate and win/loss ratio for Kelly
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        breakeven_trades = [t for t in trades if t.get('pnl', 0) == 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.get('pnl', 0)) for t in losing_trades]) if losing_trades else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Simulate equity curve with different position sizing
        equity_curve = [capital]
        current_capital = capital
        position_sizes = []
        
        for trade in trades:
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            direction = 1 if trade.get('direction', 'LONG') == 'LONG' else -1
            
            # Calculate position size based on method
            if method == 'fixed_risk':
                # Risk a fixed percentage of capital on each trade
                if 'stop_loss' in trade and trade['stop_loss'] and entry_price:
                    risk_per_unit = abs(entry_price - trade['stop_loss'])
                    if risk_per_unit > 0:
                        units = (current_capital * (risk_per_trade_pct / 100)) / risk_per_unit
                    else:
                        units = current_capital * 0.1 / entry_price  # Fallback
                else:
                    # If no stop loss, use a default 10% of capital
                    units = current_capital * 0.1 / entry_price if entry_price > 0 else 0
                    
            elif method == 'fixed_fractional':
                # Invest a fixed percentage of capital
                units = (current_capital * (risk_per_trade_pct / 100)) / entry_price if entry_price > 0 else 0
                
            elif method == 'kelly':
                # Use Kelly criterion
                kelly_pct = self.kelly_criterion(win_rate, win_loss_ratio, 0.5)
                units = (current_capital * kelly_pct) / entry_price if entry_price > 0 else 0
                
            else:
                raise ValueError(f"Unknown position sizing method: {method}")
                
            # Calculate position value
            position_value = units * entry_price
            position_sizes.append(position_value)
            
            # Calculate profit/loss
            if exit_price and entry_price:
                price_diff = (exit_price - entry_price) * direction
                trade_pnl = price_diff * units
                current_capital += trade_pnl
                equity_curve.append(current_capital)
                
        # Calculate drawdown
        equity_series = pd.Series(equity_curve)
        max_equity = equity_series.expanding().max()
        drawdown = (equity_series - max_equity) / max_equity * 100
        max_drawdown = abs(drawdown.min())
        
        # Calculate position sizing stats
        avg_position_size = np.mean(position_sizes) if position_sizes else 0
        max_position_size = max(position_sizes) if position_sizes else 0
        min_position_size = min(position_sizes) if position_sizes else 0
        
        # Calculate ROI
        roi = (current_capital / capital - 1) * 100
        
        return {
            'method': method,
            'risk_per_trade_pct': risk_per_trade_pct,
            'win_rate': win_rate * 100,
            'win_loss_ratio': win_loss_ratio,
            'kelly_percentage': self.kelly_criterion(win_rate, win_loss_ratio, 1.0) * 100,
            'half_kelly_percentage': self.kelly_criterion(win_rate, win_loss_ratio, 0.5) * 100,
            'average_position_size': avg_position_size,
            'max_position_size': max_position_size,
            'min_position_size': min_position_size,
            'position_size_as_pct_of_capital': avg_position_size / capital * 100 if capital > 0 else 0,
            'max_drawdown_pct': max_drawdown,
            'final_capital': current_capital,
            'roi_pct': roi,
            'equity_curve': equity_curve
        }
        
    def calculate_risk_metrics(self,
                             returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Optional series of benchmark returns
            
        Returns:
            Dictionary with risk metrics
        """
        if len(returns) < 5:
            logger.warning("Not enough data for reliable risk metrics")
            return {}
            
        # Basic stats
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Annualization factors (assuming daily returns)
        annual_factor = np.sqrt(252)
        annual_return = mean_return * 252
        annual_volatility = volatility * annual_factor
        
        # Sharpe ratio
        excess_return = mean_return - self.daily_risk_free
        sharpe = excess_return / volatility * annual_factor if volatility > 0 else 0
        
        # Sortino ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = excess_return / downside_volatility * annual_factor if downside_volatility > 0 else 0
        
        # Calculate max drawdown
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calculate VaR and CVaR
        var_95 = self.calculate_var(returns, 0.95, 'historical')
        var_99 = self.calculate_var(returns, 0.99, 'historical')
        cvar_95 = self.calculate_cvar(returns, 0.95)
        
        # Calculate skewness and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Calculate win rate and win/loss ratio
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = abs(losing_days.mean()) if len(losing_days) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Calculate benchmark metrics if provided
        alpha = 0
        beta = 0
        r_squared = 0
        tracking_error = 0
        information_ratio = 0
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Ensure both series are aligned
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) > 1:
                strat_returns = aligned_data.iloc[:, 0]
                bench_returns = aligned_data.iloc[:, 1]
                
                # Beta calculation
                covariance = strat_returns.cov(bench_returns)
                benchmark_variance = bench_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Alpha calculation
                bench_mean = bench_returns.mean()
                alpha = (mean_return - self.daily_risk_free) - beta * (bench_mean - self.daily_risk_free)
                alpha = alpha * 252  # Annualize
                
                # R-squared
                correlation = strat_returns.corr(bench_returns)
                r_squared = correlation ** 2
                
                # Tracking error and information ratio
                tracking_diff = strat_returns - bench_returns
                tracking_error = tracking_diff.std() * annual_factor
                information_ratio = (annual_return - bench_mean * 252) / tracking_error if tracking_error > 0 else 0
                
        # Compile all metrics
        return {
            'annual_return': annual_return * 100,
            'annual_volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown * 100,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'win_rate': win_rate * 100,
            'win_loss_ratio': win_loss_ratio,
            'skewness': skew,
            'kurtosis': kurt,
            'alpha': alpha * 100,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': tracking_error * 100 if tracking_error != 0 else 0,
            'information_ratio': information_ratio
        }
        
    def risk_contribution_analysis(self,
                                 returns_dict: Dict[str, pd.Series]) -> Dict:
        """
        Analyze risk contribution of different components.
        
        Args:
            returns_dict: Dictionary mapping component names to return series
            
        Returns:
            Dictionary with risk contribution analysis
        """
        # Convert returns to DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        if returns_df.empty or returns_df.shape[1] < 2:
            return {'error': 'Need at least two return series for risk contribution analysis'}
            
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        # Calculate portfolio volatility
        weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]  # Equal weights
        port_var = weights.dot(cov_matrix).dot(weights)
        port_vol = np.sqrt(port_var)
        
        # Calculate marginal contribution to risk
        mcr = cov_matrix.dot(weights) / port_vol
        
        # Calculate component contribution to risk
        ccr = np.multiply(weights, mcr)
        
        # Calculate percentage contribution to risk
        pcr = ccr / port_vol
        
        # Create result dictionary
        components = list(returns_dict.keys())
        result = {
            'portfolio_volatility': port_vol * 100,
            'marginal_contribution': dict(zip(components, mcr * 100)),
            'component_contribution': dict(zip(components, ccr * 100)),
            'percentage_contribution': dict(zip(components, pcr * 100)),
            'correlation_matrix': returns_df.corr().to_dict(),
            'covariance_matrix': cov_matrix.to_dict()
        }
        
        return result
        
    def plot_risk_metrics(self,
                        returns: pd.Series,
                        benchmark_returns: Optional[pd.Series] = None,
                        rolling_window: int = 60,
                        save_path: Optional[str] = None):
        """
        Plot risk metrics over time.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns
            rolling_window: Window size for rolling calculations
            save_path: Optional path to save plot
        """
        if len(returns) < rolling_window:
            logger.warning(f"Not enough data for rolling window of {rolling_window}")
            return
            
        # Create rolling metrics
        rolling_metrics = pd.DataFrame(index=returns.index[rolling_window-1:])
        
        # Calculate rolling returns
        rolling_metrics['return'] = returns.rolling(rolling_window).mean() * 252 * 100
        rolling_metrics['volatility'] = returns.rolling(rolling_window).std() * np.sqrt(252) * 100
        
        # Calculate rolling Sharpe ratio
        rolling_excess = returns - self.daily_risk_free
        rolling_metrics['sharpe'] = (rolling_excess.rolling(rolling_window).mean() / 
                                    returns.rolling(rolling_window).std()) * np.sqrt(252)
        
        # Calculate rolling Sortino ratio
        downside = returns.copy()
        downside[downside > 0] = 0
        rolling_metrics['sortino'] = (rolling_excess.rolling(rolling_window).mean() / 
                                     downside.rolling(rolling_window).std()) * np.sqrt(252)
        
        # Calculate rolling max drawdown
        equity_curve = (1 + returns).cumprod()
        rolling_dd = []
        
        for i in range(rolling_window, len(returns) + 1):
            window_equity = equity_curve.iloc[i-rolling_window:i]
            peak = window_equity.expanding().max()
            drawdown = (window_equity - peak) / peak
            max_dd = drawdown.min() * 100
            rolling_dd.append(max_dd)
            
        rolling_metrics['max_drawdown'] = rolling_dd
        
        # Calculate rolling beta if benchmark provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            betas = []
            r_squareds = []
            alphas = []
            
            for i in range(rolling_window, len(returns) + 1):
                strat_window = returns.iloc[i-rolling_window:i]
                bench_window = benchmark_returns.iloc[i-rolling_window:i]
                
                # Calculate beta
                cov = strat_window.cov(bench_window)
                var = bench_window.var()
                beta = cov / var if var > 0 else 0
                betas.append(beta)
                
                # Calculate R-squared
                corr = strat_window.corr(bench_window)
                r_squared = corr ** 2 if not np.isnan(corr) else 0
                r_squareds.append(r_squared)
                
                # Calculate alpha
                strat_mean = strat_window.mean()
                bench_mean = bench_window.mean()
                alpha = (strat_mean - self.daily_risk_free) - beta * (bench_mean - self.daily_risk_free)
                alphas.append(alpha * 252 * 100)  # Annualized percentage
                
            rolling_metrics['beta'] = betas
            rolling_metrics['r_squared'] = r_squareds
            rolling_metrics['alpha'] = alphas
            
        # Plot metrics
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Format dates if index is datetime
        if isinstance(rolling_metrics.index, pd.DatetimeIndex):
            date_formatter = plt.matplotlib.dates.DateFormatter('%Y-%m')
            for ax in axes.flat:
                ax.xaxis.set_major_formatter(date_formatter)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot return and volatility
        rolling_metrics[['return', 'volatility']].plot(ax=axes[0, 0])
        axes[0, 0].set_title(f'Rolling {rolling_window}-bar Return and Volatility')
        axes[0, 0].set_ylabel('Percent')
        axes[0, 0].grid(True)
        
        # Plot Sharpe and Sortino ratios
        rolling_metrics[['sharpe', 'sortino']].plot(ax=axes[0, 1])
        axes[0, 1].set_title(f'Rolling {rolling_window}-bar Sharpe and Sortino Ratios')
        axes[0, 1].grid(True)
        
        # Plot max drawdown
        rolling_metrics['max_drawdown'].plot(ax=axes[1, 0], color='red')
        axes[1, 0].set_title(f'Rolling {rolling_window}-bar Maximum Drawdown')
        axes[1, 0].set_ylabel('Percent')
        axes[1, 0].grid(True)
        
        # Plot benchmark metrics if available
        if 'beta' in rolling_metrics.columns:
            rolling_metrics['beta'].plot(ax=axes[1, 1])
            axes[1, 1].set_title(f'Rolling {rolling_window}-bar Beta')
            axes[1, 1].grid(True)
            
            rolling_metrics['alpha'].plot(ax=axes[2, 0])
            axes[2, 0].set_title(f'Rolling {rolling_window}-bar Alpha')
            axes[2, 0].set_ylabel('Percent')
            axes[2, 0].grid(True)
            
            rolling_metrics['r_squared'].plot(ax=axes[2, 1])
            axes[2, 1].set_title(f'Rolling {rolling_window}-bar R-Squared')
            axes[2, 1].grid(True)
        else:
            # If no benchmark, plot other relevant metrics
            # Calculate rolling VaR
            var_values = []
            for i in range(rolling_window, len(returns) + 1):
                window_returns = returns.iloc[i-rolling_window:i]
                var = self.calculate_var(window_returns, 0.95, 'historical')
                var_values.append(var)
                
            rolling_metrics['VaR_95'] = var_values
            rolling_metrics['VaR_95'].plot(ax=axes[2, 0], color='purple')
            axes[2, 0].set_title(f'Rolling {rolling_window}-bar VaR (95%)')
            axes[2, 0].set_ylabel('Percent')
            axes[2, 0].grid(True)
            
            # Calculate win rate
            win_rates = []
            for i in range(rolling_window, len(returns) + 1):
                window_returns = returns.iloc[i-rolling_window:i]
                wins = (window_returns > 0).sum()
                win_rate = wins / len(window_returns) * 100
                win_rates.append(win_rate)
                
            rolling_metrics['win_rate'] = win_rates
            rolling_metrics['win_rate'].plot(ax=axes[2, 1], color='green')
            axes[2, 1].set_title(f'Rolling {rolling_window}-bar Win Rate')
            axes[2, 1].set_ylabel('Percent')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()