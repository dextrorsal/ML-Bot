# src/backtesting/optimizer.py

"""
Strategy optimization framework for the Ultimate Data Fetcher.
"""

import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Union, Tuple, Callable, Any, Optional
import concurrent.futures
import multiprocessing
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.models import TimeRange
from src.storage.processed import ProcessedDataStorage
from src.utils.indicators.base_indicator import BaseIndicator
from src.utils.strategy.base import BaseStrategy
from src.backtesting.backtester import Backtester, BacktestResult

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """Optimize strategy parameters using grid search."""
    
    def __init__(self, 
                backtester: Backtester,
                data_storage: Optional[ProcessedDataStorage] = None,
                results_dir: str = "optimization_results"):
        """
        Initialize optimizer.
        
        Args:
            backtester: Backtester instance
            data_storage: Optional data storage for loading data
            results_dir: Directory to save optimization results
        """
        self.backtester = backtester
        self.data_storage = data_storage
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    async def load_data(self, exchange: str, market: str, 
                       resolution: str, start_time: datetime, 
                       end_time: datetime) -> pd.DataFrame:
        """
        Load data for optimization.
        
        Args:
            exchange: Exchange name
            market: Market symbol
            resolution: Candle resolution
            start_time: Start time
            end_time: End time
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.data_storage is None:
            raise ValueError("No data storage configured")
            
        return await self.data_storage.load_candles(
            exchange=exchange,
            market=market,
            resolution=resolution,
            start_time=start_time,
            end_time=end_time
        )
        
    def _generate_parameter_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """
        Generate all combinations of parameters from param_grid.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter dictionaries
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _run_single_backtest(self, 
                           df: pd.DataFrame,
                           strategy_class: Union[type[BaseStrategy], type[BaseIndicator]],
                           strategy_params: Dict,
                           backtest_params: Dict) -> Tuple[Dict, BacktestResult]:
        """
        Run a single backtest with specific parameters.
        
        Args:
            df: DataFrame with OHLCV data
            strategy_class: Strategy or indicator class
            strategy_params: Parameters for strategy initialization
            backtest_params: Parameters for backtester
            
        Returns:
            Tuple of (combined parameters, backtest result)
        """
        strategy = strategy_class(**strategy_params)
        result = self.backtester.run_backtest(df, strategy, **backtest_params)
        return {**strategy_params, **backtest_params}, result
    
    def optimize(self,
               df: pd.DataFrame,
               strategy_class: Union[type[BaseStrategy], type[BaseIndicator]],
               strategy_param_grid: Dict[str, List],
               backtest_param_grid: Optional[Dict[str, List]] = None,
               metric_name: str = "sharpe_ratio",
               higher_is_better: bool = True,
               max_workers: Optional[int] = None,
               test_name: str = None,
               strategy_config_path: Optional[str] = None) -> Dict:
        """
        Run grid search optimization.
        
        Args:
            df: DataFrame with OHLCV data
            strategy_class: Strategy or indicator class
            strategy_param_grid: Grid of strategy parameters to search
            backtest_param_grid: Optional grid of backtest parameters
            metric_name: Name of metric to optimize
            higher_is_better: Whether higher metric values are better
            max_workers: Maximum number of parallel workers
            test_name: Name of the optimization run
            strategy_config_path: Optional path to strategy configuration file
            
        Returns:
            Dictionary with optimization results
        """
        if backtest_param_grid is None:
            backtest_param_grid = {}
            
        # Default backtest params
        default_backtest_params = {
            "initial_capital": 10000.0,
            "position_size_pct": 10.0,
            "commission_pct": 0.1,
            "slippage_pct": 0.05,
            "max_positions": 1
        }
        
        # Merge default and provided backtest params
        for key, value in default_backtest_params.items():
            if key not in backtest_param_grid:
                backtest_param_grid[key] = [value]
                
        # Generate all parameter combinations
        strategy_combinations = self._generate_parameter_combinations(strategy_param_grid)
        backtest_combinations = self._generate_parameter_combinations(backtest_param_grid)
        
        logger.info(f"Running optimization with {len(strategy_combinations)} strategy combinations "
                   f"and {len(backtest_combinations)} backtest combinations")
        
        all_results = []
        
        # Use process-based parallelism if many combinations
        total_combinations = len(strategy_combinations) * len(backtest_combinations)
        use_parallel = total_combinations > 10 and max_workers != 1
        
        if use_parallel and max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            
        if use_parallel:
            # Process combinations in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for strategy_params in strategy_combinations:
                    for backtest_params in backtest_combinations:
                        # Add config_path if provided
                        strategy_params_with_config = strategy_params.copy()
                        if strategy_config_path:
                            strategy_params_with_config['config_path'] = strategy_config_path
                            
                        futures.append(
                            executor.submit(
                                self._run_single_backtest,
                                df,
                                strategy_class,
                                strategy_params_with_config,  # Use modified params
                                backtest_params
                            )
                        )
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        params, result = future.result()
                        all_results.append({
                            "params": params,
                            "metrics": result.metrics,
                            "result": result
                        })
                        logger.info(f"Completed optimization run with params: {params}")
                    except Exception as e:
                        logger.error(f"Error in optimization run: {str(e)}")
        else:
            # Run sequentially
            for strategy_params in strategy_combinations:
                for backtest_params in backtest_combinations:
                    try:
                        # Add config_path if provided 
                        strategy_params_with_config = strategy_params.copy()
                        if strategy_config_path:
                            strategy_params_with_config['config_path'] = strategy_config_path
                            
                        params, result = self._run_single_backtest(
                            df,
                            strategy_class,
                            strategy_params_with_config,  # Use modified params
                            backtest_params
                        )
                        all_results.append({
                            "params": params,
                            "metrics": result.metrics,
                            "result": result
                        })
                        logger.info(f"Completed optimization run with params: {params}")
                    except Exception as e:
                        logger.error(f"Error in optimization run: {str(e)}")
        
        # Sort results by the target metric
        all_results.sort(
            key=lambda x: x["metrics"][metric_name], 
            reverse=higher_is_better
        )
        
        # Generate timestamp for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_id = test_name or f"optimization_{timestamp}"
        
        # Save results
        result_summary = {
            "test_id": test_id,
            "timestamp": timestamp,
            "strategy_class": strategy_class.__name__,
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "total_combinations": total_combinations,
            "best_params": all_results[0]["params"] if all_results else None,
            "best_metrics": all_results[0]["metrics"] if all_results else None,
            "all_results": [{"params": r["params"], "metrics": r["metrics"]} for r in all_results]
        }
        
        # Save full results
        results_path = self.results_dir / f"{test_id}.json"
        with open(results_path, 'w') as f:
            json.dump(result_summary, f, indent=2, default=str)
            
        # If we have a best result, save it separately
        if all_results:
            best_result = all_results[0]["result"]
            best_result_path = self.results_dir / f"{test_id}_best.json"
            best_result.save_results(str(best_result_path))
            
            # Plot best result
            plot_path = self.results_dir / f"{test_id}_best_plot.png"
            best_result.plot_results(str(plot_path))
            
        logger.info(f"Optimization complete. Best {metric_name}: "
                   f"{all_results[0]['metrics'][metric_name] if all_results else 'N/A'}")
                   
        return result_summary
        
    def plot_parameter_impact(self, results: Dict, parameter: str,
                             metric: str = "sharpe_ratio", 
                             save_path: Optional[str] = None):
        """
        Plot the impact of a parameter on performance.
        
        Args:
            results: Optimization results
            parameter: Parameter to analyze
            metric: Metric to plot
            save_path: Optional path to save plot
        """
        if not results.get("all_results"):
            logger.warning("No results to plot")
            return
            
        # Extract parameter values and metrics
        param_values = []
        metric_values = []
        
        for result in results["all_results"]:
            if parameter in result["params"]:
                param_values.append(result["params"][parameter])
                metric_values.append(result["metrics"][metric])
                
        if not param_values:
            logger.warning(f"Parameter {parameter} not found in results")
            return
            
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "parameter": param_values,
            "metric": metric_values
        })
        
        # Handle numeric and categorical parameters differently
        is_numeric = all(isinstance(x, (int, float)) for x in param_values)
        
        plt.figure(figsize=(12, 6))
        
        if is_numeric:
            # For numeric parameters, use scatter plot with trendline
            plt.scatter(df["parameter"], df["metric"], alpha=0.6)
            
            # Add trendline if more than 1 point
            if len(df) > 1:
                z = np.polyfit(df["parameter"], df["metric"], 1)
                p = np.poly1d(z)
                plt.plot(df["parameter"], p(df["parameter"]), "r--", alpha=0.8)
                
            plt.xlabel(parameter)
            plt.ylabel(metric)
            
        else:
            # For categorical parameters, use boxplot
            plt.figure(figsize=(max(8, len(set(param_values))*1.5), 6))
            sns.boxplot(x="parameter", y="metric", data=df)
            plt.xticks(rotation=45)
            plt.xlabel(parameter)
            plt.ylabel(metric)
            
        plt.title(f"Impact of {parameter} on {metric}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_parameter_heatmap(self, results: Dict, 
                              param_x: str, param_y: str,
                              metric: str = "sharpe_ratio",
                              save_path: Optional[str] = None):
        """
        Plot heatmap showing interaction between two parameters.
        
        Args:
            results: Optimization results
            param_x: First parameter
            param_y: Second parameter
            metric: Metric to plot
            save_path: Optional path to save plot
        """
        if not results.get("all_results"):
            logger.warning("No results to plot")
            return
            
        # Extract parameter values and metrics
        data = []
        for result in results["all_results"]:
            if param_x in result["params"] and param_y in result["params"]:
                data.append({
                    param_x: result["params"][param_x],
                    param_y: result["params"][param_y],
                    "metric": result["metrics"][metric]
                })
                
        if not data:
            logger.warning(f"Parameters {param_x} and/or {param_y} not found in results")
            return
            
        df = pd.DataFrame(data)
        
        # Check if both parameters are numeric
        x_numeric = all(isinstance(x, (int, float)) for x in df[param_x])
        y_numeric = all(isinstance(y, (int, float)) for y in df[param_y])
        
        if x_numeric and y_numeric and len(df[param_x].unique()) > 1 and len(df[param_y].unique()) > 1:
            # If both params are numeric with multiple values, create a pivot table
            pivot = df.pivot_table(index=param_y, columns=param_x, values="metric", aggfunc='mean')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title(f"Heatmap of {metric} by {param_x} and {param_y}")
            
        else:
            # Otherwise use a grouped bar chart
            plt.figure(figsize=(max(8, len(df[param_x].unique())*1.5), 6))
            sns.barplot(x=param_x, y="metric", hue=param_y, data=df)
            plt.title(f"Impact of {param_x} and {param_y} on {metric}")
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

    async def walk_forward_optimization(self,
                                exchange: str,
                                market: str,
                                resolution: str,
                                strategy_class: Union[type[BaseStrategy], type[BaseIndicator]],
                                strategy_param_grid: Dict[str, List],
                                backtest_param_grid: Optional[Dict[str, List]] = None,
                                train_days: int = 180,
                                test_days: int = 60,
                                step_days: int = 30,
                                start_date: datetime = None,
                                end_date: datetime = None,
                                metric_name: str = "sharpe_ratio",
                                higher_is_better: bool = True,
                                max_workers: Optional[int] = None,
                                strategy_config_path: Optional[str] = None) -> Dict:
        """
        Perform walk-forward optimization.
        
        Args:
            exchange: Exchange name
            market: Market symbol
            resolution: Candle resolution
            strategy_class: Strategy or indicator class
            strategy_param_grid: Grid of strategy parameters
            backtest_param_grid: Optional grid of backtest parameters
            train_days: Number of days in training window
            test_days: Number of days in test window
            step_days: Number of days to step forward
            start_date: Start date for analysis
            end_date: End date for analysis
            metric_name: Name of metric to optimize
            higher_is_better: Whether higher metric values are better
            max_workers: Maximum number of parallel workers
            strategy_config_path: Optional path to strategy configuration file
            
        Returns:
            Dictionary with walk-forward optimization results
        """
        if self.data_storage is None:
            raise ValueError("Data storage required for walk-forward optimization")
            
        # Default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - pd.Timedelta(days=train_days + test_days)
            
        # Load full dataset
        full_df = await self.load_data(
            exchange=exchange,
            market=market,
            resolution=resolution,
            start_time=start_date,
            end_time=end_date
        )
        
        if full_df.empty:
            raise ValueError(f"No data found for {exchange} {market} {resolution}")
            
        # Ensure timestamp column is datetime
        timestamp_col = 'timestamp'
        if timestamp_col not in full_df.columns:
            # Try to find timestamp column
            datetime_cols = [c for c in full_df.columns if pd.api.types.is_datetime64_any_dtype(full_df[c])]
            if datetime_cols:
                timestamp_col = datetime_cols[0]
            elif isinstance(full_df.index, pd.DatetimeIndex):
                full_df = full_df.reset_index()
                timestamp_col = 'index'
            else:
                raise ValueError("No timestamp column found in data")
                
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(full_df[timestamp_col]):
            full_df[timestamp_col] = pd.to_datetime(full_df[timestamp_col])
            
        # Sort by timestamp
        full_df = full_df.sort_values(timestamp_col)
        
        # Generate time windows for walk-forward analysis
        current_start = start_date
        windows = []
        
        while current_start + pd.Timedelta(days=train_days+test_days) <= end_date:
            train_start = current_start
            train_end = train_start + pd.Timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_days)
            
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_start += pd.Timedelta(days=step_days)
            
        if not windows:
            raise ValueError(f"No valid time windows found between {start_date} and {end_date}")
            
        logger.info(f"Running walk-forward optimization with {len(windows)} windows")
        
        # Run optimization for each window
        window_results = []
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}: "
                       f"Train {window['train_start']} to {window['train_end']}, "
                       f"Test {window['test_start']} to {window['test_end']}")
            
            # Filter data for this window
            train_mask = (
                (full_df[timestamp_col] >= window['train_start']) & 
                (full_df[timestamp_col] < window['train_end'])
            )
            train_df = full_df[train_mask].copy()
            
            test_mask = (
                (full_df[timestamp_col] >= window['test_start']) & 
                (full_df[timestamp_col] < window['test_end'])
            )
            test_df = full_df[test_mask].copy()
            
            if train_df.empty or test_df.empty:
                logger.warning(f"Empty dataset for window {i+1}. Skipping.")
                continue
                
            # Run optimization on training data
            opt_results = self.optimize(
                df=train_df,
                strategy_class=strategy_class,
                strategy_param_grid=strategy_param_grid,
                backtest_param_grid=backtest_param_grid,
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                max_workers=max_workers,
                test_name=f"wfo_window_{i+1}_train",
                strategy_config_path=strategy_config_path
            )
            
            if not opt_results.get("best_params"):
                logger.warning(f"No optimal parameters found for window {i+1}. Skipping.")
                continue
                
            # Test optimal parameters on test data
            best_params = opt_results["best_params"]
            strategy_params = {k: v for k, v in best_params.items() 
                             if k in strategy_param_grid}
            backtest_params = {k: v for k, v in best_params.items() 
                              if k in backtest_param_grid} if backtest_param_grid else {}
                              
            # Add config_path if provided
            if strategy_config_path:
                strategy_params['config_path'] = strategy_config_path
                              
            # Instantiate strategy with best params
            strategy = strategy_class(**strategy_params)
            
            # Run backtest on test data
            test_result = self.backtester.run_backtest(
                df=test_df,
                strategy=strategy,
                **backtest_params
            )
            
            # Save test result
            test_result_path = self.results_dir / f"wfo_window_{i+1}_test.json"
            test_result.save_results(str(test_result_path))
            
            # Plot test result
            plot_path = self.results_dir / f"wfo_window_{i+1}_test_plot.png"
            test_result.plot_results(str(plot_path))
            
            # Store window results
            window_results.append({
                'window_idx': i+1,
                'train_period': (window['train_start'], window['train_end']),
                'test_period': (window['test_start'], window['test_end']),
                'train_optimization': opt_results,
                'test_result': test_result.metrics,
                'params_used': best_params
            })
            
        # Compile overall results
        wfo_summary = {
            'test_id': f"walk_forward_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exchange': exchange,
            'market': market,
            'resolution': resolution,
            'strategy': strategy_class.__name__,
            'train_days': train_days,
            'test_days': test_days,
            'step_days': step_days,
            'full_period': (start_date, end_date),
            'windows': len(windows),
            'successful_windows': len(window_results),
            'optimization_metric': metric_name,
            'window_results': window_results,
            'avg_train_metric': np.mean([
                w['train_optimization']['best_metrics'][metric_name] 
                for w in window_results
            ]) if window_results else None,
            'avg_test_metric': np.mean([
                w['test_result'][metric_name]
                for w in window_results
            ]) if window_results else None,
        }
        
        # Save walk-forward optimization results
        wfo_path = self.results_dir / f"{wfo_summary['test_id']}.json"
        with open(wfo_path, 'w') as f:
            json.dump(wfo_summary, f, indent=2, default=str)
            
        # Plot walk-forward results
        self._plot_walk_forward_results(wfo_summary, metric_name)
        
        return wfo_summary
    
    def _plot_walk_forward_results(self, wfo_results: Dict, metric_name: str):
        """
        Plot walk-forward optimization results.
        
        Args:
            wfo_results: Walk-forward optimization results
            metric_name: Name of metric to plot
        """
        if not wfo_results.get('window_results'):
            logger.warning("No window results to plot")
            return
            
        window_data = []
        for window in wfo_results['window_results']:
            window_data.append({
                'Window': window['window_idx'],
                'Train Start': window['train_period'][0],
                'Train End': window['train_period'][1],
                'Test Start': window['test_period'][0],
                'Test End': window['test_period'][1],
                f'Train {metric_name}': window['train_optimization']['best_metrics'][metric_name],
                f'Test {metric_name}': window['test_result'][metric_name]
            })
            
        df = pd.DataFrame(window_data)
        
        # Plot metrics comparison
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df['Window'], df[f'Train {metric_name}'], 'b-', label=f'Train {metric_name}')
        plt.plot(df['Window'], df[f'Test {metric_name}'], 'r-', label=f'Test {metric_name}')
        plt.xlabel('Window')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Walk-Forward Optimization Results')
        
        # Plot performance decay
        plt.subplot(2, 1, 2)
        performance_decay = df[f'Test {metric_name}'] - df[f'Train {metric_name}']
        plt.bar(df['Window'], performance_decay)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Window')
        plt.ylabel(f'{metric_name} Difference (Test - Train)')
        plt.grid(True, alpha=0.3)
        plt.title('Performance Decay')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"{wfo_results['test_id']}_results_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot parameter stability
        self._plot_parameter_stability(wfo_results)
        
    def _plot_parameter_stability(self, wfo_results: Dict):
        """
        Plot stability of optimal parameters across windows.
        
        Args:
            wfo_results: Walk-forward optimization results
        """
        if not wfo_results.get('window_results'):
            return
            
        # Extract parameters
        all_params = {}
        for window in wfo_results['window_results']:
            for param, value in window['params_used'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append((window['window_idx'], value))
                
        # Plot each parameter's stability
        for param, values in all_params.items():
            windows, param_values = zip(*values)
            
            plt.figure(figsize=(12, 6))
            
            # Check if parameter is numeric
            is_numeric = all(isinstance(x, (int, float)) for x in param_values)
            
            if is_numeric:
                plt.plot(windows, param_values, 'o-')
                plt.ylabel(param)
            else:
                # For categorical parameters, we need a different approach
                unique_values = sorted(set(param_values))
                value_map = {v: i for i, v in enumerate(unique_values)}
                numeric_values = [value_map[v] for v in param_values]
                
                plt.plot(windows, numeric_values, 'o-')
                plt.yticks(range(len(unique_values)), unique_values)
                plt.ylabel(param)
                
            plt.xlabel('Window')
            plt.grid(True, alpha=0.3)
            plt.title(f'Parameter Stability: {param}')
            
            # Save plot
            plot_path = self.results_dir / f"{wfo_results['test_id']}_{param}_stability.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()