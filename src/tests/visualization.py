"""
Visualization helper for testing and debugging.

This script demonstrates how to generate and save simple plots from sample
data such as an equity curve. You can extend this to visualize trade performance,
drawdowns, or any other metrics from your backtest.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_equity_curve(num_points=100):
    """
    Generate a sample equity curve with a random walk and an upward trend.
    
    Returns:
        A pandas Series indexed by datetime.
    """
    # Create a date range for the equity curve
    start_date = datetime.now() - timedelta(days=num_points)
    dates = pd.date_range(start=start_date, periods=num_points, freq='D')
    
    # Generate a random walk with an upward drift
    drift = 50
    noise = np.random.normal(0, 100, num_points)
    values = np.cumsum(np.ones(num_points) * drift + noise) + 10000
    
    return pd.Series(values, index=dates)

def plot_equity_curve(equity_curve, title="Equity Curve", save_path=None):
    """
    Plot the given equity curve.
    
    Args:
        equity_curve (pd.Series): Equity curve data with datetime index.
        title (str): Title for the plot.
        save_path (str): Optional path to save the plot as a file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve.values, marker='o', linestyle='-', label="Equity")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    # Generate a sample equity curve
    equity_curve = generate_sample_equity_curve()
    
    # Plot and save the equity curve
    plot_equity_curve(equity_curve, title="Sample Equity Curve", save_path="sample_equity_curve.png")

if __name__ == "__main__":
    main()
