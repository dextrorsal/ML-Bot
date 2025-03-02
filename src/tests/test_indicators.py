import pandas as pd
import numpy as np
import logging
import pytest
import matplotlib.pyplot as plt

from src.utils.indicators.wrapper_supertrend import SupertrendIndicator
from src.utils.indicators.wrapper_knn import KNNIndicator
from src.utils.indicators.wrapper_logistic import LogisticRegressionIndicator
from src.utils.indicators.wrapper_lorentzian import LorentzianIndicator
from src.backtesting.performance_metrics import PerformanceAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_data(rows=500):
    """
    Generate synthetic OHLCV market data with:
      - A random walk
      - Some cyclical behavior (sinusoidal drift)
      - Volatility that changes over time
    """
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="1h")
    
    # Base random walk
    price = 100 + np.cumsum(np.random.randn(rows) * 2)

    # Add a slow sinusoidal drift to simulate cyclical patterns
    # e.g. something like a weekly cycle if you're using hours
    # Adjust the frequency to taste
    cycle = 5 * np.sin(np.linspace(0, 4 * np.pi, rows))
    price += cycle

    # Add a bit of trending bias
    trend = np.linspace(0, 30, rows)  # e.g. a 30 point increase over the dataset
    price += trend

    # Introduce some local volatility changes (like volatility clusters)
    # e.g. multiply by a factor that randomly jumps
    vol_cluster = np.ones(rows)
    for i in range(1, rows):
        # 5% chance to “jump” to a new volatility regime
        if np.random.rand() < 0.05:
            vol_cluster[i] = np.random.uniform(0.5, 2.0)
        else:
            vol_cluster[i] = vol_cluster[i - 1]
    # Now apply that to the price as additional random noise
    price += np.random.randn(rows) * vol_cluster * 2

    # Build high/low/close
    # You can ensure high >= price >= low, for example:
    close = price + np.random.randn(rows) * 0.5
    high = np.maximum(price, close) + np.random.rand(rows) * 2
    low = np.minimum(price, close) - np.random.rand(rows) * 2

    # Random volume
    volume = np.random.randint(500, 3000, size=rows)

    return pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)

def plot_signals(ax, df, signal_col, label):
    """
    Plot the close price, and add buy/sell markers based on
    +1 (buy) / -1 (sell) signals in 'signal_col'.
    """
    ax.plot(df.index, df["close"], label="Close", color="black", alpha=0.7)
    
    # Identify buys/sells
    buys = df[df[signal_col] == 1]
    sells = df[df[signal_col] == -1]
    
    # Plot markers
    ax.plot(buys.index, buys["close"], "^", color="green", markersize=8, label="Buy")
    ax.plot(sells.index, sells["close"], "v", color="red", markersize=8, label="Sell")
    
    ax.set_title(f"{label} Indicator")
    ax.legend()

@pytest.mark.simple
def test_indicators():
    """
    Runs each indicator on synthetic data, prints basic stats,
    calculates performance metrics, and displays a UI with
    buy/sell markers for each indicator's signals.
    """
    # 1) Generate synthetic data
    df = generate_synthetic_data()
    
    logging.info("Testing Supertrend Indicator...")
    supertrend = SupertrendIndicator()
    df["Supertrend_Signal"] = supertrend.generate_signals(df)
    
    logging.info("Testing KNN Indicator...")
    knn = KNNIndicator()
    df["KNN_Signal"] = knn.generate_signals(df)
    
    logging.info("Testing Logistic Regression Indicator...")
    logistic = LogisticRegressionIndicator()
    df["Logistic_Signal"] = logistic.generate_signals(df)
    
    logging.info("Testing Lorentzian Indicator...")
    lorentzian = LorentzianIndicator()
    df["Lorentzian_Signal"] = lorentzian.generate_signals(df)
    
    # 2) Analyze performance of the raw 'close' (buy-and-hold as a placeholder)
    performance_analyzer = PerformanceAnalyzer()
    metrics_report = performance_analyzer.analyze_equity_curve(df['close'])
    
    # 3) Print a readable summary
    print("\n==== Performance Summary (Buy-and-Hold on Synthetic Close) ====")
    print(metrics_report.summary())
    
    # 4) Show basic stats on the signals
    print("\n==== Basic Signal Statistics ====")
    print(df[["Supertrend_Signal", "KNN_Signal", "Logistic_Signal", "Lorentzian_Signal"]].describe())
    
    # 5) Create a UI to visually compare each indicator with buy/sell markers
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    
    # Plot each indicator with buy/sell markers
    plot_signals(axs[0,0], df, "Supertrend_Signal", "Supertrend")
    plot_signals(axs[0,1], df, "KNN_Signal", "KNN")
    plot_signals(axs[1,0], df, "Logistic_Signal", "Logistic")
    plot_signals(axs[1,1], df, "Lorentzian_Signal", "Lorentzian")
    
    plt.tight_layout()
    plt.show()
