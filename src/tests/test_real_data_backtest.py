import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.core.config import StorageConfig
from src.storage.processed import ProcessedDataStorage
from src.utils.indicators.wrapper_supertrend import SupertrendIndicator
from src.utils.indicators.wrapper_lorentzian import LorentzianIndicator
from src.utils.strategy.multi_indicator_strategy import MultiIndicatorStrategy
from src.backtesting.backtester import Backtester
from src.backtesting.performance_metrics import PerformanceAnalyzer
from src.backtesting.risk_analysis import RiskAnalyzer

@pytest.mark.real_data
class TestRealDataBacktest:
    @pytest.mark.asyncio
    async def test_btc_backtest(self):
        """Run a complete backtest on real BTC data"""
        # Create a proper StorageConfig instance pointing to your real data folders.
        config = StorageConfig(
            data_path=Path("data"),
            historical_raw_path=Path("data/historical/raw"),
            historical_processed_path=Path("data/historical/processed"),
            live_raw_path=Path("data/live/raw"),
            live_processed_path=Path("data/live/processed"),
            use_compression=False
        )
        data_storage = ProcessedDataStorage(config)
        backtester = Backtester(data_storage=data_storage)
        
        # Load 1 year of BTC data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        df = await backtester.load_data(
            exchange="binance",
            market="BTCUSDT",
            resolution="1D",
            start_time=start_date,
            end_time=end_date
        )
        
        # Verify data is loaded
        assert not df.empty, "No data loaded"
        print(f"Loaded {len(df)} days of BTC data")
        
        # Create a strategy using multiple indicators
        supertrend = SupertrendIndicator(atr_length=14, factor=3)
        lorentzian = LorentzianIndicator()
        
        strategy = MultiIndicatorStrategy(
            config={"consensus_threshold": 0},
            indicators=[supertrend, lorentzian]
        )
        
        # Run backtest with defined parameters
        result = backtester.run_backtest(
            df=df,
            strategy=strategy,
            initial_capital=10000.0,
            position_size_pct=20.0,
            stop_loss_pct=5.0,
            take_profit_pct=15.0,
            commission_pct=0.1
        )
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        equity_curve = pd.Series(
            [point['equity'] for point in result.portfolio.equity_curve],
            index=pd.to_datetime([point['timestamp'] for point in result.portfolio.equity_curve])
        )
        
        report = analyzer.analyze_equity_curve(
            equity_curve=equity_curve,
            trades=[p.to_dict() for p in result.portfolio.closed_positions]
        )
        
        # Print key metrics and save results plot
        print(f"Total Return: {report.return_stats.total_return_pct:.2f}%")
        print(f"Sharpe Ratio: {report.risk_metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {report.drawdown_stats.max_drawdown_pct:.2f}%")
        print(f"Win Rate: {report.trade_stats.win_rate:.2f}%")
        
        result.plot_results("btc_backtest_results.png")
        
        # Verify that the backtest produced reasonable results
        assert not pd.isna(report.return_stats.total_return_pct)
        assert len(result.portfolio.positions) > 0
