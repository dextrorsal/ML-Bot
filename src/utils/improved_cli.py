"""
Ultimate Data Fetcher - Enhanced CLI
Command-line interface for fetching and managing crypto data.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
import argparse
import sys
import os
import math
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.config import Config
from ..core.models import TimeRange
from ..core.exceptions import DataFetcherError
from ..core.symbol_mapper import SymbolMapper
try:
    from .logging_wrapper import enhanced_setup_logging as setup_logging
except ImportError:
    # Fallback to original logging
    from .logging import setup_logging

logger = logging.getLogger(__name__)

# Default indicator settings
DEFAULT_INDICATOR_SETTINGS = {
    "supertrend": {
        "atr_length": 10,
        "factor": 3,
        "training_length": 100,
        "kmeans_settings": {
            "high_volatility_percentile": 0.75,
            "medium_volatility_percentile": 0.50,
            "low_volatility_percentile": 0.25
        }
    },
    "logistic_regression": {
        "price_type": "close",
        "resolution": "",
        "lookback": 10,
        "norm_lookback": 10,
        "learning_rate": 0.0009,
        "iterations": 1000,
        "filter_signals_by": "None",
        "show_curves": False,
        "easteregg": False,
        "use_price_data": True,
        "holding_period": 5,
        "lot_size": 0.01
    },
    "knn": {
        "short_period": 14,
        "long_period": 28,
        "base_neighbors": 252,
        "bar_threshold": 300,
        "timeframe": "chart",
        "volatility_filter": True,
        "wait_for_close": True
    },
    "lorentzian": {
        "general_settings": {
            "neighbors_count": 8,
            "max_bars_back": 5000,
            "color_compression": 1,
            "show_default_exits": True,
            "use_dynamic_exits": False,
            "use_worst_case_estimates": False
        },
        "features": [
            {
                "type": "RSI",
                "parameter_a": 14,
                "parameter_b": 1
            },
            {
                "type": "WT",
                "parameter_a": 10,
                "parameter_b": 11
            },
            {
                "type": "CCI",
                "parameter_a": 20,
                "parameter_b": 1
            },
            {
                "type": "ADX",
                "parameter_a": 20,
                "parameter_b": 2
            },
            {
                "type": "RSI",
                "parameter_a": 9,
                "parameter_b": 1
            }
        ],
        "filters": {
            "use_volatility_filter": False,
            "use_regime_filter": False,
            "regime_threshold": -0.1,
            "use_adx_filter": False,
            "adx_threshold": 20,
            "use_ema_filter": False,
            "ema_period": 200,
            "use_sma_filter": False,
            "sma_period": 200
        },
        "kernel_settings": {
            "lookback_window": 8,
            "relative_weighting": 8,
            "regression_level": 25,
            "show_kernel_estimate": True,
            "enhance_kernel_smoothing": False,
            "lag": 0
        }
    }
}


def parse_date(date_str: str) -> datetime:
    """Parse date string in flexible formats."""
    try:
        # Try exact format YYYY-MM-DD
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            # Try format YYYY-MM-DD HH:MM
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except ValueError:
            # Handle relative dates
            if date_str.lower() == 'yesterday':
                return (datetime.now(timezone.utc) - timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0)
            elif date_str.lower() == 'today':
                return datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0)
            elif date_str.lower() == 'now':
                return datetime.now(timezone.utc)
            elif date_str.endswith('d'):
                days = int(date_str[:-1])
                return (datetime.now(timezone.utc) - timedelta(days=days))
            elif date_str.endswith('h'):
                hours = int(date_str[:-1])
                return (datetime.now(timezone.utc) - timedelta(hours=hours))
            else:
                raise ValueError(f"Unrecognized date format: {date_str}")


def get_default_markets(config: Config) -> List[str]:
    """Get default markets from all configured exchanges."""
    markets = set()
    for exchange_config in config.exchanges.values():
        if exchange_config.enabled:
            markets.update(exchange_config.markets)
    return list(markets)


def load_indicator_settings(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load indicator settings from config file or use defaults."""
    settings = DEFAULT_INDICATOR_SETTINGS.copy()
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if "indicators" in config:
                    # Update settings with values from config file
                    for indicator, indicator_settings in config["indicators"].items():
                        indicator_name = indicator.lower()
                        if indicator_name in settings:
                            settings[indicator_name].update(indicator_settings)
        except Exception as e:
            logger.error(f"Error loading indicator settings from {config_file}: {e}")
    
    return settings


async def handle_historical_mode(fetcher, args, config):
    """Handle historical data fetching mode."""
    # Handle start and end dates
    if args.start_date and args.end_date:
        start_time = parse_date(args.start_date)
        end_time = parse_date(args.end_date)
    elif args.days:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=args.days)
    else:
        raise DataFetcherError("Either specify --start-date and --end-date, or use --days")

    # Ensure end time is not in the future
    now = datetime.now(timezone.utc)
    if end_time > now:
        logger.warning(f"End time {end_time} is in the future, adjusting to current time")
        end_time = now

    time_range = TimeRange(start=start_time, end=end_time)

    logger.info(f"Fetching historical data from {start_time} to {end_time}")
    logger.info(f"Markets: {args.markets}")
    logger.info(f"Exchanges: {args.exchanges if args.exchanges else 'All configured exchanges'}")
    logger.info(f"Resolution: {args.resolution}")

    await fetcher.fetch_historical_data(
        markets=args.markets,
        time_range=time_range,
        resolution=args.resolution,
        exchanges=args.exchanges
    )


async def handle_live_mode(fetcher, args):
    """Handle live data fetching mode."""
    logger.info("Starting live data fetching")
    logger.info(f"Markets: {args.markets}")
    logger.info(f"Exchanges: {args.exchanges if args.exchanges else 'All configured exchanges'}")
    logger.info(f"Resolution: {args.resolution}")

    await fetcher.start_live_fetching(
        markets=args.markets,
        resolution=args.resolution,
        exchanges=args.exchanges
    )


async def handle_list_mode(fetcher, args):
    """Handle listing exchanges and markets with pagination and filtering."""
    output_file = None
    original_stdout = sys.stdout
    if args.output:
        try:
            output_file = open(args.output, 'w')
            sys.stdout = output_file
        except Exception as e:
            print(f"Error opening output file {args.output}: {e}")
            return

    try:
        print("\n=== Available Exchanges and Markets ===")

        symbol_mapper = SymbolMapper()

        exchanges_to_show = list(fetcher.exchange_handlers.keys())
        if args.exchange:
            exchange_filter = args.exchange.lower()
            exchanges_to_show = [e for e in exchanges_to_show if e.lower() == exchange_filter]
            if not exchanges_to_show:
                print(f"No exchange found matching '{args.exchange}'")
                return

        all_markets = []

        for name in exchanges_to_show:
            handler = fetcher.exchange_handlers[name]
            print(f"\n[{name.upper()}]")
            try:
                exchange_markets = []
                if hasattr(handler, 'get_exchange_info'):
                    info = await handler.get_exchange_info()
                    print(f"Base URL: {handler.base_url}")
                    print(f"Rate Limit: {handler.rate_limit} requests per second")
                    if 'markets' in info:
                        exchange_markets = info['markets']
                else:
                    print(f"Base URL: {handler.base_url}")
                    print(f"Rate Limit: {handler.rate_limit} requests per second")
                    exchange_markets = handler.markets

                symbol_mapper.register_exchange(name, exchange_markets)

                if args.filter:
                    filter_term = args.filter.upper()
                    filtered_markets = [m for m in exchange_markets if filter_term in m.upper()]
                else:
                    filtered_markets = exchange_markets

                for market in sorted(filtered_markets):
                    try:
                        standard = symbol_mapper.from_exchange_symbol(name, market)
                        all_markets.append((name, market, standard))
                    except ValueError:
                        all_markets.append((name, market, "[Conversion failed]"))

                if hasattr(handler, 'get_exchange_info') and 'timeframes' in info:
                    print("\nSupported Timeframes:")
                    for tf in sorted(info['timeframes'], key=lambda x: str(x)):
                        print(f"  - {tf}")

            except Exception as e:
                print(f"Error retrieving info: {e}")

        if all_markets:
            total_items = len(all_markets)
            total_pages = math.ceil(total_items / args.page_size)
            current_page = max(1, min(args.page, total_pages))

            start_idx = (current_page - 1) * args.page_size
            end_idx = min(start_idx + args.page_size, total_items)

            print(f"\n=== Markets (Page {current_page}/{total_pages}, {total_items} total) ===")

            if args.filter:
                print(f"Filter: '{args.filter}'")

            if args.show_symbols:
                print("\nMarket Symbols (Exchange → Standard):")
                for i in range(start_idx, end_idx):
                    exchange, market, standard = all_markets[i]
                    print(f"  - {exchange.upper():8} | {market:15} → {standard}")
            else:
                print("\nAvailable Markets:")
                for i in range(start_idx, end_idx):
                    exchange, market, _ = all_markets[i]
                    print(f"  - {exchange.upper():8} | {market}")

            if total_pages > 1:
                print(f"\nUse --page <num> to view different pages (1-{total_pages})")

        if args.show_symbols:
            print("\n=== Using Standardized Symbols ===")
            print("When fetching data, you can now use standardized symbols like:")
            print("  BTC-USD, ETH-USDT, SOL-PERP")
            print("\nFor example:")
            print("  python fetch.py historical --days 7 --markets BTC-USD ETH-USD SOL-PERP")
            print("\nThe system will automatically convert to the appropriate format for each exchange.")

            common_symbols = ["BTC-USD", "ETH-USD", "SOL-PERP"]
            print("\nExample conversions for common symbols:")
            for standard in common_symbols:
                print(f"\n{standard}:")
                for exchange in exchanges_to_show:
                    try:
                        exchange_symbol = symbol_mapper.to_exchange_symbol(exchange, standard)
                        print(f"  - {exchange.upper():10}: {exchange_symbol}")
                    except ValueError:
                        print(f"  - {exchange.upper():10}: [Not supported]")

        if args.output:
            sys.stdout = original_stdout
            print(f"Output saved to {args.output}")

    finally:
        if output_file:
            output_file.close()
            if sys.stdout != original_stdout:
                sys.stdout = original_stdout


async def handle_backtest_mode(fetcher, args):
    """Handle backtesting mode."""
    from src.backtesting.backtester import Backtester
    from src.storage.processed import ProcessedDataStorage
    import json

    logger.info(f"Starting backtest for {args.market} on {args.exchange}")

    data_storage = ProcessedDataStorage()
    backtester = Backtester(data_storage)

    args.strategy_instance = load_strategy(args)

    try:
        result = backtester.run_backtest_with_args(args)
        backtester.print_results_summary(result)
        if args.output:
            result.save_results(args.output)
            logger.info(f"Results saved to {args.output}")
        if args.plot:
            result.plot_results()
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}", exc_info=args.debug)
        print(f"Error: {str(e)}")


def load_strategy(args):
    """Load a trading strategy based on CLI arguments."""
    from src.utils.strategy.base import BaseStrategy
    from src.strategy.multi_indicator_strategy import MultiIndicatorStrategy
    
    config = {}
    
    # Add threshold to config if specified
    if args.threshold is not None:
        config["consensus_threshold"] = args.threshold
    
    # Load indicators
    indicators = []
    for indicator_name in args.indicators:
        indicator = load_indicator(indicator_name)
        if indicator:
            indicators.append(indicator)
    
    if not indicators:
        logger.warning("No valid indicators specified, using default Supertrend")
        from src.utils.indicators.wrapper_supertrend import SupertrendIndicator
        indicators = [SupertrendIndicator(config_path=args.config_file)]
    
    # Create strategy based on name
    if args.strategy.lower() == "multi_indicator":
        return MultiIndicatorStrategy(config=config, indicators=indicators, weights=args.weights)
    else:
        logger.warning(f"Unknown strategy: {args.strategy}, using MultiIndicatorStrategy")
        return MultiIndicatorStrategy(config=config, indicators=indicators, weights=args.weights)


def load_indicator(indicator_name, settings=None):
    """Load an indicator by name with optional configuration."""
    indicator_name = indicator_name.lower()
    config_path = getattr(args, 'config_file', 'config/indicator_settings.json')
    
    if indicator_name == "supertrend":
        from src.utils.indicators.wrapper_supertrend import SupertrendIndicator
        return SupertrendIndicator(config_path=config_path)
    
    elif indicator_name == "rsi":
        from src.utils.indicators.wrapper_rsi import RsiIndicator
        return RsiIndicator(config_path=config_path)
    
    elif indicator_name == "macd":
        from src.utils.indicators.wrapper_macd import MacdIndicator
        return MacdIndicator(config_path=config_path)
    
    elif indicator_name == "knn":
        from src.utils.indicators.wrapper_knn import KNNIndicator
        return KNNIndicator(config_path=config_path)
    
    elif indicator_name == "logistic" or indicator_name == "logistic_regression":
        from src.utils.indicators.wrapper_logistic import LogisticRegressionIndicator
        return LogisticRegressionIndicator(config_path=config_path)
    
    elif indicator_name == "lorentzian":
        from src.utils.indicators.wrapper_lorentzian import LorentzianIndicator
        return LorentzianIndicator(config_path=config_path)
    
    else:
        logger.warning(f"Unknown indicator: {indicator_name}")
        return None


async def main():
    """Main entry point with enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="Ultimate Crypto Data Fetcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", default=".env",
                        help="Path to configuration file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    historical_parser = subparsers.add_parser("historical",
                                               help="Fetch historical data")
    historical_parser.add_argument("--markets", nargs="+",
                                   help="Markets to fetch (e.g., BTC-PERP ETH-PERP)")
    historical_parser.add_argument("--exchanges", nargs="+",
                                   help="Exchanges to use (default: all configured)")
    historical_parser.add_argument("--resolution", default="1D",
                                   choices=["1", "5", "15", "30", "60", "240", "1D", "1W"],
                                   help="Candle resolution")
    time_group = historical_parser.add_argument_group("Time Range Options")
    time_group.add_argument("--start-date",
                            help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM or relative like 7d)")
    time_group.add_argument("--end-date",
                            help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM or 'now')")
    time_group.add_argument("--days", type=int,
                            help="Number of days to fetch from today (alternative to date range)")

    live_parser = subparsers.add_parser("live",
                                         help="Fetch live data continuously")
    live_parser.add_argument("--markets", nargs="+",
                             help="Markets to fetch (e.g., BTC-PERP ETH-PERP)")
    live_parser.add_argument("--exchanges", nargs="+",
                             help="Exchanges to use (default: all configured)")
    live_parser.add_argument("--resolution", default="1",
                             choices=["1", "5", "15", "30", "60", "240"],
                             help="Candle resolution")
    live_parser.add_argument("--interval", type=int, default=60,
                             help="Update interval in seconds")

    list_parser = subparsers.add_parser("list",
                                         help="List available exchanges and markets")
    list_parser.add_argument("--show-symbols", action="store_true",
                             help="Show standardized symbol mappings for each exchange")
    list_parser.add_argument("--output", "-o", type=str,
                             help="Save output to a file instead of displaying it")
    list_parser.add_argument("--filter", "-f", type=str,
                             help="Filter markets containing this string (e.g. 'BTC')")
    list_parser.add_argument("--page", "-p", type=int, default=1,
                             help="Page number when displaying long lists")
    list_parser.add_argument("--page-size", type=int, default=20,
                             help="Number of items per page")
    list_parser.add_argument("--exchange", "-e", type=str,
                             help="Show only markets from this exchange")

    backtest_parser = subparsers.add_parser("backtest",
                                             help="Run strategy backtests on historical data")
    backtest_parser.add_argument("--market", required=True,
                                 help="Market to backtest (e.g., BTC-USD, SOL-PERP)")
    backtest_parser.add_argument("--exchange", default="binance",
                                 help="Exchange to use for data")
    backtest_parser.add_argument("--strategy", default="multi_indicator",
                                 help="Strategy name to test")
    backtest_parser.add_argument("--indicators", nargs="+", default=["supertrend"],
                                 help="List of indicators to use (e.g., supertrend knn rsi lorentzian logistic)")
    backtest_parser.add_argument("--weights", nargs="+", type=float,
                                 help="Optional weights for indicators")
    backtest_parser.add_argument("--config-file",
                                 help="JSON configuration file for strategy parameters")
    backtest_parser.add_argument("--days", type=int, default=90,
                                 help="Number of days of historical data to use")
    backtest_parser.add_argument("--resolution", default="1D",
                                 choices=["1", "5", "15", "30", "60", "240", "1D", "1W"],
                                 help="Candle resolution")
    backtest_parser.add_argument("--threshold", type=float, default=0,
                                 help="Consensus threshold for multi-indicator strategy")
    backtest_parser.add_argument("--capital", type=float, default=10000.0,
                                 help="Initial capital for backtest")
    backtest_parser.add_argument("--position-size", type=float, default=10.0,
                                 help="Position size as percentage of capital")
    backtest_parser.add_argument("--stop-loss", type=float,
                                 help="Stop loss percentage")
    backtest_parser.add_argument("--take-profit", type=float,
                                 help="Take profit percentage")
    backtest_parser.add_argument("--output",
                                 help="Output file path for results")
    backtest_parser.add_argument("--plot", action="store_true",
                                 help="Display performance plot")
    backtest_parser.add_argument("--optimize", action="store_true",
                                 help="Run parameter optimization")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging()
    logging.getLogger().setLevel(log_level)

    config = Config(args.config)

    from .. import ultimate_fetcher
    from ..ultimate_fetcher import UltimateDataFetcher
    fetcher = UltimateDataFetcher(config)

    await fetcher.start()

    try:
        if args.mode == "historical":
            if not args.markets:
                args.markets = get_default_markets(config)
                logger.info(f"No markets specified, using defaults: {args.markets}")
            await handle_historical_mode(fetcher, args, config)
        elif args.mode == "live":
            if not args.markets:
                args.markets = get_default_markets(config)
                logger.info(f"No markets specified, using defaults: {args.markets}")
            await handle_live_mode(fetcher, args)
        elif args.mode == "list":
            await handle_list_mode(fetcher, args)
        elif args.mode == "backtest":
            await handle_backtest_mode(fetcher, args)
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)
    finally:
        await fetcher.stop()


if __name__ == "__main__":
    asyncio.run(main())