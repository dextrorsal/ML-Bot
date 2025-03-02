"""
Ultimate Data Fetcher - Main orchestrator for fetching and managing crypto data.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import argparse

from .core.config import Config
from .core.models import TimeRange
from .core.exceptions import DataFetcherError
from .exchanges import get_exchange_handler
from .storage.raw import RawDataStorage
from .storage.processed import ProcessedDataStorage
from .utils.logging import setup_logging
from .core.symbol_mapper import SymbolMapper
from .storage.live import LiveDataStorage

logger = logging.getLogger(__name__)

class UltimateDataFetcher:
    """Main orchestrator for fetching and managing crypto data."""

    def __init__(self, config=None, config_path: str = ".env"):
        """Initialize the data fetcher with configuration."""
        if config:
            self.config = config
        else:
            self.config = Config.from_ini(config_path)
        self.raw_storage = RawDataStorage(self.config.storage)
        self.processed_storage = ProcessedDataStorage(self.config.storage)

        
        # Add live data storage
        self.live_storage = LiveDataStorage(self.config.storage)
        
        self.exchange_handlers = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start the data fetcher and initialize all configured exchanges."""
        logger.info("Starting Ultimate Data Fetcher")
        # Initialize the symbol mapper (once)
        self.initialize_symbol_mapper()

        # Initialize exchange handlers
        for exchange_name, exchange_config in self.config.exchanges.items():
            if exchange_config.enabled:
                try:
                    handler = get_exchange_handler(exchange_config)
                    await handler.start()
                    self.exchange_handlers[exchange_name] = handler
                    logger.info(f"Initialized {exchange_name} handler")
                except Exception as e:
                    logger.error(f"Failed to initialize {exchange_name} handler: {e}")
                    # Optionally, reinitialize symbol mapper if needed
                    self.initialize_symbol_mapper()

    async def stop(self):
        """Stop all exchange handlers and cleanup."""
        logger.info("Stopping Ultimate Data Fetcher")
        for name, handler in self.exchange_handlers.items():
            try:
                await handler.stop()
                logger.info(f"Stopped {name} handler")
            except Exception as e:
                logger.error(f"Error stopping {name} handler: {e}")

    def initialize_symbol_mapper(self):
        """Initialize the symbol mapper with all available markets."""
        self.symbol_mapper = SymbolMapper()
        
        # Register markets for each exchange
        for exchange_name, handler in self.exchange_handlers.items():
            if hasattr(handler, 'markets') and handler.markets:
                self.symbol_mapper.register_exchange(exchange_name, handler.markets)

    async def fetch_historical_data(
        self,
        markets: List[str],
        time_range: TimeRange,
        resolution: str,
        exchanges: Optional[List[str]] = None
    ):
        """Fetch historical data for specified markets and time range."""
        if not exchanges:
            exchanges = list(self.exchange_handlers.keys())

        for exchange_name in exchanges:
            handler = self.exchange_handlers.get(exchange_name)
            if not handler:
                logger.warning(f"Exchange {exchange_name} not initialized, skipping")
                continue

            for market_symbol in markets:
                try:
                    # Check two cases:
                    # 1. The market_symbol is already in the exchange's native format
                    # 2. The market_symbol is in standard format and needs conversion

                    # First, check if the market is directly in the handler's markets list
                    is_valid_market = market_symbol in handler.markets

                    # If not found directly, try validating as a standard symbol
                    if not is_valid_market:
                        result = handler.validate_standard_symbol(market_symbol)
                        if asyncio.iscoroutine(result):
                            is_valid_market = await result
                        else:
                            is_valid_market = result

                    if not is_valid_market:
                        logger.info(f"Market {market_symbol} not supported by {exchange_name}, skipping")
                        continue

                    # Get the exchange-specific format if needed
                    try:
                        if market_symbol in handler.markets:
                            exchange_market = market_symbol
                        else:
                            exchange_market = self.symbol_mapper.to_exchange_symbol(exchange_name, market_symbol)
                    except:
                        exchange_market = market_symbol

                    logger.info(f"Fetching historical data for {market_symbol} (exchange format: {exchange_market}) from {exchange_name}")
                    candles = await handler.fetch_historical_candles(exchange_market, time_range, resolution)

                    await self.raw_storage.store_candles(
                        exchange_name,
                        market_symbol,
                        resolution,
                        candles
                    )

                    await self.processed_storage.store_candles(
                        exchange_name,
                        market_symbol,
                        resolution,
                        candles
                    )

                    logger.info(f"Stored {len(candles)} candles for {market_symbol} from {exchange_name}")

                except Exception as e:
                    logger.error(f"Error fetching data for {market_symbol} from {exchange_name}: {e}")

    async def start_live_fetching(
        self,
        markets: List[str],
        resolution: str,
        exchanges: Optional[List[str]] = None
    ):
        """Start live data fetching for specified markets."""
        if not exchanges:
            exchanges = list(self.exchange_handlers.keys())

        logger.info(f"Starting live data fetching for {markets} with resolution {resolution}")
        logger.info(f"Using exchanges: {exchanges}")

        try:
            while True:
                for exchange_name in exchanges:
                    handler = self.exchange_handlers.get(exchange_name)
                    if not handler:
                        continue

                    for standard_market in markets:
                        try:
                            exchange_market = self.symbol_mapper.to_exchange_symbol(exchange_name, standard_market)
                            
                            # Validate market symbol; await if necessary.
                            result = handler.validate_standard_symbol(standard_market)
                            if asyncio.iscoroutine(result):
                                valid = await result
                            else:
                                valid = result
                            if not valid:
                                continue
                            
                            logger.debug(f"Fetching live data for {standard_market} from {exchange_name}")
                            candle = await handler.fetch_live_candles(
                                market=exchange_market,
                                resolution=resolution
                            )

                            await self.live_storage.store_raw_candle(
                                exchange_name,
                                standard_market,
                                resolution,
                                candle
                            )

                            await self.live_storage.store_processed_candle(
                                exchange_name,
                                standard_market,
                                resolution,
                                candle
                            )

                            logger.debug(f"Stored live candle for {standard_market} from {exchange_name}")

                        except Exception as e:
                            logger.error(f"Error fetching live data for {standard_market} from {exchange_name}: {e}")

                await asyncio.sleep(60)  # Adjust based on resolution
                
        except KeyboardInterrupt:
            logger.info("Live data fetching interrupted by user")
        except Exception as e:
            logger.error(f"Error in live data fetching: {e}")
            raise


    async def main():
        """Main entry point for the Ultimate Data Fetcher."""
        parser = argparse.ArgumentParser(description="Ultimate Crypto Data Fetcher")
        parser.add_argument("--config", default=".env", help="Path to configuration file")
        parser.add_argument("--mode", choices=["historical", "live"], required=True,
                            help="Fetching mode: historical or live")
        parser.add_argument("--markets", nargs="+", help="Markets to fetch (e.g., BTC-PERP ETH-PERP)")
        parser.add_argument("--exchanges", nargs="+", help="Exchanges to use")
        parser.add_argument("--resolution", default="1", choices=["1", "15", "60", "240", "1D" , "1W", "5", "30"], help="Candle resolution (e.g., 1, 5, 15)")
        parser.add_argument("--start-date", help="Start date for historical data (YYYY-MM-DD)")
        parser.add_argument("--end-date", help="End date for historical data (YYYY-MM-DD)")

        args = parser.parse_args()

        # Setup logging
        setup_logging()

        # Initialize fetcher
        config = Config(args.config)
        fetcher = UltimateDataFetcher(config)
        await fetcher.start()

        try:
            if args.mode == "historical":
                if not args.start_date or not args.end_date:
                    raise DataFetcherError("Start and end dates required for historical mode")

                time_range = TimeRange(
                    start=datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    end=datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                )

                await fetcher.fetch_historical_data(
                    args.markets,
                    time_range,
                    args.resolution,
                    args.exchanges
                )
            else:  # live mode
                await fetcher.start_live_fetching(
                    markets=args.markets,
                    resolution=args.resolution,
                    exchanges=args.exchanges
                )
        finally:
            await fetcher.stop()

    if __name__ == "__main__":
        asyncio.run(main())
