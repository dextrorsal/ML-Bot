"""
Coinbase exchange handler implementation with public API endpoints.
"""

import logging
import time
from typing import List, Dict, Optional, Union
import asyncio
from datetime import datetime, timezone
import json

from .base import BaseExchangeHandler
from ..core.models import StandardizedCandle, TimeRange
from ..core.exceptions import ExchangeError, ValidationError, RateLimitError

logger = logging.getLogger(__name__)

class CoinbaseHandler(BaseExchangeHandler):
    """Handler for Coinbase exchange data."""
    
    def __init__(self, config):
        """Initialize Coinbase handler with configuration."""
        super().__init__(config)
        self.base_url = "https://api.coinbase.com"
        # Updated to match Coinbase's exact granularity values
        self.timeframe_map = {
            "1": "ONE_MINUTE",
            "5": "FIVE_MINUTE",
            "15": "FIFTEEN_MINUTE",
            "30": "THIRTY_MINUTE",
            "60": "ONE_HOUR",
            "120": "TWO_HOUR",
            "360": "SIX_HOUR",
            "1D": "ONE_DAY"
        }
        logger.info("Initialized Coinbase handler with correct granularity mapping")

    def _convert_market_symbol(self, market: str) -> str:
        """
        Convert internal market symbol to Coinbase format.
        Example: 
        - SOL-PERP -> SOL-USD
        - SOLUSDT -> SOL-USD
        """
        if not isinstance(market, str):
            raise ValidationError("Market must be a string")
            
        market = market.upper()  # Ensure consistent casing
        logger.debug(f"Converting market symbol: {market}")
        # Remove -PERP suffix if present (from Drift format)
        market = market.replace('-PERP', '')
        
        # Convert from Binance format (SOLUSDT -> SOL-USD)
        if 'USDT' in market:
            base = market.replace('USDT', '')
            return f"{base}-USD"
            
        # If already in Coinbase format (SOL-USD), return as is
        if '-USD' in market:
            return market
            
        # Default case: add -USD if no other format detected
        return f"{market}-USD"

    def _get_headers(self, method: str = "GET", path: str = "") -> Dict:
        """
        Return headers for API requests.
        If credentials are provided, include authentication headers.
        """
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.config.credentials:
            timestamp = str(int(datetime.now(timezone.utc).timestamp()))
            signature = self._generate_signature(method, path, timestamp)
            headers.update({
                'CB-ACCESS-KEY': self.config.credentials.api_key,
                'CB-ACCESS-SIGN': signature,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': self.config.credentials.additional_params.get("passphrase", "")
            })
        return headers

    def _generate_signature(self, method: str, path: str, timestamp: str) -> str:
        """
        Stub for generating a signature.
        (In tests, this method is patched.)
        """
        return "stub_signature"

    async def fetch_historical_candles(
        self,
        market: str,
        time_range: TimeRange,
        resolution: str
    ) -> List[StandardizedCandle]:
        """Fetch historical candle data from Coinbase public API."""
        self.validate_market(market)
        coinbase_symbol = self._convert_market_symbol(market)
        granularity = self.timeframe_map.get(resolution)
        if not granularity:
            raise ValidationError(f"Invalid resolution: {resolution}")

        if time_range.end < time_range.start:
            raise ValidationError("End time must be after start time")
        
        logger.debug(f"Fetching {market} data from {time_range.start} to {time_range.end}")
        candles = []
        start_time = int(time_range.start.timestamp())
        end_time = int(time_range.end.timestamp())

        empty_batch_counter = 0  # Counts consecutive empty batches

        try:
            while start_time < end_time:
                batch_duration = self._get_granularity_seconds(resolution) * 300
                current_end = min(start_time + batch_duration, end_time)
                
                path = f'/api/v3/brokerage/market/products/{coinbase_symbol}/candles'
                params = {
                    'start': str(start_time),
                    'end': str(current_end),
                    'granularity': granularity
                }

                try:
                    response = await self._make_request(
                        method='GET',
                        endpoint=path,
                        params=params,
                        headers=self._get_headers("GET", path)
                    )
                except RateLimitError as e:
                    logger.warning(f"Rate limit hit: {e}. Waiting 5 seconds before retrying.")
                    await asyncio.sleep(5)
                    continue
                except Exception as e:
                    logger.error(f"Error fetching data: {e}")
                    await asyncio.sleep(5)
                    continue

                # Support both list response and dict with a 'candles' key
                if isinstance(response, list):
                    candle_list = response
                elif isinstance(response, dict) and 'candles' in response:
                    candle_list = response['candles']
                else:
                    candle_list = []
                
                if candle_list:
                    empty_batch_counter = 0  # reset counter on receiving data
                    for candle_data in candle_list:
                        try:
                            # If candle_data is a list, assume the order:
                            # [timestamp, open, high, low, close, volume]
                            if isinstance(candle_data, list):
                                candle = StandardizedCandle(
                                    timestamp=self.standardize_timestamp(int(candle_data[0])),
                                    open=float(candle_data[1]),
                                    high=float(candle_data[2]),
                                    low=float(candle_data[3]),
                                    close=float(candle_data[4]),
                                    volume=float(candle_data[5]),
                                    source='coinbase',
                                    resolution=resolution,
                                    market=market,
                                    raw_data=candle_data
                                )
                            else:
                                candle = StandardizedCandle(
                                    timestamp=self.standardize_timestamp(int(candle_data['start'])),
                                    open=float(candle_data['open']),
                                    high=float(candle_data['high']),
                                    low=float(candle_data['low']),
                                    close=float(candle_data['close']),
                                    volume=float(candle_data['volume']),
                                    source='coinbase',
                                    resolution=resolution,
                                    market=market,
                                    raw_data=candle_data
                                )
                            self.validate_candle(candle)
                            candles.append(candle)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Skipping invalid candle: {str(e)}")
                            continue
                else:
                    empty_batch_counter += 1
                    logger.warning("Empty candle batch received; moving to next interval")
                    if empty_batch_counter >= 5:
                        logger.warning("Received 5 consecutive empty batches. Waiting an extra 5 seconds before continuing.")
                        await asyncio.sleep(5)
                        empty_batch_counter = 0

                start_time = current_end
                await self._handle_rate_limit()
                logger.debug(f"Processed batch up to {current_end}")
                if len(candles) % 1000 == 0 and candles:
                    logger.info(f"Fetched {len(candles)} candles so far for {market}")

        except Exception as e:
            raise ExchangeError(f"Failed to fetch historical candles: {e}")

        return sorted(candles, key=lambda x: x.timestamp)

    async def fetch_live_candles(
        self,
        market: str,
        resolution: str
    ) -> StandardizedCandle:
        """Fetch live ticker data from Coinbase public API."""
        self.validate_market(market)
        coinbase_symbol = self._convert_market_symbol(market)

        try:
            path = f'/api/v3/brokerage/market/products/{coinbase_symbol}/ticker'
            params = {
                'limit': 1
            }

            response = await self._make_request(
                method='GET',
                endpoint=path,
                params=params,
                headers=self._get_headers("GET", path)
            )

            if not response or 'trades' not in response:
                raise ExchangeError(f"No live data available for {market}")

            trade_data = response['trades'][0]
            # Updated time parsing to handle fractional seconds
            current_time = self.standardize_timestamp(
                int(datetime.strptime(trade_data['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())
            )

            price = float(trade_data['price'])

            candle = StandardizedCandle(
                timestamp=current_time,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=float(trade_data['size']),
                source='coinbase',
                resolution=resolution,
                market=market,
                raw_data={
                    'trade': trade_data,
                    'best_bid': float(response['best_bid']),
                    'best_ask': float(response['best_ask'])
                }
            )

            self.validate_candle(candle)
            return candle

        except Exception as e:
            raise ExchangeError(f"Failed to fetch live data: {e}")

    def _get_granularity_seconds(self, resolution: str) -> int:
        """Convert resolution to seconds for calculating time windows."""
        resolution_map = {
            "1": 60,
            "5": 300,
            "15": 900,
            "30": 1800,
            "60": 3600,
            "240": 14400,
            "1D": 86400
        }
        return resolution_map.get(resolution, 60)

    async def get_markets(self) -> List[str]:
        """Get available markets from Coinbase public API."""
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/api/v3/brokerage/market/products',
                headers=self._get_headers("GET", "/api/v3/brokerage/market/products")
            )
            
            return [product['product_id'] for product in response.get('products', [])
                   if product.get('status') == 'online']
            
        except Exception as e:
            raise ExchangeError(f"Failed to fetch markets: {e}")
