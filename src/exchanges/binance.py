"""
Binance exchange handler implementation.
"""

import logging
import hmac
import hashlib
import time
from typing import Union, List, Dict, Optional
from datetime import datetime
import asyncio
import aiohttp
from .base import BaseExchangeHandler
from ..core.models import StandardizedCandle, TimeRange
from ..core.exceptions import ExchangeError, ValidationError, RateLimitError

logger = logging.getLogger(__name__)

class BinanceHandler(BaseExchangeHandler):
    """Handler for Binance exchange data."""

    def __init__(self, config):
        """Initialize Binance handler with configuration."""
        super().__init__(config)
        self.timeframe_map = {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "1D": "1d"
        }

    def _get_headers(self) -> Dict:
        """Get headers for API requests."""
        headers = {'Accept': 'application/json'}
        # Only add API key if credentials exist and have an API key
        if self.credentials and self.credentials.api_key:
            headers['X-MBX-APIKEY'] = self.credentials.api_key
        return headers

    def _generate_signature(self, params: Dict) -> str:
        """Generate signed message for authenticated requests."""
        # Skip if no credentials or no API secret
        if not self.credentials or not self.credentials.api_secret:
            return None
            
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.credentials.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _convert_market_symbol(self, market: str) -> str:
        """Convert internal market symbol to Binance format."""
        binance_market = market.replace('-PERP', 'USDT').replace('-', '')

        # Adjust for Binance Futures naming convention (if needed)
        if "USDT" in binance_market and "PERP" in market:
            binance_market += "_PERP"  # e.g., SOLUSDT_PERP for Binance Futures

        return binance_market

    def _parse_raw_candle(self, raw_data: List, market: str, resolution: str) -> StandardizedCandle:
        """Parse raw candle data into StandardizedCandle format."""
        try:
            # Binance kline format:
            # [
            #   0  Open time
            #   1  Open
            #   2  High
            #   3  Low
            #   4  Close
            #   5  Volume
            #   6  Close time
            #   7  Quote asset volume
            #   8  Number of trades
            #   9  Taker buy base asset volume
            #   10 Taker buy quote asset volume
            #   11 Ignore
            # ]

            candle = StandardizedCandle(
                timestamp=self.standardize_timestamp(raw_data[0]),  # Open time
                open=float(raw_data[1]),
                high=float(raw_data[2]),
                low=float(raw_data[3]),
                close=float(raw_data[4]),
                volume=float(raw_data[5]),
                source='binance',
                resolution=resolution,
                market=market,
                raw_data=raw_data
            )

            self.validate_candle(candle)
            return candle

        except (IndexError, ValueError) as e:
            raise ValidationError(f"Error parsing Binance candle data: {str(e)}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error parsing Binance candle: {str(e)}")

    async def fetch_historical_candles(
        self,
        market: str,
        time_range: TimeRange,
        resolution: str
    ) -> List[StandardizedCandle]:
        """Fetch historical candle data from Binance."""
        self.validate_market(market)
        binance_symbol = self._convert_market_symbol(market)
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        candles = []
        start_time = int(time_range.start.timestamp() * 1000)
        end_time = int(time_range.end.timestamp() * 1000)
        
        try:
            while start_time < end_time:  # Simplified loop condition
                params = {
                    'symbol': binance_symbol,
                    'interval': interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000  # Maximum allowed by Binance
                }

                # Make API request
                response = await self._make_request(
                    method='GET',
                    endpoint='/api/v3/klines',
                    params=params,
                    headers=self._get_headers()
                )

                # Process response
                if not response:
                    break  # No more data, exit loop
                    
                for raw_candle in response:
                    try:
                        candle = self._parse_raw_candle(raw_candle, market, resolution)
                        candles.append(candle)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid candle: {str(e)}")
                        continue

                # Update start_time for next iteration - prevent infinite loop
                if len(response) > 0:
                    last_timestamp = int(response[-1][0])
                    if last_timestamp <= start_time:
                        # If timestamp isn't increasing, force increment
                        start_time += 1
                    else:
                        start_time = last_timestamp + 1
                else:
                    break  # Empty response

                # Respect rate limits
                await self._handle_rate_limit()

        except RateLimitError:
            logger.warning("Rate limit hit, waiting before retry")
            raise
        except Exception as e:
            raise ExchangeError(f"Failed to fetch historical data: {str(e)}")

        return candles

    async def fetch_live_candles(
        self,
        market: str,
        resolution: str
    ) -> StandardizedCandle:
        """Fetch live candle data from Binance."""
        self.validate_market(market)
        binance_symbol = self._convert_market_symbol(market)
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        try:
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': 1  # We only need the latest candle
            }

            response = await self._make_request(
                method='GET',
                endpoint='/api/v3/klines',
                params=params,
                headers=self._get_headers()
            )

            if not response:
                raise ExchangeError(f"No live data available for {market}")

            return self._parse_raw_candle(response[0], market, resolution)

        except Exception as e:
            raise ExchangeError(f"Failed to fetch live data: {str(e)}")

    async def _make_request(self, method: str, endpoint: str, params: Dict = None, headers: Dict = None) -> Union[Dict, List, None]:
        """
        Handles making requests to the Binance API, including rate limiting and error handling.
        """
        url = self.base_url + endpoint  # Correctly construct full URL
        #print(f"Debug Binance: _make_request - Method: {method}, URL: {url}, Params: {params}")  # DEBUG PRINT REQUEST DETAILS

        try:
            async with self._session.request(method, url, params=params, headers=headers, timeout=10) as response:  # Using self._session directly
                #print(f"Debug Binance: _make_request - Response Status: {response.status}")  # DEBUG PRINT RESPONSE STATUS
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                return await response.json()
        except asyncio.TimeoutError:
            #print(f"Debug Binance: _make_request - Timeout Error for URL: {url}")  # DEBUG PRINT TIMEOUT
            raise Exception(f"Timeout error for URL: {url}")  # Re-raise as generic Exception for handler to catch
        except aiohttp.ClientError as e:
            #print(f"Debug Binance: _make_request - Client Error for URL: {url}, Error: {e}")  # DEBUG PRINT CLIENT ERROR
            raise Exception(f"API request failed: {e}")  # Re-raise as generic Exception

    async def get_markets(self) -> List[str]:
        """Get available markets from Binance."""
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/api/v3/exchangeInfo',
                headers=self._get_headers()
            )

            markets = []
            for symbol in response.get('symbols', []):
                if symbol['status'] == 'TRADING':
                    markets.append(f"{symbol['baseAsset']}-{symbol['quoteAsset']}")

            return markets

        except Exception as e:
            raise ExchangeError(f"Failed to fetch markets: {str(e)}")

    async def get_exchange_info(self) -> Dict:
        """Get exchange information."""
        try:
            response = await self._make_request(
                method='GET',
                endpoint='/api/v3/exchangeInfo',
                headers=self._get_headers()
            )

            return {
                "name": self.name,
                "markets": await self.get_markets(),
                "timeframes": list(self.timeframe_map.values()),
                "has_live_data": True,
                "rate_limit": self.rate_limit,
                "server_time": response.get('serverTime'),
                "exchange_filters": response.get('exchangeFilters', [])
            }

        except Exception as e:
            raise ExchangeError(f"Failed to fetch exchange info: {str(e)}")