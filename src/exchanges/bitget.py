"""
Bitget exchange handler implementation.
"""

import logging
import hmac
import hashlib
import time
from typing import Union, List, Dict, Optional
from datetime import datetime
import asyncio
import aiohttp
from aiohttp.client_exceptions import ContentTypeError # Import ContentTypeError from aiohttp
from .base import BaseExchangeHandler
from ..core.models import StandardizedCandle, TimeRange
from ..core.exceptions import ExchangeError, ValidationError, RateLimitError
import json # Import json for debugging prints

logger = logging.getLogger(__name__)

class BitgetHandler(BaseExchangeHandler):
    """Handler for Bitget exchange data."""

    def __init__(self, config):
        """Initialize Bitget handler with configuration."""
        super().__init__(config)
        self.timeframe_map = {
            "1": "1m", # Verify Bitget timeframe codes
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "1D": "1D", # Or "1d", verify Bitget daily timeframe code
            "1W": "1W", # Verify weekly
            "1M": "1M"  # Verify monthly
        }
        self.inverse_timeframe_map = {v: k for k, v in self.timeframe_map.items()} # Create inverse map

    def _get_headers(self) -> Dict:
        """Get headers for API requests."""
        headers = {'Accept': 'application/json'}
        # For public endpoints, no API key is needed for now
        return headers

    def _generate_signature(self, timestamp: str, method: str, endpoint: str, params: Dict = None, body: str = None) -> str:
        """Generate signed message for authenticated requests."""
        if not self.credentials or not self.credentials.api_secret:
            return None

        message = timestamp + method + endpoint
        if params: # For query parameters in GET requests (though Bitget might use body for POST auth as well - verify)
            message += '?' + '&'.join([f"{k}={v}" for k, v in sorted(params.items())]) # Ensure consistent parameter order (if needed)
        if body: # For request body (POST, PUT, DELETE - verify if Bitget uses body for signature)
            message += body

        signature = hmac.new(
            self.credentials.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _convert_market_symbol(self, market: str) -> str:
        """Convert internal market symbol to Bitget format."""
        # Assuming Bitget spot symbols are like BTCUSDT_SPBL (COINQUOTE_SPBL)
        # and futures are like BTCUSDT (COIN-QUOTE) for contract market
        return market.replace('-', '').upper() + "_SPBL" # Tentative spot symbol conversion

    def _parse_raw_candle(self, raw_data: List, market: str, resolution: str) -> StandardizedCandle:
        """Parse raw candle data into StandardizedCandle format."""
        # Bitget Candle Data Format (verify from docs):
        # [ts, open, close, high, low, volume, ...] - Verify order and data types
        try:
            candle = StandardizedCandle(
                timestamp=self.standardize_timestamp(int(raw_data[0])), # Millisecond timestamp
                open=float(raw_data[1]),
                high=float(raw_data[2]),
                low=float(raw_data[3]),
                close=float(raw_data[4]),
                volume=float(raw_data[5]), # Verify volume is base or quote currency
                source='bitget',
                resolution=resolution,
                market=market,
                raw_data=raw_data
            )
            self.validate_candle(candle)
            return candle
        except (IndexError, ValueError) as e:
            raise ValidationError(f"Error parsing Bitget candle data: {str(e)}, Raw Data: {raw_data}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error parsing Bitget candle: {str(e)}, Raw Data: {raw_data}")


    async def fetch_historical_candles(self, market: str, time_range: TimeRange, resolution: str) -> List[StandardizedCandle]:
        """Fetch historical candle data from Bitget."""
        self.validate_market(market)
        bitget_symbol = self._convert_market_symbol(market) # Convert market symbol
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        candles: List[StandardizedCandle] = []
        start_timestamp_ms = int(time_range.start.timestamp() * 1000) # to milliseconds
        end_timestamp_ms = int(time_range.end.timestamp() * 1000)

        limit = 1000 # Max limit for Bitget historical candles (verify in docs)

        current_start_ms = start_timestamp_ms

        try:
            while current_start_ms < end_timestamp_ms:
                params = {
                    'symbol': bitget_symbol,
                    'period': interval,
                    'limit': limit,
                    'startTime': current_start_ms, # Bitget uses startTime/endTime
                    'endTime': end_timestamp_ms
                }
                print(f"Debug Bitget: fetch_historical_candles - Params: {params}") # DEBUG

                response_data = await self._make_request(
                    method='GET',
                    endpoint='/api/spot/v1/market/history-candles', # Bitget endpoint for historical candles (verify)
                    params=params,
                    headers=self._get_headers()
                )
                print(f"Debug Bitget: fetch_historical_candles - Response Data: {response_data}") # DEBUG

                if not response_data or not response_data['data']: # Check for empty data response
                    logger.info(f"No more historical data from {datetime.fromtimestamp(current_start_ms/1000)} to {datetime.fromtimestamp(end_timestamp_ms/1000)} for {market} {resolution}")
                    break # No more data

                raw_candles = response_data['data'] # Adjust based on actual response structure
                for raw_candle in raw_candles:
                    try:
                        candle = self._parse_raw_candle(raw_candle, market, resolution)
                        candles.append(candle)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid candle: {e}")

                if len(raw_candles) < limit:
                    logger.info(f"Less than limit ({limit}) candles received, assuming end of data for this chunk.")
                    break # Less than requested limit, assume no more data in this range

                # Bitget historical API seems to return data in ascending order (oldest first).
                # Next start time should be the timestamp of the last candle + 1ms to avoid overlap.
                last_candle_timestamp = int(raw_candles[-1][0]) # Get timestamp of last candle
                current_start_ms = last_candle_timestamp + 1 # Move start time for next request

                await self._handle_rate_limit()


        except RateLimitError:
            logger.warning("Bitget rate limit hit during historical data fetch.")
            raise # Re-raise to allow for retry/handling upstream
        except ExchangeError as e:
            raise ExchangeError(f"Bitget historical data fetch failed: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error fetching Bitget historical candles: {e}")

        return candles


    async def fetch_live_candles(self, market: str, resolution: str) -> StandardizedCandle:
        """Fetch live candle data from Bitget."""
        self.validate_market(market)
        bitget_symbol = self._convert_market_symbol(market)
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        try:
            params = {
                'symbol': bitget_symbol,
                'period': interval, # Correct parameter name? Verify Bitget docs
                'limit': 1 # Get only the latest candle
            }
            print(f"Debug Bitget: fetch_live_candles - Params: {params}") # DEBUG

            response_data = await self._make_request(
                method='GET',
                endpoint='/api/spot/v1/market/candles', # Or /api/spot/v1/ticker ? Verify endpoint
                params=params,
                headers=self._get_headers()
            )
            print(f"Debug Bitget: fetch_live_candles - Response Data: {response_data}") # DEBUG

            if not response_data or not response_data['data']:
                raise ExchangeError(f"No live data available for {market} from Bitget")

            raw_candles = response_data['data'] # Assuming data is list of candles
            if raw_candles:
                return self._parse_raw_candle(raw_candles[0], market, resolution) # Parse the first (and should be only) candle
            else:
                raise ExchangeError(f"No candle data returned in live response for {market}")


        except ExchangeError as e:
            raise ExchangeError(f"Bitget live data fetch failed: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error fetching Bitget live candles: {e}")


    async def get_markets(self) -> List[str]:
        """Get available markets from Bitget."""
        try:
            response_data = await self._make_request(
                method='GET',
                endpoint='/api/spot/v1/market/symbols', # Endpoint to get symbols - VERIFY in Bitget API docs
                headers=self._get_headers()
            )
            print(f"Debug Bitget: get_markets - Response Data: {response_data}") # DEBUG

            if not response_data or not response_data['data']:
                raise ExchangeError("Could not retrieve market list from Bitget")

            markets = []
            symbols_data = response_data['data'] # Adjust based on actual response structure
            for symbol_info in symbols_data:
                # Assuming symbol format is like "BTCUSDT_SPBL" and we need "BTC-USDT"
                symbol_str = symbol_info['symbol'] # Example: "BTCUSDT_SPBL"
                if symbol_str.endswith("_SPBL"): # Spot symbol format?
                    base_currency = symbol_str[:-5][:symbol_str[:-5].find('USDT')].upper() # Extract 'BTC' from 'BTCUSDT_SPBL'
                    quote_currency = 'USDT' # Quote is USDT for these spot pairs - verify
                    markets.append(f"{base_currency}-{quote_currency}") # Format as "BTC-USDT"
                else:
                    logger.warning(f"Unexpected symbol format: {symbol_str}, skipping")


            return markets

        except ExchangeError as e:
            raise ExchangeError(f"Failed to fetch Bitget markets: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error getting Bitget markets: {e}")


    async def get_exchange_info(self) -> Dict:
        """Get exchange information from Bitget."""
        try:
            response_data = await self._make_request(
                method='GET',
                endpoint='/api/spot/v1/common/timestamp', # Or exchange info endpoint? Verify Bitget API for general info
                headers=self._get_headers()
            )
            server_time = response_data['serverTime'] if response_data and 'serverTime' in response_data else None # Adjust key if needed
            print(f"Debug Bitget: get_exchange_info - Timestamp Response: {response_data}") # DEBUG


            exchange_info = {
                "name": self.name,
                "markets": await self.get_markets(),
                "timeframes": list(self.timeframe_map.values()),
                "has_live_data": True, # Assume live data is supported
                "rate_limit": self.rate_limit, # Use default rate limit from base class for now
                "server_time": server_time, # Server time from endpoint
                "exchange_filters": [] # Bitget might not have exchange filters like Binance in same format - adjust as needed.
            }
            return exchange_info

        except ExchangeError as e:
            raise ExchangeError(f"Failed to fetch Bitget exchange info: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error getting Bitget exchange info: {e}")


    async def _make_request(self, method: str, endpoint: str, params: Dict = None, headers: Dict = None) -> Union[Dict, List, None]:
        """
        Handles making requests to the Bitget API, including rate limiting and error handling.
        """
        url = self.base_url + endpoint # Construct full URL

        async with self._session_manager.get_session() as session:
            try:
                async with session.request(method, url, params=params, headers=headers, timeout=10) as response: # Added timeout
                    print(f"Debug Bitget: _make_request - URL: {url}, Status: {response.status}") # DEBUG URL and Status
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    # Check for rate limit headers and handle them if necessary (if Bitget provides specific headers)
                    # Example (might need adjustment for Bitget's actual headers):
                    # if 'X-RateLimit-Remaining' in response.headers:
                    #     remaining = int(response.headers['X-RateLimit-Remaining'])
                    #     if remaining < some_threshold: # Define a threshold
                    #         retry_after = int(response.headers.get('Retry-After', 1)) # Or use default
                    #         logger.warning(f"Approaching rate limit, waiting {retry_after} seconds.")
                    #         await asyncio.sleep(retry_after) # Wait before next request

                    return await response.json()

            except asyncio.TimeoutError:
                logger.warning(f"Timeout error for Bitget API request to {url}")
                raise ExchangeError(f"Timeout error for URL: {url}") # Re-raise as ExchangeError
            except aiohttp.ClientError as e:
                logger.error(f"Bitget API Client error for {url}: {e}")
                raise ExchangeError(f"API request failed: {e}") # Re-raise as ExchangeError
            except ContentTypeError: # Changed to ContentTypeError from aiohttp
                logger.error(f"JSON decode error for Bitget API response from {url}")
                raise ExchangeError(f"Failed to decode JSON response from {url}")