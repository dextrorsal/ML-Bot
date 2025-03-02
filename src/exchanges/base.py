"""
Base exchange handler providing common functionality for all exchanges.
"""
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Dict, Optional, Union
import aiohttp
import asyncio

from ..core.models import StandardizedCandle, ExchangeCredentials, TimeRange
from ..core.exceptions import ValidationError, ExchangeError, RateLimitError, ApiError
from ..core.config import ExchangeConfig

logger = logging.getLogger(__name__)

class BaseExchangeHandler(ABC):
    """Abstract base class for all exchange handlers."""

    def __init__(self, config: ExchangeConfig):
        """Initialize the exchange handler with configuration."""
        self.config = config
        self.name = config.name
        self.credentials = config.credentials
        self.rate_limit = config.rate_limit
        self.markets = config.markets
        self.base_url = config.base_url
        
        # Rate limiting
        self._last_request_time = 0
        self._request_interval = 1.0 / self.rate_limit if self.rate_limit > 0 else 0
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start the exchange handler."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        logger.info(f"Started {self.name} exchange handler")

    async def stop(self):
        """Stop the exchange handler."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info(f"Stopped {self.name} exchange handler")

    @abstractmethod
    async def fetch_historical_candles(
        self,
        market: str,
        time_range: TimeRange,
        resolution: str
    ) -> List[StandardizedCandle]:
        """Fetch historical candle data."""
        pass

    @abstractmethod
    async def fetch_live_candles(
        self,
        market: str,
        resolution: str
    ) -> StandardizedCandle:
        """Fetch live candle data."""
        pass

    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: int = 30
    ) -> Dict:
        """Make an HTTP request with rate limiting and error handling."""
        if self._session is None:
            await self.start()

        # Debug logging
        #print(f"\nDEBUG REQUEST:")
        #print(f"URL: {self.base_url}{endpoint}")
        #print(f"Method: {method}")
        #print(f"Params: {params}")
        #print(f"Headers: {headers}")

        try:
            url = f"{self.base_url}{endpoint}"
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=data,
                timeout=timeout
            ) as response:
                #print(f"\nDEBUG RESPONSE:")
                #print(f"Status: {response.status}")
                
                # Get response text
                try:
                    response_text = await response.text()
                    #print(f"Response Text: {response_text}")
                except:
                    #print("Could not get response text")
                    pass

                if response.status == 401:
                    #print("Got 401 Unauthorized!")
                    error_text = await response.text()
                    #print(f"Error details: {error_text}")
                    raise ApiError(f"Unauthorized: {error_text}")

                if response.status != 200:
                    error_text = await response.text()
                    #print(f"Error response: {error_text}")
                    raise ApiError(f"coinbase API error: {response.status} - {error_text}")

                # Try to parse JSON response
                try:
                    json_response = await response.json()
                    #print(f"Parsed JSON: {json.dumps(json_response, indent=2)}")
                    return json_response
                except Exception as e:
                    #print(f"Failed to parse JSON: {str(e)}")
                    raise

        except aiohttp.ClientError as e:
            #print(f"Request failed with error: {str(e)}")
            raise ExchangeError(f"Request failed: {str(e)}")
        except asyncio.TimeoutError:
            #print("Request timed out")
            raise ExchangeError("Request timed out")
        except Exception as e:
            #print(f"Unexpected error: {str(e)}")
            raise

        # Rate limiting
        await self._handle_rate_limit()

        try:
            url = f"{self.base_url}{endpoint}"
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=data,
                timeout=timeout
            ) as response:
                # Update last request time
                self._last_request_time = datetime.now().timestamp()

                # Handle rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    raise RateLimitError(
                        f"{self.name} rate limit exceeded. Retry after {retry_after} seconds"
                    )

                # Handle other errors
                if response.status >= 400:
                    error_text = await response.text()
                    raise ApiError(
                        f"{self.name} API error: {response.status} - {error_text}"
                    )

                return await response.json()

        except aiohttp.ClientError as e:
            raise ExchangeError(f"{self.name} request failed: {str(e)}")
        except asyncio.TimeoutError:
            raise ExchangeError(f"{self.name} request timed out")
        except Exception as e:
            if isinstance(e, (RateLimitError, ApiError, ExchangeError)):
                raise
            raise ExchangeError(f"Unexpected error in {self.name}: {str(e)}")

    async def _handle_rate_limit(self):
        """Handle rate limiting between requests."""
        if self._last_request_time == 0:
            return

        current_time = datetime.now().timestamp()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self._request_interval:
            delay = self._request_interval - time_since_last_request
            logger.debug(f"Rate limiting {self.name}: waiting {delay:.2f} seconds")
            await asyncio.sleep(delay)

    def validate_market(self, market: str):
        """
        Validate market symbol.
        
        Args:
            market: Market symbol (can be in exchange-specific or standard format)
        
        Raises:
            ValidationError: If market is not supported by this exchange
        """
        # First check if market is directly in the markets list (exchange-specific format)
        if market in self.markets:
            return
        
        # If not found directly, try checking if it can be converted from standard format
        if self.validate_standard_symbol(market):
            return
            
        # If we get here, the market is not supported
        raise ValidationError(f"Invalid market {market} for {self.name}")

    def validate_standard_symbol(self, standard_symbol: str) -> bool:
        """
        Validate if a standard symbol is supported by this exchange.
        
        Args:
            standard_symbol: Symbol in standard format (e.g., BTC-USD) or exchange format (e.g., BTCUSDT)
            
        Returns:
            True if supported, False otherwise
        """
        self._init_symbol_mapper()
        
        # First, check if the symbol is already in the exchange's native format
        if standard_symbol in self.markets:
            return True
            
        # If not directly found, try to convert from standard format
        try:
            exchange_symbol = self._symbol_mapper.to_exchange_symbol(self.name, standard_symbol)
            return exchange_symbol in self.markets
        except ValueError:
            return False

    def standardize_timestamp(self, ts: Union[int, float, str, datetime]) -> datetime:
        """Convert various timestamp formats to datetime."""
        try:
            if isinstance(ts, datetime):
                return ts
            elif isinstance(ts, str):
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            elif isinstance(ts, (int, float)):
                # Handle milliseconds vs seconds
                if len(str(int(ts))) > 10:
                    return datetime.fromtimestamp(ts / 1000)
                return datetime.fromtimestamp(ts)
            raise ValidationError(f"Unsupported timestamp format: {ts}")
        except Exception as e:
            raise ValidationError(f"Failed to standardize timestamp: {str(e)}")

    def validate_candle(self, candle: StandardizedCandle):
        """Validate candle data."""
        try:
            candle.validate()
        except ValidationError as e:
            raise ValidationError(f"Invalid candle data for {self.name}: {str(e)}")

    def format_timeframe(self, resolution: str) -> str:
        """Format timeframe for exchange API."""
        # Override in specific exchange handlers if needed
        return resolution

    def _init_symbol_mapper(self):
        """Initialize the symbol mapper if not already set globally."""
        from ..core.symbol_mapper import SymbolMapper
        if not hasattr(self, '_symbol_mapper'):
            # Create a new mapper instance if needed
            self._symbol_mapper = SymbolMapper()
            
            # Register known markets for this exchange
            if self.markets:
                self._symbol_mapper.register_exchange(self.name, self.markets)

    def convert_to_exchange_symbol(self, standard_symbol: str) -> str:
        """
        Convert a standard symbol to this exchange's specific format.
        
        Args:
            standard_symbol: Symbol in standard format (e.g., BTC-USD)
            
        Returns:
            Symbol in exchange-specific format
        """
        self._init_symbol_mapper()
        return self._symbol_mapper.to_exchange_symbol(self.name, standard_symbol)

    def convert_from_exchange_symbol(self, exchange_symbol: str) -> str:
        """
        Convert this exchange's symbol to standard format.
        
        Args:
            exchange_symbol: Symbol in exchange-specific format
            
        Returns:
            Symbol in standard format (e.g., BTC-USD)
        """
        self._init_symbol_mapper()
        return self._symbol_mapper.from_exchange_symbol(self.name, exchange_symbol)

