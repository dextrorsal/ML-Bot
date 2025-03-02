"""
DriftHandler with proper timezone handling to fix datetime comparison issues.
"""
import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime, timezone, timedelta
from io import StringIO
import asyncio

from src.core.config import ExchangeConfig
from src.core.models import TimeRange, StandardizedCandle
from src.exchanges.base import BaseExchangeHandler
from src.core.exceptions import ExchangeError, ValidationError

logger = logging.getLogger(__name__)

class DriftHandler(BaseExchangeHandler):
    """Handler for the Drift exchange with timezone-aware datetime handling."""

    def __init__(self, config: ExchangeConfig):
        """Initialize Drift handler with configuration."""
        super().__init__(config)
        self.base_url = self.config.base_url  # e.g. "https://test.exchange.com"
        logger.info(f"DriftHandler initialized with URL: {self.base_url}")

    def _get_market_key(self, market: str) -> str:
        """
        Convert market symbol to a 'marketKey' for the S3 path.
        For example: "SOL-PERP" -> "perp_0".
        """
        if market.endswith("-PERP"):
            market_indices = {
                "SOL-PERP": "perp_0",
                "BTC-PERP": "perp_1",
                "ETH-PERP": "perp_2",
            }
            return market_indices.get(market, "perp_0")
        else:
            return "spot_0"

    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """
        Ensure datetime is timezone-aware by adding UTC timezone if naive.
        """
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    async def get_markets(self) -> List[str]:
        """
        Return a list of available markets.
        (Minimal implementation: use the markets from configuration.)
        """
        return self.config.markets

    async def get_exchange_info(self) -> Dict:
        """
        Return a dictionary with exchange information.
        (Minimal implementation: return the config name and markets.)
        """
        return {"name": self.config.name, "markets": self.config.markets}

    async def fetch_historical_candles(
        self,
        market: str,
        time_range: TimeRange,
        resolution: str
    ) -> List[StandardizedCandle]:
        """
        Fetch historical candlestick data from Drift using the S3 bucket.
        Uses the /candle-history/{year}/{market_key}/{resolution}.csv structure.
        """
        candles = []
        market_key = self._get_market_key(market)
        
        # Ensure time_range datetimes are timezone-aware
        start_time = self._ensure_timezone_aware(time_range.start)
        end_time = self._ensure_timezone_aware(time_range.end)
        
        logger.info(f"Fetching historical data for {market} (market_key: {market_key}) with resolution {resolution}")
        logger.info(f"Time range: {start_time} to {end_time}")
        
        for year in range(start_time.year, end_time.year + 1):
            endpoint = f"/candle-history/{year}/{market_key}/{resolution}.csv"
            logger.info(f"Requesting endpoint: {endpoint}")
            
            try:
                csv_text = await self._make_request(method="GET", endpoint=endpoint, timeout=30)
                csv_text = csv_text.strip()  # Remove extra whitespace
                logger.info(f"Successfully fetched data for {market} in {year}")
                
                csv_preview = csv_text.split('\n')[:5]
                logger.debug(f"CSV preview: {csv_preview}")
                
                df = pd.read_csv(StringIO(csv_text))
                logger.info(f"Parsed CSV with {len(df)} rows and columns: {df.columns.tolist()}")
                
                if not df.empty:
                    logger.debug(f"Sample row: {df.iloc[0].to_dict()}")
                
                for _, row in df.iterrows():
                    try:
                        candle = self._parse_raw_candle(row.to_dict(), market, resolution)
                        candle_time = self._ensure_timezone_aware(candle.timestamp)
                        # Zero microseconds for comparison so integer timestamps match
                        if (start_time.replace(microsecond=0) <= candle_time.replace(microsecond=0) <= end_time.replace(microsecond=0)):
                            candles.append(candle)
                        else:
                            logger.debug(f"Candle outside time range: {candle_time}")
                    except Exception as e:
                        logger.warning(f"Error processing candle: {e}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {market} in {year}: {e}")
        
        logger.info(f"Total candles collected: {len(candles)}")
        return candles

    def _parse_raw_candle(self, raw_data: Dict, market: str, resolution: str) -> StandardizedCandle:
        """
        Parse a single raw candle from Drift data.
        Handles both 'start' and 'ts' timestamp fields.
        """
        try:
            ts = None
            if 'start' in raw_data:
                ts = int(raw_data['start'])  # milliseconds
            elif 'ts' in raw_data:
                ts = int(raw_data['ts']) * 1000  # convert seconds -> milliseconds
            else:
                logger.debug(f"No timestamp field in raw data. Keys: {raw_data.keys()}")
                raise ValidationError("No timestamp field found in data")
                    
            timestamp = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            open_price = float(raw_data.get('fillOpen', raw_data.get('open', 0)))
            high_price = float(raw_data.get('fillHigh', raw_data.get('high', 0)))
            low_price = float(raw_data.get('fillLow', raw_data.get('low', 0)))
            close_price = float(raw_data.get('fillClose', raw_data.get('close', 0)))
            volume = float(raw_data.get('baseVolume', raw_data.get('volume', 0)))
            
            candle = StandardizedCandle(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                source="drift",
                resolution=resolution,
                market=market,
                raw_data=raw_data
            )
            return candle
            
        except Exception as e:
            logger.debug(f"Raw data causing error: {raw_data}")
            raise ValidationError(f"Error parsing candle: {e}")

    async def fetch_live_candles(
        self,
        market: str,
        resolution: str,
    ) -> StandardizedCandle:
        """
        Fetches the latest candlestick data for a market from Drift.
        Simplified implementation that gets the latest candle from historical data.
        """
        market_key = self._get_market_key(market)
        current_year = datetime.now().year
        
        endpoint = f"/candle-history/{current_year}/{market_key}/{resolution}.csv"
        logger.info(f"Fetching live data from endpoint: {endpoint}")
        
        try:
            csv_text = await self._make_request(method="GET", endpoint=endpoint, timeout=10)
            csv_text = csv_text.strip()
            df = pd.read_csv(StringIO(csv_text))
            
            if df.empty:
                logger.warning("Empty data returned")
                raise ExchangeError("No live candle data available")
                
            if 'start' in df.columns:
                df['timestamp'] = df['start']
            elif 'ts' in df.columns:
                df['timestamp'] = df['ts'] * 1000  # convert to ms
                
            df = df.sort_values(by='timestamp', ascending=False)
            latest_row = df.iloc[0]
            
            return self._parse_raw_candle(latest_row.to_dict(), market, resolution)
            
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            raise ExchangeError(f"Failed to fetch live candle: {e}")
