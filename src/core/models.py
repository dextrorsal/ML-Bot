"""
Core data models for the Ultimate Data Fetcher.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Union, List, Any

from .exceptions import ValidationError

@dataclass
class StandardizedCandle:
    """Standardized format for OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    resolution: str
    market: str
    raw_data: Dict[str, Any]

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate candle data integrity."""
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object.")
        if any(not isinstance(x, (int, float)) for x in [self.open, self.high, self.low, self.close, self.volume]):
            raise ValidationError("OHLCV values must be numeric.")
        if self.volume < 0:
            raise ValidationError("Volume cannot be negative.")

    @classmethod
    def create_empty(cls, market: str, source: str, resolution: str) -> 'StandardizedCandle':
        """Create an empty candle with zeros."""
        now = datetime.now(timezone.utc)
        return cls(
            timestamp=now,
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0.0,
            source=source,
            resolution=resolution,
            market=market,
            raw_data={}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of candle."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'source': self.source,
            'resolution': self.resolution,
            'market': self.market,
            'raw_data': self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardizedCandle':
        """Create a StandardizedCandle from a dictionary."""
        try:
            timestamp = datetime.fromisoformat(data['timestamp'])
            return cls(**{**data, 'timestamp': timestamp})
        except (ValueError, KeyError, TypeError) as e:  # Include TypeError
            raise ValidationError(f"Invalid candle data: {e}")


@dataclass
class TimeRange:
    """Time range for historical data fetching."""
    start: datetime
    end: datetime

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise ValidationError("Start and end times must be datetime objects.")

        if self.start.tzinfo is None or self.end.tzinfo is None:  # Check for timezone awareness
            raise ValidationError("Start and end times must be timezone-aware.")

        if self.start > self.end:
            raise ValidationError("Start time must be before end time.")

        if self.end > datetime.now(timezone.utc):
            raise ValidationError("End time cannot be in the future.")

    def to_dict(self) -> Dict[str, Any]:
        return {'start': self.start.isoformat(), 'end': self.end.isoformat()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeRange':
        try:
            start = datetime.fromisoformat(data['start'])
            end = datetime.fromisoformat(data['end'])
            return cls(start, end)
        except (ValueError, KeyError, TypeError) as e: # Include TypeError
            raise ValidationError(f"Invalid TimeRange data: {e}")


@dataclass
class ExchangeCredentials:
    """Credentials for exchange API access."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    additional_params: Dict = None

@dataclass
class Market:
    """Market information."""
    symbol: str
    base_asset: str
    quote_asset: str
    min_price: float
    min_quantity: float
    price_decimals: int
    quantity_decimals: int
    status: str = "active"