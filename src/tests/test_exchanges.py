"""
Tests for exchange handlers.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from io import StringIO

from src.core.config import ExchangeConfig, ExchangeCredentials
from src.core.models import TimeRange
from src.exchanges import DriftHandler, BinanceHandler, CoinbaseHandler

# ---------------------------------------------------------------------------
# Fixture: Mock exchange configuration
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_config():
    """Mock exchange configuration."""
    config = ExchangeConfig(
        name="test_exchange",
        credentials=ExchangeCredentials(
            api_key="test_key",
            api_secret="test_secret",
            additional_params={"passphrase": "test_passphrase"}
        ),
        rate_limit=10,
        markets=["BTC-PERP", "ETH-PERP", "SOL-PERP", "BTC-USD", "ETH-USD", "SOL-USD"],
        base_url="https://test.exchange.com",
        enabled=True
    )
    return config

# ---------------------------------------------------------------------------
# Fixture: Provide a one-day time range (timezone-aware)
# ---------------------------------------------------------------------------
@pytest.fixture
def time_range():
    """Test time range."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=1)
    return TimeRange(start=start, end=end)

# ---------------------------------------------------------------------------
# Binance Handler Fixture using async initialization
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def binance_handler():
    """Create a Binance handler with test configuration."""
    config = ExchangeConfig(
        name="binance",
        credentials=None,  # Public endpoints require no credentials
        rate_limit=10,
        markets=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        base_url="https://api.binance.com",
        enabled=True
    )
    handler = BinanceHandler(config)
    await handler.start()  # Initialize the handler
    yield handler
    await handler.stop()

# ---------------------------------------------------------------------------
# Tests for DriftHandler
# ---------------------------------------------------------------------------
class TestDriftHandler:
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_get_markets(self, mock_config):
        """Test fetching markets from Drift."""
        handler = DriftHandler(mock_config)
        await handler.start()
        markets = await handler.get_markets()
        assert isinstance(markets, list)
        assert len(markets) > 0
        # Expect at least one of the configured markets to be present.
        assert "SOL-PERP" in markets
        await handler.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_get_exchange_info(self, mock_config):
        """Test fetching exchange info from Drift."""
        handler = DriftHandler(mock_config)
        await handler.start()
        info = await handler.get_exchange_info()
        assert isinstance(info, dict)
        assert info["name"] == "test_exchange"
        assert "markets" in info
        assert isinstance(info["markets"], list)
        assert len(info["markets"]) > 0
        await handler.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_fetch_historical_candles(self, mock_config, time_range):
        """Test fetching historical candles from Drift."""
        handler = DriftHandler(mock_config)
        await handler.start()
        start_timestamp_ms = int(time_range.start.timestamp() * 1000)
        mock_csv_response = (
            f"start,open,high,low,close,volume\n"
            f"{start_timestamp_ms},100.0,105.0,95.0,102.0,1000.0\n"
        )
        with patch.object(handler, '_make_request', return_value=mock_csv_response):
            candles = await handler.fetch_historical_candles(
                market="BTC-PERP",
                time_range=time_range,
                resolution="15"
            )
            assert len(candles) > 0
            assert candles[0].market == "BTC-PERP"
        await handler.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_fetch_live_candles(self, mock_config):
        """Test fetching live candles from Drift."""
        handler = DriftHandler(mock_config)
        await handler.start()
        current_timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        mock_csv_response = (
            f"start,open,high,low,close,volume\n"
            f"{current_timestamp_ms},100.0,105.0,95.0,102.0,1000.0\n"
        )
        with patch.object(handler, '_make_request', return_value=mock_csv_response):
            candle = await handler.fetch_live_candles(
                market="BTC-PERP",
                resolution="1"
            )
            assert candle.market == "BTC-PERP"
            assert candle.resolution == "1"
        await handler.stop()

# ---------------------------------------------------------------------------
# Tests for BinanceHandler
# ---------------------------------------------------------------------------
class TestBinanceHandler:
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_fetch_historical_candles(self, binance_handler, time_range):
        """Test fetching historical candles from Binance."""
        call_count = 0
        def mock_response_once(*args, **kwargs):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return [[
                    int(time_range.start.timestamp() * 1000),
                    "100.0",
                    "105.0",
                    "95.0",
                    "102.0",
                    "1000.0",
                    int(time_range.start.timestamp() * 1000),
                    "100000.0",
                    100,
                    "500.0",
                    "50000.0",
                    "0"
                ]]
            else:
                return []
        with patch.object(binance_handler, '_get_headers', return_value={'Accept': 'application/json'}):
            with patch.object(binance_handler, '_make_request', side_effect=mock_response_once) as mock_req:
                candles = await binance_handler.fetch_historical_candles(
                    market="BTCUSDT",
                    time_range=time_range,
                    resolution="15"
                )
                assert mock_req.call_count >= 1
                assert len(candles) > 0
                assert "BTCUSDT" in candles[0].market

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_live_candles(self, binance_handler):
        """Test fetching live candles from Binance."""
        with patch.object(binance_handler, '_get_headers', return_value={'Accept': 'application/json'}):
            mock_response = [[
                int(datetime.now(timezone.utc).timestamp() * 1000),
                "100.0",
                "105.0",
                "95.0",
                "102.0",
                "1000.0",
                int(datetime.now(timezone.utc).timestamp() * 1000),
                "100000.0",
                100,
                "500.0",
                "50000.0",
                "0"
            ]]
            with patch.object(binance_handler, '_make_request', return_value=mock_response) as mock_req:
                candle = await binance_handler.fetch_live_candles(
                    market="BTCUSDT",
                    resolution="1"
                )
                mock_req.assert_called_once()
                assert "BTCUSDT" in candle.market
                assert candle.resolution == "1"

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_rate_limiting(self, binance_handler):
        """Test rate limiting functionality."""
        binance_handler._last_request_time = datetime.now().timestamp()
        original_rate_limit = binance_handler.rate_limit
        binance_handler.rate_limit = 1
        binance_handler._request_interval = 1.0

        start_time = datetime.now()
        mock_response = [[
            int(datetime.now(timezone.utc).timestamp() * 1000),
            "100.0",
            "105.0",
            "95.0",
            "102.0",
            "1000.0",
            int(datetime.now(timezone.utc).timestamp() * 1000),
            "100000.0",
            100,
            "500.0",
            "50000.0",
            "0"
        ]]
        with patch.object(binance_handler, '_get_headers', return_value={'Accept': 'application/json'}):
            with patch.object(binance_handler, '_make_request', return_value=mock_response):
                for _ in range(2):
                    await binance_handler._handle_rate_limit()
                    binance_handler._last_request_time = datetime.now().timestamp()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        binance_handler.rate_limit = original_rate_limit
        binance_handler._request_interval = 1.0 / original_rate_limit if original_rate_limit > 0 else 0
        assert duration >= 0.95

# ---------------------------------------------------------------------------
# Tests for CoinbaseHandler
# ---------------------------------------------------------------------------
class TestCoinbaseHandler:
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_fetch_historical_candles(self, mock_config, time_range):
        """Test fetching historical candles from Coinbase."""
        handler = CoinbaseHandler(mock_config)
        await handler.start()
        with patch.object(handler, 'validate_market'):
            with patch.object(handler, '_get_headers', return_value={'Accept': 'application/json'}):
                mock_response = [[
                    int(time_range.start.timestamp()),
                    100.0,
                    105.0,
                    95.0,
                    102.0,
                    1000.0
                ]]
                with patch.object(handler, '_make_request', return_value=mock_response) as mock_req:
                    candles = await handler.fetch_historical_candles(
                        market="BTC-USD",
                        time_range=time_range,
                        resolution="15"
                    )
                    mock_req.assert_called_once()
                    assert len(candles) > 0
                    assert candles[0].market == "BTC-USD"
        await handler.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_authentication(self, mock_config):
        """Test authentication header generation for Coinbase."""
        mock_config.credentials.additional_params = {"passphrase": "test_passphrase"}
        handler = CoinbaseHandler(mock_config)
        await handler.start()
        with patch.object(handler, '_generate_signature', return_value='test_signature'):
            headers = handler._get_headers('GET', '/products')
            assert 'CB-ACCESS-KEY' in headers
            assert 'CB-ACCESS-SIGN' in headers
            assert 'CB-ACCESS-TIMESTAMP' in headers
            assert 'CB-ACCESS-PASSPHRASE' in headers
        await handler.stop()
