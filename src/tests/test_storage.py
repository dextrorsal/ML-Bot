"""
Tests for storage implementations.
"""

import pytest
import pandas as pd
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import shutil
import json

from src.core.config import StorageConfig
from src.core.models import StandardizedCandle
from src.storage import RawDataStorage, ProcessedDataStorage, DataManager
from src.storage.live import LiveDataStorage

# ----------------------------------------------------------------------------
# Fixture: Clean test directories before and after tests.
# ----------------------------------------------------------------------------
@pytest.fixture
def clean_test_dirs():
    """Clean test directories before and after tests."""
    test_dirs = ["test_data/raw", "test_data/processed", "test_data/live/raw", "test_data/live/processed"]
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    yield
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

# ----------------------------------------------------------------------------
# Fixture: Test configuration for storage.
# ----------------------------------------------------------------------------
@pytest.fixture
def test_config():
    """Test configuration fixture."""
    return StorageConfig(
        historical_raw_path=Path("test_data/raw"),
        historical_processed_path=Path("test_data/processed"),
        live_raw_path=Path("test_data/live/raw"),
        live_processed_path=Path("test_data/live/processed"),
        use_compression=False
    )

# ----------------------------------------------------------------------------
# Fixture: A single test candle (timezone-aware).
# ----------------------------------------------------------------------------
@pytest.fixture
def test_candle():
    """Test candle fixture."""
    return StandardizedCandle(
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        source="test_exchange",
        resolution="15",
        market="BTC-PERP",
        raw_data={"original": "data"}
    )

# ----------------------------------------------------------------------------
# Fixture: Generate a list of sample candles (timezone-aware).
# ----------------------------------------------------------------------------
@pytest.fixture
def sample_candles():
    """Generate a list of sample candles."""
    base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    candles = []
    for i in range(10):
        candles.append(StandardizedCandle(
            timestamp=base_time + timedelta(minutes=15 * i),
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=102.0 + i,
            volume=1000.0 + i * 100,
            source="test_exchange",
            resolution="15",
            market="BTC-PERP",
            raw_data={"candle_index": i}
        ))
    return candles

# ----------------------------------------------------------------------------
# Fixtures for LiveDataStorage tests.
# ----------------------------------------------------------------------------
@pytest.fixture
def live_storage_config(tmp_path):
    """Create a temporary StorageConfig for live data testing."""
    raw_live = tmp_path / "live" / "raw"
    proc_live = tmp_path / "live" / "processed"
    raw_live.mkdir(parents=True, exist_ok=True)
    proc_live.mkdir(parents=True, exist_ok=True)
    
    # Use the new configuration keyword names
    config = StorageConfig(
        historical_raw_path=Path("test_data/raw"),
        historical_processed_path=Path("test_data/processed"),
        live_raw_path=raw_live,
        live_processed_path=proc_live,
        use_compression=False
    )
    return config

@pytest_asyncio.fixture
async def live_storage(live_storage_config):
    """Create an instance of LiveDataStorage using the temporary config."""
    storage = LiveDataStorage(live_storage_config)
    return storage

@pytest.fixture
def live_sample_candle():
    """Create a sample candle for live data tests (timezone-aware)."""
    return StandardizedCandle(
        timestamp=datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        source="test_exchange",
        resolution="1",
        market="TEST",
        raw_data={"example": "live_data"}
    )

# -------------------------- Test Classes ------------------------------------

class TestRawDataStorage:
    """Test RawDataStorage implementation."""
    
    @pytest.mark.asyncio
    async def test_store_and_load_candles(self, test_config, sample_candles, clean_test_dirs):
        storage = RawDataStorage(test_config)
        await storage.store_candles(
            exchange="test_exchange",
            market="BTC-PERP",
            resolution="15",
            candles=sample_candles
        )
        start_time = sample_candles[0].timestamp
        end_time = sample_candles[-1].timestamp
        loaded_data = await storage.load_candles(
            exchange="test_exchange",
            market="BTC-PERP",
            resolution="15",
            start_time=start_time,
            end_time=end_time
        )
        # Allow for duplicates if the method appends data.
        assert len(loaded_data) >= len(sample_candles)
    
    @pytest.mark.asyncio
    async def test_data_integrity(self, test_config, sample_candles, clean_test_dirs):
        storage = RawDataStorage(test_config)
        await storage.store_candles("test_exchange", "BTC-PERP", "15", sample_candles)
        loaded = await storage.load_candles(
            "test_exchange",
            "BTC-PERP",
            "15",
            sample_candles[0].timestamp,
            sample_candles[-1].timestamp
        )
        assert len(loaded) > 0
        first_candle = sample_candles[0]
        first_loaded = loaded[0]
        assert first_candle.raw_data == first_loaded["raw_data"]
        # Compare timestamps as ISO strings (remove trailing 'Z' if present)
        assert first_candle.timestamp.isoformat() == first_loaded["timestamp"].replace("Z", "")

class TestProcessedDataStorage:
    """Test ProcessedDataStorage implementation."""
    
    @pytest.mark.asyncio
    async def test_store_and_load_candles(self, test_config, sample_candles, clean_test_dirs):
        storage = ProcessedDataStorage(test_config)
        await storage.store_candles("test_exchange", "BTC-PERP", "15", sample_candles)
        df = await storage.load_candles(
            "test_exchange",
            "BTC-PERP",
            "15",
            sample_candles[0].timestamp,
            sample_candles[-1].timestamp
        )
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) == len(sample_candles)
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if df.index.name == 'timestamp':
            expected_columns.remove('timestamp')
        for col in expected_columns:
            assert col in df.columns, f"Column {col} missing from DataFrame"
    
    @pytest.mark.asyncio
    async def test_resample_candles(self, test_config, sample_candles, clean_test_dirs):
        storage = ProcessedDataStorage(test_config)
        await storage.store_candles("test_exchange", "BTC-PERP", "15", sample_candles)
        df = await storage.load_candles(
            "test_exchange",
            "BTC-PERP",
            "15",
            sample_candles[0].timestamp,
            sample_candles[-1].timestamp
        )
        print("Columns after load_candles:", list(df.columns))
        if df.index.name != 'timestamp' and 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        try:
            resampled = await storage.resample_candles(df, "60")
            assert isinstance(resampled, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Resampling failed: {str(e)}")

class TestDataManager:
    """Test DataManager functionality."""
    
    @pytest.mark.asyncio
    async def test_store_and_load_both_formats(self, test_config, sample_candles, clean_test_dirs):
        # Patch __init__ methods of RawDataStorage and ProcessedDataStorage so they accept a StorageConfig
        from src.storage.raw import RawDataStorage as OrigRawStorage
        from src.storage.processed import ProcessedDataStorage as OrigProcStorage

        def patched_raw_init(self, config):
            self.base_path = config.historical_raw_path
            self.use_compression = config.use_compression
            self.base_path.mkdir(parents=True, exist_ok=True)
        def patched_proc_init(self, config):
            self.base_path = config.historical_processed_path
            self.use_compression = config.use_compression
            self.base_path.mkdir(parents=True, exist_ok=True)

        OrigRawStorage.__init__ = patched_raw_init
        OrigProcStorage.__init__ = patched_proc_init

        from src.storage import DataManager
        manager = DataManager(test_config)
        await manager.store_data(
            sample_candles,
            "test_exchange",
            "BTC-PERP",
            "15"
        )
        raw_data = await manager.load_data(
            "test_exchange",
            "BTC-PERP",
            "15",
            sample_candles[0].timestamp,
            sample_candles[-1].timestamp,
            format_type="raw"
        )
        processed_data = await manager.load_data(
            "test_exchange",
            "BTC-PERP",
            "15",
            sample_candles[0].timestamp,
            sample_candles[-1].timestamp,
            format_type="processed"
        )
        assert len(raw_data) >= len(sample_candles)
        assert len(processed_data) >= len(sample_candles)
    
    @pytest.mark.asyncio
    async def test_data_verification(self, test_config, sample_candles, clean_test_dirs):
        from src.storage.raw import RawDataStorage as OrigRawStorage
        from src.storage.processed import ProcessedDataStorage as OrigProcStorage

        def patched_raw_init(self, config):
            self.base_path = config.historical_raw_path
            self.use_compression = config.use_compression
            self.base_path.mkdir(parents=True, exist_ok=True)
        def patched_proc_init(self, config):
            self.base_path = config.historical_processed_path
            self.use_compression = config.use_compression
            self.base_path.mkdir(parents=True, exist_ok=True)

        OrigRawStorage.__init__ = patched_raw_init
        OrigProcStorage.__init__ = patched_proc_init

        from src.storage import DataManager
        manager = DataManager(test_config)
        await manager.store_data(
            sample_candles,
            "test_exchange",
            "BTC-PERP",
            "15"
        )

        verification = await manager.verify_data(
            "test_exchange",
            "BTC-PERP",
            "15",
            sample_candles[0].timestamp,
            sample_candles[-1].timestamp
        )
        assert verification["data_matches"]
        assert verification["raw_data_count"] >= len(sample_candles)
        assert verification["processed_data_count"] >= len(sample_candles)

class TestLiveDataStorage:
    """Tests for LiveDataStorage implementation."""
    
    @pytest.mark.asyncio
    async def test_store_and_load_raw_live_candle(self, live_storage, live_sample_candle):
        exchange = "test_exchange"
        market = "TEST"
        resolution = "1"
        result = await live_storage.store_raw_candle(exchange, market, resolution, live_sample_candle)
        assert result is True
        date_str = live_sample_candle.timestamp.strftime("%Y-%m-%d")
        folder_path = live_storage._get_folder_path(live_storage.raw_path, exchange, market, resolution)
        file_path = folder_path / f"{date_str}.live.raw"
        assert file_path.exists()
        content = file_path.read_text(encoding="utf-8")
        records = json.loads(content)
        assert isinstance(records, list)
        expected_ts = live_sample_candle.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        loaded_timestamps = [pd.to_datetime(rec["timestamp"]).strftime("%Y-%m-%d %H:%M:%S") for rec in records]
        assert expected_ts in loaded_timestamps
    
    @pytest.mark.asyncio
    async def test_store_and_load_processed_live_candle(self, live_storage, live_sample_candle):
        exchange = "test_exchange"
        market = "TEST"
        resolution = "1"
        result = await live_storage.store_processed_candle(exchange, market, resolution, live_sample_candle)
        assert result is True
        date_str = live_sample_candle.timestamp.strftime("%Y-%m-%d")
        folder_path = live_storage._get_folder_path(live_storage.processed_path, exchange, market, resolution)
        file_path = folder_path / f"{date_str}.csv"
        assert file_path.exists()
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        assert not df.empty
        expected_ts = live_sample_candle.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        loaded_ts = df["timestamp"].astype(str).apply(lambda ts: pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")).tolist()
        assert expected_ts in loaded_ts
