# src/tests/conftest.py
import pytest
import pytest_asyncio
import os
import sys
import asyncio

# Configure pytest-asyncio
pytest_asyncio_config = {
    "asyncio_mode": "strict",
    "asyncio_default_fixture_loop_scope": "function",
}

def pytest_configure(config):
    """Configure matplotlib for non-interactive testing and register custom marks."""
    import matplotlib.pyplot as plt
    # plt.ioff()  # Turn off interactive mode if desired
    config.addinivalue_line("markers", "timeout: mark test with a timeout value in seconds")
    config.addinivalue_line("markers", "real_data: mark test that uses real exchange data")
    config.addinivalue_line("markers", "simple: mark test as 'simple'")

@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')

def pytest_collection_modifyitems(config, items):
    """
    Skip real_data tests by default, so you don't accidentally hit live endpoints.
    Uncomment to enable skipping by default.
    """
    # skip_real_data = pytest.mark.skip(reason="real_data tests are disabled by default")
    # for item in items:
    #     if "real_data" in item.keywords:
    #         item.add_marker(skip_real_data)

@pytest_asyncio.fixture
async def mock_processed_storage(tmp_path_factory):
    """
    Mock ProcessedDataStorage for normal tests (not real data).
    """
    from src.storage.processed import ProcessedDataStorage
    from src.core.config import StorageConfig
    import pandas as pd
    from unittest.mock import patch

    temp_dir = tmp_path_factory.mktemp("processed_data")

    config = StorageConfig(
        data_path=temp_dir,
        historical_raw_path=temp_dir / "historical" / "raw",
        historical_processed_path=temp_dir / "historical" / "processed",
        live_raw_path=temp_dir / "live" / "raw",
        live_processed_path=temp_dir / "live" / "processed",
        use_compression=False
    )

    storage = ProcessedDataStorage(config)

    # Always return an empty DataFrame for load_candles in normal tests
    with patch.object(storage, 'load_candles', return_value=pd.DataFrame()):
        yield storage

@pytest.fixture(autouse=True)
def patch_real_data_backtest(monkeypatch, tmp_path_factory, request):
    """
    Patch ProcessedDataStorage for tests EXCEPT those marked with @pytest.mark.real_data.
    So real_data tests will use the actual data/historical/... directories.
    """
    # If the test has the "real_data" mark, do NOT patch it (i.e., use real folders).
    if "real_data" in request.keywords:
        return

    from pathlib import Path
    from src.core.config import StorageConfig
    from src.storage.processed import ProcessedDataStorage

    temp_dir = tmp_path_factory.mktemp("real_data_test")

    def mock_init(*args, **kwargs):
        storage_config = StorageConfig(
            data_path=temp_dir,
            historical_raw_path=temp_dir / "historical" / "raw",
            historical_processed_path=temp_dir / "historical" / "processed",
            live_raw_path=temp_dir / "live" / "raw",
            live_processed_path=temp_dir / "live" / "processed",
            use_compression=False
        )
        return ProcessedDataStorage(storage_config)

    try:
        import src.tests.test_real_data_backtest
        # Replace references to ProcessedDataStorage with mock_init
        monkeypatch.setattr(
            'src.tests.test_real_data_backtest.ProcessedDataStorage',
            mock_init
        )
    except ImportError:
        pass

# -------------------------------------------------------------------------
# Optional: Example fixtures for testing new exchange handlers
# -------------------------------------------------------------------------
@pytest_asyncio.fixture
async def mock_coinbase_handler():
    """
    Example fixture for testing the updated CoinbaseHandler.
    This fixture starts the handler and stops it after tests finish.
    """
    from src.exchanges.coinbase import CoinbaseHandler
    from src.core.config import Config
    
    # Create a minimal config or a mock config
    config = Config()
    
    handler = CoinbaseHandler(config)
    await handler.start()
    yield handler
    await handler.stop()

@pytest_asyncio.fixture
async def mock_drift_handler():
    """
    Example fixture for testing the updated DriftHandler.
    """
    from src.exchanges.drift import DriftHandler
    from src.core.config import Config
    
    config = Config()
    
    handler = DriftHandler(config)
    await handler.start()
    yield handler
    await handler.stop()
