"""
Storage module initialization.
Provides unified access to raw and processed data storage capabilities.
"""

from typing import Dict, Type

from .raw import RawDataStorage
from .processed import ProcessedDataStorage
from ..core.config import StorageConfig
import logging
logger = logging.getLogger(__name__)

# Export main storage classes
__all__ = [
    'RawDataStorage', 
    'ProcessedDataStorage', 
    'get_storage',
    'StorageType'
]

# Define storage types
class StorageType:
    RAW = "raw"
    PROCESSED = "processed"

# Storage registry
STORAGE_HANDLERS: Dict[str, Type] = {
    StorageType.RAW: RawDataStorage,
    StorageType.PROCESSED: ProcessedDataStorage
}

def get_storage(storage_type: str, config: StorageConfig):
    """
    Factory function to get appropriate storage handler.
    
    Args:
        storage_type: Type of storage ("raw" or "processed")
        config: Storage configuration
        
    Returns:
        Initialized storage handler
        
    Raises:
        ValueError: If storage type is not supported
    """
    storage_class = STORAGE_HANDLERS.get(storage_type.lower())
    if not storage_class:
        raise ValueError(f"Unsupported storage type: {storage_type}")
        
    return storage_class(config)

class DataManager:
    """Manages both raw and processed data storage."""
    
    def __init__(self, config: StorageConfig):
        """Initialize data manager with configuration."""
        self.config = config
        self.raw_storage = RawDataStorage(config)
        self.processed_storage = ProcessedDataStorage(config)

    async def store_data(self, data, exchange: str, market: str, resolution: str):
        """Store data in both raw and processed formats."""
        # Store raw data
        await self.raw_storage.store_candles(
            exchange=exchange,
            market=market,
            resolution=resolution,
            candles=data
        )
        
        # Store processed data
        await self.processed_storage.store_candles(
            exchange=exchange,
            market=market,
            resolution=resolution,
            candles=data
        )

    async def load_data(
        self,
        exchange: str,
        market: str,
        resolution: str,
        start_time,
        end_time,
        format_type: str = StorageType.PROCESSED
    ):
        """
        Load data from storage.
        
        Args:
            exchange: Exchange name
            market: Market symbol
            resolution: Candle resolution
            start_time: Start timestamp
            end_time: End timestamp
            format_type: Type of data to load ("raw" or "processed")
            
        Returns:
            Loaded data in specified format
        """
        if format_type == StorageType.RAW:
            return await self.raw_storage.load_candles(
                exchange, market, resolution, start_time, end_time
            )
        else:
            return await self.processed_storage.load_candles(
                exchange, market, resolution, start_time, end_time
            )

    async def verify_data(
        self,
        exchange: str,
        market: str,
        resolution: str,
        start_time,
        end_time
    ) -> Dict:
        """Verify data integrity in both storage types."""
        # Get verification results from processed storage
        processed_results = await self.processed_storage.verify_data_integrity(
            exchange, market, resolution, start_time, end_time
        )
        
        # Load raw data for comparison
        raw_data = await self.raw_storage.load_candles(
            exchange, market, resolution, start_time, end_time
        )
        
        # Compare data counts
        raw_count = len(raw_data) if raw_data else 0
        processed_count = processed_results.get('total_candles', 0)
        
        return {
            "raw_data_count": raw_count,
            "processed_data_count": processed_count,
            "data_matches": raw_count == processed_count,
            "processed_verification": processed_results
        }

    async def backup_all_data(self, backup_path):
        """Create backup of both raw and processed data."""
        try:
            await self.raw_storage.backup_data(backup_path / "raw")
            await self.processed_storage.backup_data(backup_path / "processed")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False