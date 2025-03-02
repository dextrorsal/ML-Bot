"""
Exchange handlers package.
Provides a unified interface to different cryptocurrency exchanges.
"""

from typing import Dict, Type, Optional

from .base import BaseExchangeHandler
from .drift import DriftHandler
from .binance import BinanceHandler
from .coinbase import CoinbaseHandler
from ..core.config import ExchangeConfig
from ..core.exceptions import ExchangeError

# Export handlers directly
__all__ = ['DriftHandler', 'BinanceHandler', 'CoinbaseHandler', 'get_exchange_handler']

# Registry of available exchange handlers
EXCHANGE_HANDLERS: Dict[str, Type[BaseExchangeHandler]] = {
    'drift': DriftHandler,
    'binance': BinanceHandler,
    'coinbase': CoinbaseHandler
}

def get_exchange_handler(config: ExchangeConfig) -> BaseExchangeHandler:
    """
    Factory function to get the appropriate exchange handler.
    
    Args:
        config: Exchange configuration including name and credentials
        
    Returns:
        Initialized exchange handler
        
    Raises:
        ExchangeError: If exchange is not supported
    """
    handler_class = EXCHANGE_HANDLERS.get(config.name.lower())
    if not handler_class:
        raise ExchangeError(f"Unsupported exchange: {config.name}")
        
    return handler_class(config)

# Convenience function to get all supported exchanges
def get_supported_exchanges() -> list:
    """Get list of supported exchange names."""
    return list(EXCHANGE_HANDLERS.keys())