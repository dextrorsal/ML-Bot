"""
Wrapper for logging setup to provide compatibility with different logging configurations.
"""

import logging
from ..utils.logging import setup_logging as original_setup_logging

def enhanced_setup_logging(log_level=logging.INFO):
    """
    Enhanced setup_logging function that accepts a log_level parameter.
    
    Args:
        log_level: Logging level (default: logging.INFO)
    """
    # Call the original setup_logging
    original_setup_logging()
    
    # Set the log level for the root logger
    logging.getLogger().setLevel(log_level)
    
    # Also set for our main loggers
    for logger_name in ['src', 'ultimate_data_fetcher']:
        logging.getLogger(logger_name).setLevel(log_level)