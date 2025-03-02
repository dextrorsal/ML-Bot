#!/usr/bin/env python3
"""
Crypto Data Fetcher - Launcher Script
Run this script to fetch data from cryptocurrency exchanges.

Example usage:
    # Show available exchanges and markets
    python fetch.py list
    
    # Fetch historical data for the last 7 days
    python fetch.py historical --days 7 --markets BTC-PERP ETH-PERP SOL-PERP --resolution 1D
    
    # Fetch historical data with specific date range
    python fetch.py historical --start-date "2023-01-01" --end-date "2023-01-31" --markets BTC-PERP --exchanges binance
    
    # Start live data fetching
    python fetch.py live --markets BTC-PERP ETH-PERP --resolution 15 --interval 30
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

# Import the main function from your package
from src.utils.improved_cli import main

if __name__ == "__main__":
    try:
        # On Windows, use a different event loop policy to avoid issues
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)