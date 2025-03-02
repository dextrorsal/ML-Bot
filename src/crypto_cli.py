"""
CLI Entry point for the Crypto Data Fetcher
"""

import asyncio
import sys

def cli_entry():
    """Entry point function for console script."""
    # On Windows, use a different event loop policy to avoid issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main async function
    asyncio.run(main())

# Import the actual main function here
from .utils.improved_cli import main

if __name__ == "__main__":
    cli_entry()