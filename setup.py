from setuptools import setup, find_packages

setup(
    name="crypto-data-fetcher",
    version="0.1.0",
    description="Ultimate Crypto Data Fetcher - Tool for fetching and storing cryptocurrency market data",
    author="Crypto Trader",
    packages=find_packages(),
    install_requires=[
        "aiohttp",
        "asyncio",
        "pandas",
        "python-dotenv",
        "aiofiles",
    ],
    # Commenting out entry_points until we resolve module structure
    # entry_points={
    #     "console_scripts": [
    #         "crypto-fetch=src.crypto_cli:cli_entry",
    #     ],
    # },
    python_requires=">=3.8",
)