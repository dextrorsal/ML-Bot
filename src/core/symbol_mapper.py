"""
Symbol standardization for cryptocurrency exchanges.
Maps between a standard internal symbol format and exchange-specific formats.
"""

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class SymbolMapper:
    """
    Manages symbol mapping between a standardized internal format and exchange-specific formats.
    
    The internal standard format is: BASE-QUOTE
    Examples: BTC-USD, ETH-USDT, SOL-USDC, BTC-PERP
    
    For perpetual contracts, we use BASE-PERP as the standard format.
    """
    
    # Standard asset naming (handles aliases like XBT -> BTC)
    ASSET_ALIASES = {
        "XBT": "BTC",
        "DOGE": "DOGE",
        "SHIB": "SHIB",
        "PEPE": "PEPE"
    }
    
    # Common stablecoins to handle quote currencies
    STABLECOINS = ['USD', 'USDT', 'USDC', 'BUSD', 'DAI']
    
    # Common perpetual contract suffixes in different exchanges
    PERPETUAL_IDENTIFIERS = ['PERP', 'PERPETUAL', '-P', '-PERP', 'SWAP']
    
    def __init__(self):
        """Initialize the symbol mapper with empty mapping tables."""
        # Maps exchange:standard_symbol → exchange_symbol
        self.to_exchange_map: Dict[str, Dict[str, str]] = {}
        
        # Maps exchange:exchange_symbol → standard_symbol
        self.from_exchange_map: Dict[str, Dict[str, str]] = {}
        
        # Sets of supported symbols for each exchange
        self.supported_symbols: Dict[str, Set[str]] = {}
    
    def register_exchange(self, exchange_name: str, symbols: List[str]) -> None:
        """
        Register an exchange and its supported symbols.
        
        Args:
            exchange_name: Name of the exchange
            symbols: List of symbols in exchange-specific format
        """
        exchange_id = exchange_name.lower()
        
        if exchange_id not in self.to_exchange_map:
            self.to_exchange_map[exchange_id] = {}
            self.from_exchange_map[exchange_id] = {}
            self.supported_symbols[exchange_id] = set()
        
        for symbol in symbols:
            try:
                standard_symbol = self._exchange_to_standard(exchange_id, symbol)
                
                # Register the mapping in both directions
                self.to_exchange_map[exchange_id][standard_symbol] = symbol
                self.from_exchange_map[exchange_id][symbol] = standard_symbol
                self.supported_symbols[exchange_id].add(standard_symbol)
                
                logger.debug(f"Registered symbol {symbol} → {standard_symbol} for {exchange_name}")
            except ValueError as e:
                logger.warning(f"Skipping invalid symbol {symbol} for {exchange_name}: {str(e)}")
    
    def to_exchange_symbol(self, exchange_name: str, standard_symbol: str) -> str:
        """
        Convert a standard symbol to exchange-specific format.
        
        Args:
            exchange_name: Name of the exchange
            standard_symbol: Symbol in standard format (e.g., BTC-USD) or 
                            exchange-specific format (e.g., BTCUSDT)
            
        Returns:
            Symbol in exchange-specific format
            
        Raises:
            ValueError: If the symbol is not supported by the exchange
        """
        exchange_id = exchange_name.lower()

        # First check if the symbol is already in an exchange-specific format we recognize
        if exchange_id == 'binance' and re.match(r'^[A-Z]+(USDT|USDC|BTC|ETH)(_PERP)?$', standard_symbol):
            # Already in valid Binance format (e.g. BTCUSDT, ETHBTC, SOLUSDT_PERP)
            return standard_symbol
        elif exchange_id == 'coinbase' and '-' in standard_symbol:
            # If symbol has a dash and we're converting for Coinbase, it might already be valid
            return standard_symbol.upper()
        elif exchange_id == 'drift' and '-PERP' in standard_symbol:
            # If symbol has -PERP and we're converting for Drift, it might already be valid
            return standard_symbol.upper()

        # If the symbol is provided as a bare base asset (like "SOL")
        if '-' not in standard_symbol:
            if exchange_id == 'coinbase':
                standard_symbol = f"{standard_symbol}-USD"
            elif exchange_id == 'binance':
                standard_symbol = f"{standard_symbol}-USDT"
            elif exchange_id == 'drift':
                standard_symbol = f"{standard_symbol}-PERP"
        
        # Check if we have a direct mapping
        if exchange_id in self.to_exchange_map and standard_symbol in self.to_exchange_map[exchange_id]:
            return self.to_exchange_map[exchange_id][standard_symbol]
        
        # Try to generate the exchange symbol format
        try:
            if exchange_id == "binance":
                return self._standard_to_binance(standard_symbol)
            elif exchange_id == "coinbase":
                return self._standard_to_coinbase(standard_symbol)
            elif exchange_id == "drift":
                return self._standard_to_drift(standard_symbol)
            else:
                # For unknown exchanges, use a default conversion
                return standard_symbol.replace("-", "")
        except Exception as e:
            raise ValueError(f"Cannot convert {standard_symbol} to {exchange_name} format: {str(e)}")
    
    def from_exchange_symbol(self, exchange_name: str, exchange_symbol: str) -> str:
        """
        Convert an exchange-specific symbol to standard format.
        
        Args:
            exchange_name: Name of the exchange
            exchange_symbol: Symbol in exchange-specific format
            
        Returns:
            Symbol in standard format (e.g., BTC-USD)
            
        Raises:
            ValueError: If the symbol cannot be converted
        """
        exchange_id = exchange_name.lower()
        
        # First check if the symbol is already in standard format (contains a dash)
        if '-' in exchange_symbol and not exchange_symbol.endswith('_PERP'):
            # Validate that it has the correct BASE-QUOTE format
            parts = exchange_symbol.split('-')
            if len(parts) == 2:
                # Looks like standard format, return it directly
                return exchange_symbol.upper()
        
        # Check if we have a direct mapping
        if exchange_id in self.from_exchange_map and exchange_symbol in self.from_exchange_map[exchange_id]:
            return self.from_exchange_map[exchange_id][exchange_symbol]
        
        # Try to generate the standard symbol format
        try:
            if exchange_id == "binance":
                return self._binance_to_standard(exchange_symbol)
            elif exchange_id == "coinbase":
                return self._coinbase_to_standard(exchange_symbol)
            elif exchange_id == "drift":
                return self._drift_to_standard(exchange_symbol)
            else:
                # For unknown exchanges, try to detect base/quote with common patterns
                return self._generic_to_standard(exchange_symbol)
        except Exception as e:
            raise ValueError(f"Cannot convert {exchange_symbol} from {exchange_name} format: {str(e)}")
    
    def get_supported_exchanges(self, standard_symbol: str) -> List[str]:
        """
        Get a list of exchanges that support the given standard symbol.
        
        Args:
            standard_symbol: Symbol in standard format
            
        Returns:
            List of exchange names that support this symbol
        """
        supported = []
        for exchange_id, symbols in self.supported_symbols.items():
            if standard_symbol in symbols:
                supported.append(exchange_id)
        return supported
    
    def get_supported_symbols(self, exchange_name: str) -> List[str]:
        """
        Get a list of standard symbols supported by the given exchange.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            List of supported symbols in standard format
        """
        exchange_id = exchange_name.lower()
        if exchange_id in self.supported_symbols:
            return list(self.supported_symbols[exchange_id])
        return []
    
    def standardize_asset(self, asset: str) -> str:
        """
        Standardize an asset name, handling aliases.
        
        Args:
            asset: Asset name or symbol
            
        Returns:
            Standardized asset name
        """
        asset = asset.upper()
        return self.ASSET_ALIASES.get(asset, asset)
    
    # -------------------------------------------------------------------------
    # Private methods for exchange-specific conversions
    # -------------------------------------------------------------------------
    
    def _exchange_to_standard(self, exchange_id: str, symbol: str) -> str:
        """Convert an exchange symbol to standard format based on exchange type."""
        if exchange_id == "binance":
            return self._binance_to_standard(symbol)
        elif exchange_id == "coinbase":
            return self._coinbase_to_standard(symbol)
        elif exchange_id == "drift":
            return self._drift_to_standard(symbol)
        else:
            return self._generic_to_standard(symbol)
    
    def _binance_to_standard(self, symbol: str) -> str:
        """Convert Binance symbol format to standard format."""
        # Check if already in standard format
        if '-' in symbol:
            parts = symbol.split('-')
            if len(parts) == 2:
                return symbol.upper()
                
        # Handle perpetual futures (e.g., BTCUSDT_PERP)
        if "_PERP" in symbol:
            base = re.search(r'([A-Z]+)USDT_PERP', symbol)
            if base:
                return f"{base.group(1)}-PERP"
        
        # Handle spot markets
        for stablecoin in self.STABLECOINS:
            if symbol.endswith(stablecoin):
                base = symbol[:-len(stablecoin)]
                return f"{base}-{stablecoin}"
        
        # Handle other pairs (e.g., BTCETH)
        for quote_length in [3, 4, 5]:
            if len(symbol) > quote_length:
                base = symbol[:-quote_length]
                quote = symbol[-quote_length:]
                return f"{base}-{quote}"
        
        raise ValueError(f"Cannot parse Binance symbol: {symbol}")
    
    def _coinbase_to_standard(self, symbol: str) -> str:
        """Convert Coinbase symbol format to standard format."""
        # Coinbase typically uses BASE-QUOTE format already
        if "-" in symbol:
            parts = symbol.split("-")
            if len(parts) == 2:
                base, quote = parts
                return f"{base}-{quote}"
        
        raise ValueError(f"Cannot parse Coinbase symbol: {symbol}")
    
    def _drift_to_standard(self, symbol: str) -> str:
        """Convert Drift symbol format to standard format."""
        # Drift typically uses BASE-PERP for perpetuals
        if symbol.endswith("-PERP"):
            return symbol
        
        # Handle spot pairs if any
        if "-" in symbol:
            parts = symbol.split("-")
            if len(parts) == 2:
                base, quote = parts
                return f"{base}-{quote}"
        
        raise ValueError(f"Cannot parse Drift symbol: {symbol}")
    
    def _generic_to_standard(self, symbol: str) -> str:
        """
        Try to convert a generic exchange symbol to standard format.
        Uses common patterns to identify base and quote assets.
        """
        # Check if it's already in standard format
        if "-" in symbol:
            parts = symbol.split("-")
            if len(parts) == 2:
                return symbol.upper()
        
        # Check for perpetual identifiers
        for perp_id in self.PERPETUAL_IDENTIFIERS:
            if perp_id in symbol:
                # Extract base currency
                base = re.sub(f"{perp_id}.*", "", symbol)
                # Clean up any separators
                base = re.sub(r'[^A-Z]', '', base.upper())
                return f"{base}-PERP"
        
        # Try to identify stablecoin quote currencies
        for stablecoin in self.STABLECOINS:
            if symbol.endswith(stablecoin):
                base = symbol[:-len(stablecoin)]
                return f"{base}-{stablecoin}"
        
        # Fallback: try to split at common pattern boundaries
        match = re.match(r'([A-Z]+)([A-Z]{3,5})$', symbol.upper())
        if match:
            base, quote = match.groups()
            return f"{base}-{quote}"
        
        raise ValueError(f"Cannot determine base/quote for symbol: {symbol}")
    
    def _standard_to_binance(self, standard_symbol: str) -> str:
        """Convert standard symbol format to Binance format."""
        if standard_symbol.endswith("-PERP"):
            base = standard_symbol.split("-")[0]
            return f"{base}USDT_PERP"
        
        parts = standard_symbol.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid standard symbol: {standard_symbol}")
        
        base, quote = parts
        return f"{base}{quote}"
    
    def _standard_to_coinbase(self, standard_symbol: str) -> str:
        """Convert standard symbol format to Coinbase format."""
        if standard_symbol.endswith("-PERP"):
            base = standard_symbol.split("-")[0]
            return f"{base}-USD"
        
        # Coinbase already uses similar format, just ensure it's uppercase
        return standard_symbol.upper()
    
    def _standard_to_drift(self, standard_symbol: str) -> str:
        """Convert standard symbol format to Drift format."""
        # Drift typically uses BASE-PERP for perpetuals
        return standard_symbol.upper()


# Example usage:
if __name__ == "__main__":
    # Create a symbol mapper
    mapper = SymbolMapper()
    
    # Register some exchanges with their symbols
    mapper.register_exchange("binance", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCUSDT_PERP"])
    mapper.register_exchange("coinbase", ["BTC-USD", "ETH-USD", "SOL-USD"])
    mapper.register_exchange("drift", ["SOL-PERP", "BTC-PERP", "ETH-PERP"])
    
    # Test conversions with standard symbols
    print("Testing conversions with standard symbols:")
    standard_symbols = ["BTC-USD", "ETH-USDT", "SOL-USDC", "BTC-PERP"]
    for standard_symbol in standard_symbols:
        print(f"\nTesting standard symbol: {standard_symbol}")
        for exchange in ["binance", "coinbase", "drift"]:
            try:
                exchange_symbol = mapper.to_exchange_symbol(exchange, standard_symbol)
                print(f"  {standard_symbol} → {exchange}: {exchange_symbol}")
                
                back_to_standard = mapper.from_exchange_symbol(exchange, exchange_symbol)
                print(f"  {exchange}: {exchange_symbol} → {back_to_standard}")
            except ValueError as e:
                print(f"  Error for {exchange}: {str(e)}")
    
    # Test with symbols that are already in exchange format
    print("\nTesting with symbols already in exchange format:")
    exchange_formats = ["BTCUSDT", "ETH-USD", "SOL-PERP"]
    for symbol in exchange_formats:
        print(f"\nTesting symbol: {symbol}")
        for exchange in ["binance", "coinbase", "drift"]:
            try:
                exchange_symbol = mapper.to_exchange_symbol(exchange, symbol)
                print(f"  {symbol} → {exchange}: {exchange_symbol}")
            except ValueError as e:
                print(f"  Error for {exchange}: {str(e)}")
    
    # Test with bare base assets
    print("\nTesting with bare base assets:")
    bare_assets = ["BTC", "ETH", "SOL"]
    for asset in bare_assets:
        print(f"\nTesting asset: {asset}")
        for exchange in ["binance", "coinbase", "drift"]:
            try:
                exchange_symbol = mapper.to_exchange_symbol(exchange, asset)
                print(f"  {asset} → {exchange}: {exchange_symbol}")
            except ValueError as e:
                print(f"  Error for {exchange}: {str(e)}")