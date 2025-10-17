"""
Data loading utilities for financial data.
Handles loading raw data from various sources (APIs, files, databases).
"""

import pandas as pd
import yfinance as yf
import requests
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """Load financial data from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_price_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load price data for given symbols using yfinance.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, 5m, etc.)
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        price_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not data.empty:
                    price_data[symbol] = data
                    logger.info(f"Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        return price_data
    
    def load_news_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        api_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load news data for given symbols.
        Note: This is a placeholder - implement with actual news API.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            api_key: API key for news service
        
        Returns:
            DataFrame with news data
        """
        # Placeholder implementation
        logger.warning("News data loading not implemented - using placeholder")
        
        # Create sample news data
        news_data = []
        for symbol in symbols:
            for i in range(10):  # Sample 10 news items per symbol
                news_data.append({
                    'symbol': symbol,
                    'date': datetime.now() - timedelta(days=i),
                    'title': f"Sample news for {symbol} #{i+1}",
                    'content': f"This is sample news content for {symbol}.",
                    'source': 'sample_source',
                    'sentiment': 'neutral'
                })
        
        return pd.DataFrame(news_data)
    
    def load_fundamental_data(
        self, 
        symbols: List[str]
    ) -> Dict[str, Dict]:
        """
        Load fundamental data for given symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary mapping symbols to fundamental data
        """
        fundamental_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract key fundamental metrics
                fundamental_data[symbol] = {
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'roe': info.get('returnOnEquity'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry')
                }
                
                logger.info(f"Loaded fundamental data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading fundamental data for {symbol}: {e}")
        
        return fundamental_data
    
    def save_data(self, data: Union[pd.DataFrame, Dict], filename: str) -> None:
        """
        Save data to file.
        
        Args:
            data: Data to save
            filename: Filename (without extension)
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{filepath}.csv", index=True)
        elif isinstance(data, dict):
            import json
            with open(f"{filepath}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved data to {filepath}")
    
    def load_saved_data(self, filename: str) -> Union[pd.DataFrame, Dict]:
        """
        Load previously saved data.
        
        Args:
            filename: Filename (without extension)
        
        Returns:
            Loaded data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(f"{filepath}.csv"):
            return pd.read_csv(f"{filepath}.csv", index_col=0)
        elif os.path.exists(f"{filepath}.json"):
            import json
            with open(f"{filepath}.json", 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No saved data found for {filename}")


if __name__ == "__main__":
    # Example usage
    loader = FinancialDataLoader()
    
    # Load sample data
    symbols = ["AAPL", "GOOGL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Load price data
    price_data = loader.load_price_data(symbols, start_date, end_date)
    
    # Load fundamental data
    fundamental_data = loader.load_fundamental_data(symbols)
    
    # Save data
    for symbol, data in price_data.items():
        loader.save_data(data, f"price_{symbol}")
    
    loader.save_data(fundamental_data, "fundamental_data")