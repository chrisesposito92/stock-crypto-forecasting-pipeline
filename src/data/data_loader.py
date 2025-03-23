import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataLoader:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_market_data(self, ticker, start_date, end_date=None, interval='1d', asset_type='stock'):
        """
        Download historical market data for a specific ticker.
        
        Args:
            ticker (str): Market ticker symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to today
            interval (str, optional): Data interval, defaults to '1d' (daily)
            asset_type (str, optional): Type of asset ('stock' or 'crypto')
            
        Returns:
            pd.DataFrame: DataFrame containing the market data
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        # Format crypto tickers properly if needed
        if asset_type == 'crypto' and '-' not in ticker:
            ticker = f"{ticker}-USD"
            
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
        
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return None
                
            # Create a clean ticker name for the file (remove any USD suffix for crypto)
            clean_ticker = ticker.split('-')[0] if '-' in ticker else ticker
                
            # Save data to CSV
            file_path = os.path.join(self.output_dir, f"{clean_ticker}_{start_date}_{end_date}.csv")
            data.to_csv(file_path)
            logger.info(f"Saved data to {file_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None
            
    def download_multiple_assets(self, tickers, start_date, end_date=None, interval='1d', asset_type='stock'):
        """
        Download data for multiple market tickers.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to today
            interval (str, optional): Data interval, defaults to '1d' (daily)
            asset_type (str, optional): Type of assets ('stock' or 'crypto')
            
        Returns:
            dict: Dictionary with tickers as keys and DataFrames as values
        """
        result = {}
        
        for ticker in tickers:
            # Get a clean ticker name (without USD suffix for crypto)
            clean_ticker = ticker.split('-')[0] if '-' in ticker else ticker
            
            data = self.download_market_data(ticker, start_date, end_date, interval, asset_type)
            if data is not None:
                result[clean_ticker] = data
                
        return result
    
    def load_market_data(self, ticker, start_date, end_date=None, asset_type='stock'):
        """
        Load existing market data from CSV file, or download if not available.
        
        Args:
            ticker (str): Market ticker symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to today
            asset_type (str, optional): Type of asset ('stock' or 'crypto')
            
        Returns:
            pd.DataFrame: DataFrame containing the market data
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        # Format crypto tickers properly if needed
        display_ticker = ticker
        if asset_type == 'crypto' and '-' not in ticker:
            display_ticker = f"{ticker}-USD"
            
        # Get a clean ticker name for the file (remove any USD suffix)
        clean_ticker = ticker.split('-')[0] if '-' in ticker else ticker
        file_path = os.path.join(self.output_dir, f"{clean_ticker}_{start_date}_{end_date}.csv")
        
        if os.path.exists(file_path):
            logger.info(f"Loading {display_ticker} data from {file_path}")
            try:
                # Try to load using the standard format
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Clean the data if needed
                # Check if any column is non-numeric (except datetime index)
                numeric_cols = [col for col in df.columns if col not in ['Date']]
                for col in numeric_cols:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        # Convert to numeric, forcing non-numeric values to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN
                df = df.dropna()
                
                return df
            except Exception as e:
                logger.warning(f"Error loading data from file: {str(e)}. Downloading fresh data.")
                return self.download_market_data(display_ticker, start_date, end_date, '1d', asset_type)
        else:
            logger.info(f"No local data found for {display_ticker}, downloading...")
            return self.download_market_data(display_ticker, start_date, end_date, '1d', asset_type)
            
    # Legacy methods for backward compatibility
    def download_stock_data(self, ticker, start_date, end_date=None, interval='1d'):
        """Legacy method for backward compatibility"""
        return self.download_market_data(ticker, start_date, end_date, interval, 'stock')
        
    def download_multiple_stocks(self, tickers, start_date, end_date=None, interval='1d'):
        """Legacy method for backward compatibility"""
        return self.download_multiple_assets(tickers, start_date, end_date, interval, 'stock')
        
    def load_stock_data(self, ticker, start_date, end_date=None):
        """Legacy method for backward compatibility"""
        return self.load_market_data(ticker, start_date, end_date, 'stock')