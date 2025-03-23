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

class StockDataLoader:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_stock_data(self, ticker, start_date, end_date=None, interval='1d'):
        """
        Download historical stock data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to today
            interval (str, optional): Data interval, defaults to '1d' (daily)
            
        Returns:
            pd.DataFrame: DataFrame containing the stock data
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
        
        try:
            stock_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if stock_data.empty:
                logger.warning(f"No data found for {ticker}")
                return None
                
            # Save data to CSV
            file_path = os.path.join(self.output_dir, f"{ticker}_{start_date}_{end_date}.csv")
            stock_data.to_csv(file_path)
            logger.info(f"Saved data to {file_path}")
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None
            
    def download_multiple_stocks(self, tickers, start_date, end_date=None, interval='1d'):
        """
        Download data for multiple stock tickers.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to today
            interval (str, optional): Data interval, defaults to '1d' (daily)
            
        Returns:
            dict: Dictionary with tickers as keys and DataFrames as values
        """
        result = {}
        
        for ticker in tickers:
            data = self.download_stock_data(ticker, start_date, end_date, interval)
            if data is not None:
                result[ticker] = data
                
        return result
    
    def load_stock_data(self, ticker, start_date, end_date=None):
        """
        Load existing stock data from CSV file, or download if not available.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to today
            
        Returns:
            pd.DataFrame: DataFrame containing the stock data
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        file_path = os.path.join(self.output_dir, f"{ticker}_{start_date}_{end_date}.csv")
        
        if os.path.exists(file_path):
            logger.info(f"Loading {ticker} data from {file_path}")
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
                return self.download_stock_data(ticker, start_date, end_date)
        else:
            logger.info(f"No local data found for {ticker}, downloading...")
            return self.download_stock_data(ticker, start_date, end_date)