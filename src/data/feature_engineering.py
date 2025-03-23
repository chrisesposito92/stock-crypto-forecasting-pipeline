import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, input_dir='data/raw', output_dir='data/features'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def add_time_features(self, df):
        """
        Add date-based features to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with DatetimeIndex
            
        Returns:
            pd.DataFrame: DataFrame with additional time features
        """
        df_copy = df.copy()
        
        # Add day of week, month, year, etc.
        df_copy['day_of_week'] = df_copy.index.dayofweek
        df_copy['month'] = df_copy.index.month
        df_copy['year'] = df_copy.index.year
        df_copy['is_month_start'] = df_copy.index.is_month_start.astype(int)
        df_copy['is_month_end'] = df_copy.index.is_month_end.astype(int)
        
        return df_copy
        
    def add_technical_indicators(self, df):
        """
        Add technical indicators for stock market analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV stock data
            
        Returns:
            pd.DataFrame: DataFrame with additional technical indicators
        """
        df_copy = df.copy()
        
        # Simple Moving Averages
        df_copy['SMA_5'] = df_copy['Close'].rolling(window=5).mean()
        df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
        df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df_copy['EMA_12'] = df_copy['Close'].ewm(span=12, adjust=False).mean()
        df_copy['EMA_26'] = df_copy['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
        df_copy['MACD_signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df_copy['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_copy['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df_copy['BB_middle'] = df_copy['Close'].rolling(window=20).mean()
        df_copy['BB_std'] = df_copy['Close'].rolling(window=20).std()
        df_copy['BB_upper'] = df_copy['BB_middle'] + 2 * df_copy['BB_std']
        df_copy['BB_lower'] = df_copy['BB_middle'] - 2 * df_copy['BB_std']
        
        # Daily Returns
        df_copy['daily_return'] = df_copy['Close'].pct_change()
        
        # Volatility (standard deviation of returns)
        df_copy['volatility_5d'] = df_copy['daily_return'].rolling(window=5).std()
        df_copy['volatility_20d'] = df_copy['daily_return'].rolling(window=20).std()
        
        # Price momentum
        df_copy['momentum_5d'] = df_copy['Close'] / df_copy['Close'].shift(5) - 1
        df_copy['momentum_20d'] = df_copy['Close'] / df_copy['Close'].shift(20) - 1
        
        return df_copy
        
    def add_target_variables(self, df, forecast_horizon=5):
        """
        Add target variables for prediction.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            forecast_horizon (int): Number of days to forecast ahead
            
        Returns:
            pd.DataFrame: DataFrame with target variables
        """
        df_copy = df.copy()
        
        # Future price
        df_copy[f'future_price_{forecast_horizon}d'] = df_copy['Close'].shift(-forecast_horizon)
        
        # Future return
        df_copy[f'future_return_{forecast_horizon}d'] = (df_copy[f'future_price_{forecast_horizon}d'] / df_copy['Close']) - 1
        
        # Binary target: 1 if price goes up, 0 if down
        df_copy[f'price_up_{forecast_horizon}d'] = (df_copy[f'future_return_{forecast_horizon}d'] > 0).astype(int)
        
        return df_copy
        
    def process_stock_data(self, ticker, add_time=True, add_technicals=True, add_targets=True, forecast_horizon=5):
        """
        Process stock data by adding engineered features.
        
        Args:
            ticker (str): Stock ticker symbol
            add_time (bool): Whether to add time-based features
            add_technicals (bool): Whether to add technical indicators
            add_targets (bool): Whether to add target variables
            forecast_horizon (int): Number of days to forecast ahead
            
        Returns:
            pd.DataFrame: Processed DataFrame with engineered features
        """
        logger.info(f"Processing features for {ticker}")
        
        # Find the most recent file for this ticker
        files = [f for f in os.listdir(self.input_dir) if f.startswith(ticker + '_')]
        if not files:
            logger.warning(f"No data files found for {ticker}")
            return None
        
        latest_file = sorted(files)[-1]
        file_path = os.path.join(self.input_dir, latest_file)
        
        try:
            # Read the CSV file
            raw_df = pd.read_csv(file_path)
            
            # Clean the data - remove the first rows with 'Price', 'Ticker', etc.
            # Convert the Date column to datetime and set as index
            cleaned_df = raw_df.copy()
            
            # Find where the actual data begins (look for rows with dates)
            for i, val in enumerate(raw_df.iloc[:, 0]):
                try:
                    pd.to_datetime(val)
                    # If we reach here, the value could be parsed as a date
                    start_row = i
                    break
                except:
                    continue
            
            # Extract only the rows with actual data
            cleaned_df = raw_df.iloc[start_row:].copy()
            
            # Rename the first column to 'Date'
            col_names = cleaned_df.columns.tolist()
            col_names[0] = 'Date'
            cleaned_df.columns = col_names
            
            # Convert Date to datetime and set as index
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
            df = cleaned_df.set_index('Date')
            
            # Convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            # Apply feature engineering
            if add_time:
                df = self.add_time_features(df)
            
            if add_technicals:
                df = self.add_technical_indicators(df)
            
            if add_targets:
                df = self.add_target_variables(df, forecast_horizon)
            
            # Save processed data
            output_file = os.path.join(self.output_dir, f"{ticker}_processed.csv")
            df.to_csv(output_file)
            logger.info(f"Saved processed data to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {str(e)}")
            return None