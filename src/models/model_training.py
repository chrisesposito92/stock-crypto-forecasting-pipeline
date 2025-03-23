import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_dir='data/features', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
    def prepare_data(self, df, target_col, feature_cols=None, test_size=0.2, time_series_split=True):
        """
        Prepare data for model training.
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            target_col (str): Name of the target column
            feature_cols (list, optional): List of feature columns to use
            test_size (float): Proportion of data to use for testing
            time_series_split (bool): Whether to use time series split
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, scaler
        """
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features if provided, otherwise use all numeric columns except targets
        if feature_cols is None:
            # Exclude target columns and non-numeric columns
            feature_cols = [col for col in df.columns if not col.startswith('future_') 
                           and not col.startswith('price_up_')
                           and df[col].dtype in ['int64', 'float64']]
        
        # Create feature matrix and target vector
        X = df[feature_cols]
        y = df[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        if time_series_split:
            # For time series data, we use a chronological split
            split_idx = int(len(df) * (1 - test_size))
            X_train = X_scaled[:split_idx]
            X_test = X_scaled[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
        else:
            # Random split for non-time-series applications
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
        
        return X_train, X_test, y_train, y_test, scaler, feature_cols
    
    def train_model(self, X_train, y_train, model_type='random_forest', **kwargs):
        """
        Train a model on the prepared data.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            model_type (str): Type of model to train
            **kwargs: Additional parameters for the model
            
        Returns:
            object: Trained model
        """
        logger.info(f"Training {model_type} model")
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=42
            )
        elif model_type == 'linear':
            model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a trained model.
        
        Args:
            model (object): Trained model
            X_test (np.array): Test features
            y_test (np.array): Test target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate baseline (using mean of training data)
        baseline_pred = np.full_like(y_test, y_test.mean())
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        baseline_rmse = np.sqrt(baseline_mse)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'baseline_rmse': baseline_rmse,
            'improvement': (baseline_rmse - rmse) / baseline_rmse * 100  # % improvement over baseline
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, model, scaler, feature_cols, ticker, target_col, model_type):
        """
        Save the trained model and metadata.
        
        Args:
            model (object): Trained model
            scaler (object): Feature scaler
            feature_cols (list): Feature column names
            ticker (str): Stock ticker symbol
            target_col (str): Target column name
            model_type (str): Type of model
            
        Returns:
            str: Path to the saved model
        """
        # Create a model package with all necessary components for prediction
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'ticker': ticker,
            'target_col': target_col,
            'model_type': model_type,
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save the model package
        model_file = os.path.join(self.models_dir, f"{ticker}_{target_col}_{model_type}_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Model saved to {model_file}")
        return model_file
    
    def load_model(self, model_file):
        """
        Load a trained model and metadata.
        
        Args:
            model_file (str): Path to the model file
            
        Returns:
            dict: Model package
        """
        with open(model_file, 'rb') as f:
            model_package = pickle.load(f)
        
        logger.info(f"Loaded model from {model_file}")
        return model_package
    
    def train_stock_model(self, ticker, target_col, model_type='random_forest', **kwargs):
        """
        End-to-end model training for a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            target_col (str): Target column to predict
            model_type (str): Type of model to train
            **kwargs: Additional parameters
            
        Returns:
            tuple: model, metrics, model_file_path
        """
        # Load processed data
        data_file = os.path.join(self.data_dir, f"{ticker}_processed.csv")
        if not os.path.exists(data_file):
            logger.error(f"Processed data file not found: {data_file}")
            return None, None, None
        
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, feature_cols = self.prepare_data(
            df, target_col, **kwargs
        )
        
        # Train model
        model = self.train_model(X_train, y_train, model_type, **kwargs)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Save model
        model_file = self.save_model(model, scaler, feature_cols, ticker, target_col, model_type)
        
        return model, metrics, model_file