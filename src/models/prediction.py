import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, models_dir='models', output_dir='data/predictions'):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model(self, model_file):
        """
        Load a trained model package.
        
        Args:
            model_file (str): Path to the model file
            
        Returns:
            dict: Model package containing model, scaler, and metadata
        """
        try:
            with open(model_file, 'rb') as f:
                model_package = pickle.load(f)
            
            logger.info(f"Loaded model from {model_file}")
            return model_package
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def prepare_prediction_data(self, df, feature_cols, scaler):
        """
        Prepare data for prediction.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            feature_cols (list): Feature column names
            scaler (object): Feature scaler
            
        Returns:
            np.array: Scaled features for prediction
        """
        # Select features and handle missing values
        X = df[feature_cols].copy()
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    def predict(self, model_package, df):
        """
        Make predictions using a trained model.
        
        Args:
            model_package (dict): Model package
            df (pd.DataFrame): DataFrame with features
            
        Returns:
            np.array: Predictions
        """
        model = model_package['model']
        scaler = model_package['scaler']
        feature_cols = model_package['feature_cols']
        
        # Prepare data
        X = self.prepare_prediction_data(df, feature_cols, scaler)
        
        # Make predictions
        predictions = model.predict(X)
        
        return predictions
    
    def predict_for_ticker(self, ticker, df=None, model_type='random_forest', target_col='future_price_5d'):
        """
        Make predictions for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            df (pd.DataFrame, optional): DataFrame with features, if None will load from file
            model_type (str): Type of model to use
            target_col (str): Target column that was predicted
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # Find model file
        model_file = os.path.join(self.models_dir, f"{ticker}_{target_col}_{model_type}_model.pkl")
        if not os.path.exists(model_file):
            logger.error(f"Model file not found: {model_file}")
            return None
        
        # Load model
        model_package = self.load_model(model_file)
        if model_package is None:
            return None
        
        # Load data if not provided
        if df is None:
            processed_file = f"data/features/{ticker}_processed.csv"
            if not os.path.exists(processed_file):
                logger.error(f"Processed data file not found: {processed_file}")
                return None
            
            df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        
        # Make predictions
        predictions = self.predict(model_package, df)
        
        # Add predictions to DataFrame
        result_df = pd.DataFrame(index=df.index)
        result_df['actual_price'] = df['Close']
        result_df['predicted_target'] = predictions
        
        # For future price prediction, we can calculate the predicted price directly
        if target_col.startswith('future_price_'):
            result_df['predicted_price'] = predictions
        elif target_col.startswith('future_return_'):
            # For return prediction, convert to price
            result_df['predicted_return'] = predictions
            result_df['predicted_price'] = df['Close'] * (1 + predictions)
        
        # Save predictions
        output_file = os.path.join(self.output_dir, f"{ticker}_predictions.csv")
        result_df.to_csv(output_file)
        logger.info(f"Saved predictions to {output_file}")
        
        return result_df
    
    def evaluate_predictions(self, predictions_df, target_col='future_price_5d'):
        """
        Evaluate prediction accuracy.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            target_col (str): Target column that was predicted
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # For future price predictions, we need to compare with the actual future price
        if target_col.startswith('future_price_'):
            days_ahead = int(target_col.split('_')[-1][:-1])  # Extract number from 'future_price_5d'
            
            try:
                # Shift the actual price back to align with predictions
                actual_future = predictions_df['actual_price'].shift(-days_ahead)
                predicted = predictions_df['predicted_price']
                
                # Make a copy of the actual price for directional comparison
                actual_price = predictions_df['actual_price'].copy()
                
                # Remove NaN values (at the end of the dataset where we don't have actual future values)
                valid_idx = ~actual_future.isna()
                actual_future = actual_future[valid_idx]
                predicted = predicted[valid_idx]
                actual_price_valid = actual_price[valid_idx]
                
                # Calculate metrics
                mse = np.mean((actual_future - predicted) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(actual_future - predicted))
                
                # Calculate directional accuracy (if price movement direction was predicted correctly)
                # Convert to numpy arrays to avoid index alignment issues
                actual_direction = (actual_future.values > actual_price_valid.values).astype(int)
                predicted_direction = (predicted.values > actual_price_valid.values).astype(int)
                directional_accuracy = np.mean(actual_direction == predicted_direction)
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'directional_accuracy': directional_accuracy
                }
                
                logger.info(f"Prediction evaluation metrics: {metrics}")
                return metrics
            except Exception as e:
                logger.error(f"Error evaluating predictions: {str(e)}")
                # Return basic metrics without directional accuracy
                return {
                    'mse': 0,
                    'rmse': 0,
                    'mae': 0,
                    'directional_accuracy': 0
                }
        
        return None