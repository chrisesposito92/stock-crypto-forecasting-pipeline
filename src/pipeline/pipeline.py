import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.model_training import ModelTrainer
from src.models.prediction import StockPredictor
from src.visualization.visualize import StockVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockForecastingPipeline:
    def __init__(self, 
                 tickers=None, 
                 start_date=None, 
                 end_date=None,
                 forecast_horizon=5,
                 model_type='random_forest'):
        """
        Initialize the pipeline with configuration.
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str): End date for historical data (YYYY-MM-DD)
            forecast_horizon (int): Number of days to forecast ahead
            model_type (str): Type of model to use
        """
        # Set default tickers if none provided
        self.tickers = tickers or ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
        
        # Set default date range if none provided (e.g., last 5 years)
        if start_date is None:
            self.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        
        # Initialize components
        self.data_loader = StockDataLoader(output_dir='data/raw')
        self.feature_engineer = FeatureEngineer(input_dir='data/raw', output_dir='data/features')
        self.model_trainer = ModelTrainer(data_dir='data/features', models_dir='models')
        self.predictor = StockPredictor(models_dir='models', output_dir='data/predictions')
        self.visualizer = StockVisualizer(output_dir='visualization')
        
        # Create output directories
        os.makedirs('logs', exist_ok=True)
        
        logger.info(f"Initialized pipeline with tickers: {self.tickers}")
    
    def run_data_collection(self):
        """
        Collect historical stock data.
        
        Returns:
            dict: Dictionary with tickers as keys and DataFrames as values
        """
        logger.info("Starting data collection...")
        stock_data = self.data_loader.download_multiple_stocks(
            self.tickers, self.start_date, self.end_date
        )
        logger.info(f"Collected data for {len(stock_data)} stocks")
        return stock_data
    
    def run_feature_engineering(self):
        """
        Process stock data and engineer features.
        
        Returns:
            dict: Dictionary with tickers as keys and processed DataFrames as values
        """
        logger.info("Starting feature engineering...")
        processed_data = {}
        
        for ticker in self.tickers:
            df = self.feature_engineer.process_stock_data(
                ticker, 
                add_time=True, 
                add_technicals=True, 
                add_targets=True, 
                forecast_horizon=self.forecast_horizon
            )
            
            if df is not None:
                processed_data[ticker] = df
        
        logger.info(f"Engineered features for {len(processed_data)} stocks")
        return processed_data
    
    def run_model_training(self):
        """
        Train predictive models for each stock.
        
        Returns:
            dict: Dictionary with tickers as keys and model metrics as values
        """
        logger.info("Starting model training...")
        models_info = {}
        
        for ticker in self.tickers:
            logger.info(f"Training model for {ticker}...")
            
            target_col = f'future_price_{self.forecast_horizon}d'
            model, metrics, model_file = self.model_trainer.train_stock_model(
                ticker, target_col, self.model_type
            )
            
            if model is not None:
                models_info[ticker] = {
                    'metrics': metrics,
                    'model_file': model_file
                }
        
        logger.info(f"Trained models for {len(models_info)} stocks")
        return models_info
    
    def run_predictions(self):
        """
        Generate predictions using trained models.
        
        Returns:
            dict: Dictionary with tickers as keys and prediction DataFrames as values
        """
        logger.info("Generating predictions...")
        predictions = {}
        
        for ticker in self.tickers:
            logger.info(f"Predicting for {ticker}...")
            
            target_col = f'future_price_{self.forecast_horizon}d'
            pred_df = self.predictor.predict_for_ticker(
                ticker, df=None, model_type=self.model_type, target_col=target_col
            )
            
            if pred_df is not None:
                predictions[ticker] = pred_df
        
        logger.info(f"Generated predictions for {len(predictions)} stocks")
        return predictions
    
    def run_visualization(self, stock_data, processed_data, predictions):
        """
        Create visualizations of data and predictions.
        
        Args:
            stock_data (dict): Dictionary with raw stock data
            processed_data (dict): Dictionary with processed data
            predictions (dict): Dictionary with predictions
            
        Returns:
            dict: Dictionary with paths to visualization files
        """
        logger.info("Creating visualizations...")
        visualization_files = {}
        
        for ticker in self.tickers:
            if ticker in stock_data and ticker in processed_data and ticker in predictions:
                # Plot price history
                self.visualizer.plot_stock_price(stock_data[ticker], ticker)
                
                # Plot technical indicators
                self.visualizer.plot_technical_indicators(processed_data[ticker], ticker)
                
                # Plot predictions
                self.visualizer.plot_predictions(predictions[ticker], ticker, self.forecast_horizon)
                
                # Plot feature importance
                model_file = f"models/{ticker}_future_price_{self.forecast_horizon}d_{self.model_type}_model.pkl"
                if os.path.exists(model_file):
                    model_package = self.model_trainer.load_model(model_file)
                    self.visualizer.plot_feature_importance(model_package)
                
                # Plot correlation matrix
                self.visualizer.plot_correlation_matrix(processed_data[ticker], ticker)
                
                visualization_files[ticker] = {
                    'price_history': f"visualization/{ticker}_price_history.png",
                    'technical_indicators': f"visualization/{ticker}_technical_indicators.png",
                    'predictions': f"visualization/{ticker}_predictions.png",
                    'feature_importance': f"visualization/{ticker}_feature_importance.png",
                    'correlation_matrix': f"visualization/{ticker}_correlation_matrix.png"
                }
        
        logger.info(f"Created visualizations for {len(visualization_files)} stocks")
        return visualization_files
    
    def run_pipeline(self):
        """
        Execute the full pipeline.
        
        Returns:
            dict: Dictionary with results from each stage
        """
        logger.info("Starting full pipeline execution...")
        
        # Step 1: Data Collection
        stock_data = self.run_data_collection()
        
        # Step 2: Feature Engineering
        processed_data = self.run_feature_engineering()
        
        # Step 3: Model Training
        models_info = self.run_model_training()
        
        # Step 4: Predictions
        predictions = self.run_predictions()
        
        # Step 5: Visualizations
        visualization_files = self.run_visualization(stock_data, processed_data, predictions)
        
        logger.info("Pipeline execution completed")
        
        return {
            'stock_data': stock_data,
            'processed_data': processed_data,
            'models_info': models_info,
            'predictions': predictions,
            'visualization_files': visualization_files
        }
    
    def update_data(self):
        """
        Update data with the latest stock prices.
        
        Returns:
            dict: Dictionary with updated stock data
        """
        logger.info("Updating data with latest stock prices...")
        
        # Set date range for update (last week to today)
        update_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        update_end = datetime.now().strftime('%Y-%m-%d')
        
        # Download latest data
        updated_data = self.data_loader.download_multiple_stocks(
            self.tickers, update_start, update_end
        )
        
        # Process updated data
        for ticker in updated_data:
            self.feature_engineer.process_stock_data(ticker)
        
        logger.info(f"Updated data for {len(updated_data)} stocks")
        return updated_data
    
    def retrain_models(self):
        """
        Retrain models with the latest data.
        
        Returns:
            dict: Dictionary with updated model metrics
        """
        logger.info("Retraining models with latest data...")
        return self.run_model_training()
    
    def generate_report(self, predictions):
        """
        Generate a simple performance report.
        
        Args:
            predictions (dict): Dictionary with predictions
            
        Returns:
            pd.DataFrame: Report DataFrame
        """
        logger.info("Generating performance report...")
        
        report_data = []
        
        for ticker, pred_df in predictions.items():
            metrics = self.predictor.evaluate_predictions(
                pred_df, target_col=f'future_price_{self.forecast_horizon}d'
            )
            
            if metrics:
                report_data.append({
                    'ticker': ticker,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'directional_accuracy': metrics['directional_accuracy']
                })
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_path = 'data/predictions/performance_report.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f"Saved performance report to {report_path}")
        
        return report_df