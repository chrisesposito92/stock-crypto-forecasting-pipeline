#!/usr/bin/env python
import argparse
import logging
import json
from src.pipeline.pipeline import StockForecastingPipeline

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Market Forecasting Pipeline')
    
    parser.add_argument(
        '--tickers', 
        type=str, 
        nargs='+', 
        default=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
        help='List of ticker symbols'
    )
    
    parser.add_argument(
        '--asset-type',
        type=str,
        default='stock',
        choices=['stock', 'crypto'],
        help='Type of asset to forecast (stock or crypto)'
    )
    
    parser.add_argument(
        '--start-date', 
        type=str, 
        help='Start date for historical data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str, 
        help='End date for historical data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--forecast-horizon', 
        type=int, 
        default=5,
        help='Number of days to forecast ahead'
    )
    
    parser.add_argument(
        '--model-type', 
        type=str, 
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'linear'],
        help='Type of model to use'
    )
    
    parser.add_argument(
        '--update-only', 
        action='store_true',
        help='Only update data without running the full pipeline'
    )
    
    parser.add_argument(
        '--retrain', 
        action='store_true',
        help='Retrain models with latest data'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    # Set default tickers based on asset type if none provided
    if args.asset_type == 'crypto' and args.tickers == ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']:
        args.tickers = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
        logger.info(f"Using default crypto tickers: {args.tickers}")
    
    # Initialize pipeline
    pipeline = StockForecastingPipeline(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        forecast_horizon=args.forecast_horizon,
        model_type=args.model_type,
        asset_type=args.asset_type
    )
    
    # Run pipeline or update data based on arguments
    if args.update_only:
        logger.info("Running update-only mode")
        updated_data = pipeline.update_data()
        
        if args.retrain:
            logger.info("Retraining models with updated data")
            pipeline.retrain_models()
            
        logger.info("Generating latest predictions")
        predictions = pipeline.run_predictions()
        
        # Generate performance report
        report = pipeline.generate_report(predictions)
        logger.info(f"\nPerformance Report:\n{report}")
    else:
        logger.info("Running full pipeline")
        results = pipeline.run_pipeline()
        
        # Generate performance report
        report = pipeline.generate_report(results['predictions'])
        logger.info(f"\nPerformance Report:\n{report}")
        
        # Log model evaluation metrics
        for ticker, info in results['models_info'].items():
            logger.info(f"\nModel metrics for {ticker}:\n{json.dumps(info['metrics'], indent=2)}")
    
    logger.info("Pipeline execution completed")
    
    # Automatically show predictions after pipeline run
    logger.info("Displaying predictions for this run...")
    
    # Import and use the show_predictions functionality
    from show_predictions import show_recent_predictions
    show_recent_predictions(
        tickers=args.tickers, 
        forecast_horizon=args.forecast_horizon,
        asset_type=args.asset_type,
        decimal_places=6 if args.asset_type == 'crypto' else 2
    )

if __name__ == "__main__":
    main()