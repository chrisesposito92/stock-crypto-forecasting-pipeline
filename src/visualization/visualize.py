import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.dates import DateFormatter

class StockVisualizer:
    def __init__(self, output_dir='visualization'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_context("talk")
    
    def plot_stock_price(self, df, ticker, save=True):
        """
        Plot historical stock price with volume.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            ticker (str): Stock ticker symbol
            save (bool): Whether to save the plot
            
        Returns:
            tuple: Figure and axes objects
        """
        # Ensure all data is numeric and properly formatted
        df_clean = df.copy()
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Drop any NaN values
        df_clean = df_clean.dropna()
        
        # Convert to pandas dataframe with datetime index if it's not already
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except:
                # If conversion fails, create a numerical index
                df_clean = df_clean.reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot price using line plot
        ax1.plot(df_clean['Close'], label='Close Price', color='blue', linewidth=2)
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f"{ticker} Stock Price")
        ax1.legend()
        ax1.grid(True)
        
        # Plot volume using line plot instead of bar
        ax2.plot(df_clean['Volume'], color='gray', alpha=0.7)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        # Format axis
        if isinstance(df_clean.index, pd.DatetimeIndex):
            date_format = DateFormatter('%Y-%m-%d')
            ax2.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            fig.savefig(os.path.join(self.output_dir, f"{ticker}_price_history.png"))
        
        return fig, (ax1, ax2)
    
    def plot_technical_indicators(self, df, ticker, indicators=None, save=True):
        """
        Plot stock price with technical indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            ticker (str): Stock ticker symbol
            indicators (list, optional): List of indicators to plot
            save (bool): Whether to save the plot
            
        Returns:
            tuple: Figure and axes objects
        """
        if indicators is None:
            indicators = ['SMA_20', 'SMA_50', 'BB_upper', 'BB_lower']
        
        # Ensure all data is numeric and properly formatted
        df_clean = df.copy()
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Drop any NaN values
        df_clean = df_clean.dropna()
        
        # Convert to pandas dataframe with datetime index if it's not already
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except:
                # If conversion fails, create a numerical index
                df_clean = df_clean.reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax.plot(df_clean['Close'], label='Close Price', color='blue', linewidth=2)
        
        # Plot selected indicators
        colors = ['green', 'red', 'purple', 'orange', 'brown', 'pink']
        for i, indicator in enumerate(indicators):
            if indicator in df_clean.columns:
                color = colors[i % len(colors)]
                ax.plot(df_clean[indicator], label=indicator, color=color, alpha=0.7)
        
        ax.set_ylabel('Price ($)')
        ax.set_xlabel('Date')
        ax.set_title(f"{ticker} Stock Price with Technical Indicators")
        ax.legend()
        ax.grid(True)
        
        # Format axis
        if isinstance(df_clean.index, pd.DatetimeIndex):
            date_format = DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            fig.savefig(os.path.join(self.output_dir, f"{ticker}_technical_indicators.png"))
        
        return fig, ax
    
    def plot_predictions(self, predictions_df, ticker, days_ahead=5, save=True):
        """
        Plot actual vs predicted prices.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            ticker (str): Stock ticker symbol
            days_ahead (int): Number of days ahead for prediction
            save (bool): Whether to save the plot
            
        Returns:
            tuple: Figure and axes objects
        """
        # Ensure all data is numeric and properly formatted
        df_clean = predictions_df.copy()
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Drop any NaN values
        df_clean = df_clean.dropna()
        
        # Convert to pandas dataframe with datetime index if it's not already
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except:
                # If conversion fails, create a numerical index
                df_clean = df_clean.reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot actual price
        ax.plot(df_clean['actual_price'], 
                label='Actual Price', color='blue', linewidth=2)
        
        # Plot predicted price (need to align with actual future price for fair comparison)
        if 'predicted_price' in df_clean.columns:
            # Shift the predictions forward to align with the dates they predict
            predicted_aligned = df_clean['predicted_price'].shift(days_ahead)
            ax.plot(predicted_aligned, 
                    label=f'Predicted Price ({days_ahead}-day Forecast)', 
                    color='red', linestyle='--', linewidth=2)
        
        ax.set_ylabel('Price ($)')
        ax.set_xlabel('Date')
        ax.set_title(f"{ticker} Actual vs Predicted Stock Price")
        ax.legend()
        ax.grid(True)
        
        # Format axis
        if isinstance(df_clean.index, pd.DatetimeIndex):
            date_format = DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()
        
        # Add annotation about model performance if available
        try:
            from src.models.prediction import StockPredictor
            predictor = StockPredictor()
            metrics = predictor.evaluate_predictions(df_clean, f'future_price_{days_ahead}d')
            if metrics:
                text = f"RMSE: ${metrics['rmse']:.2f}\nMAE: ${metrics['mae']:.2f}\nDirectional Accuracy: {metrics['directional_accuracy']:.1%}"
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            fig.savefig(os.path.join(self.output_dir, f"{ticker}_predictions.png"))
        
        return fig, ax
    
    def plot_feature_importance(self, model_package, top_n=15, save=True):
        """
        Plot feature importance from a trained model.
        
        Args:
            model_package (dict): Model package with model and feature columns
            top_n (int): Number of top features to show
            save (bool): Whether to save the plot
            
        Returns:
            tuple: Figure and axes objects
        """
        model = model_package['model']
        feature_cols = model_package['feature_cols']
        ticker = model_package['ticker']
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            return None, None
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Select top N features
        top_indices = indices[:min(top_n, len(indices))]
        top_features = [feature_cols[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_importances)), top_importances, align='center')
        ax.set_yticks(range(len(top_importances)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {min(top_n, len(top_importances))} Feature Importance for {ticker}')
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            fig.savefig(os.path.join(self.output_dir, f"{ticker}_feature_importance.png"))
        
        return fig, ax
    
    def plot_correlation_matrix(self, df, ticker, save=True):
        """
        Plot correlation matrix of features.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            ticker (str): Stock ticker symbol
            save (bool): Whether to save the plot
            
        Returns:
            tuple: Figure and axes objects
        """
        # Ensure all data is numeric and properly formatted
        df_clean = df.copy()
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Select only numeric columns
        numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
        
        # Calculate correlation matrix
        corr = numeric_df.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        
        ax.set_title(f'Feature Correlation Matrix for {ticker}')
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            fig.savefig(os.path.join(self.output_dir, f"{ticker}_correlation_matrix.png"))
        
        return fig, ax