# Stock Market Data Pipeline & Price Prediction

This project is a comprehensive data pipeline for analyzing historical stock market data and predicting future stock prices. It demonstrates both data engineering and data science capabilities through a modular, maintainable codebase.

## Project Structure

```
stock-forecasting-pipeline/
├── data/                    # Data storage
│   ├── raw/                 # Raw stock data from APIs
│   ├── features/            # Processed data with engineered features
│   └── predictions/         # Model predictions
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data/                # Data acquisition and processing
│   ├── models/              # Model training and prediction
│   ├── visualization/       # Data visualization
│   └── pipeline/            # Pipeline orchestration
├── visualization/           # Generated visualizations
├── main.py                  # Command-line interface
└── requirements.txt         # Project dependencies
```

## Features

- **Data Collection**: Downloads historical stock data from Yahoo Finance using the yfinance library
- **Feature Engineering**: Adds technical indicators, time-based features, and target variables for prediction
- **Model Training**: Trains machine learning models (Random Forest, Gradient Boosting, Linear Regression) to predict future stock prices
- **Prediction**: Generates price forecasts for specified time horizons
- **Visualization**: Creates insightful visualizations of stock data, technical indicators, and predictions
- **Pipeline Automation**: Orchestrates the entire process from data collection to prediction

## Installation

1. Clone the repository:

```bash
git clone https://github.com/chrisesposito92/stock-forecasting-pipeline.git
cd stock-forecasting-pipeline
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the pipeline with default settings (analyzes AAPL, MSFT, GOOG, AMZN, META):

```bash
python main.py
```

Specify custom tickers and date range:

```bash
python main.py --tickers AAPL MSFT GOOGL --start-date 2022-01-01 --end-date 2023-12-31
```

Change the forecast horizon (number of days to predict ahead):

```bash
python main.py --forecast-horizon 10
```

Change the model type:

```bash
python main.py --model-type gradient_boosting
```

Update data only (without retraining models):

```bash
python main.py --update-only
```

Update data and retrain models:

```bash
python main.py --update-only --retrain
```

### Jupyter Notebook

Explore the pipeline interactively using the provided notebook:

```bash
jupyter notebook notebooks/stock_analysis.ipynb
```

## Model Evaluation

The pipeline evaluates model performance using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Directional Accuracy (percentage of times the price movement direction is predicted correctly)

A performance report is generated after each pipeline run, comparing the metrics across different stocks.

## Visualizations

The pipeline generates various visualizations, saved to the `visualization/` directory:

- Price history with volume
- Technical indicators (moving averages, Bollinger Bands, etc.)
- Actual vs. predicted prices
- Feature importance
- Correlation matrix

## Pipeline Architecture

The pipeline consists of the following components:

1. **Data Collection**: Downloads historical stock data using yfinance
2. **Feature Engineering**: Processes raw data and adds features useful for prediction
3. **Model Training**: Prepares data, trains models, and evaluates performance
4. **Prediction**: Generates forecasts using trained models
5. **Visualization**: Creates visual representations of the data and results

## Requirements

- Python 3.7+
- yfinance
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
- joblib

## Future Improvements

- Add support for more data sources (Alpha Vantage, Twelve Data, etc.)
- Implement more advanced models (LSTM, Prophet, etc.)
- Create a web dashboard for interactive visualization
- Add portfolio optimization features
- Implement automated data updates with Apache Airflow

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for providing stock data
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [pandas](https://pandas.pydata.org/) for data manipulation
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualization
