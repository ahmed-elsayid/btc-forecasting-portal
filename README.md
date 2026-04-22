# Bitcoin Price Forecasting Portal

An interactive web application built with Streamlit to analyze and forecast Bitcoin (BTC) price trends using Statistical and Machine Learning models.

## Features
- **Data Agnostic**: Supports any standard Kaggle historical BTC CSV file
- **Auto-parsing**: Automatically detects datetime columns and allows target price selection (Close, Open, High, Low)
- **3 Forecasting Engines**: Prophet, ARIMA, and ML Ensemble (Linear Regression + Random Forest)
- **Backtesting**: 80/20 train-test split with out-of-sample MAE and RMSE in USD
- **Interactive UI**: Adjustable forecast horizon, confidence intervals, and moving average overlay

## Dataset
**Kaggle BTC Dataset used for testing:**
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

Download `btcusd_1-min_data.csv` or any daily OHLC CSV from that page.

## Setup Instructions

1. **Clone the repository or extract the `.zip`**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows