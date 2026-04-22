# Bitcoin Price Forecasting Portal

An interactive web application built with Streamlit to analyze and forecast Bitcoin (BTC) price trends using Statistical, Machine Learning, and Deep Learning models.

## Features
- **Data Agnostic**: Supports any standard Kaggle historical BTC CSV file
- **Auto-parsing**: Automatically detects datetime columns and allows target price selection (Close, Open, High, Low)
- **4 Forecasting Engines**: Prophet, ARIMA, ML Ensemble (LR + Random Forest), Deep Learning (LSTM)
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
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Upload your CSV** from the sidebar and configure your forecast settings.

