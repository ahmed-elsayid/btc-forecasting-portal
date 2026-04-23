# Bitcoin Price Forecaster 🪙

This project analyzes and predicts Bitcoin (BTC) price trends using historical data. It features three different forecasting engines (Prophet, ARIMA, and a Machine Learning Ensemble) and calculates real-time accuracy metrics through automated backtesting. 

You can interact with the models using either a visual web dashboard or a backend API.

## Dataset
This app is designed to work with standard Kaggle BTC data. For testing, we used the [Bitcoin Historical Dataset from Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data).

## Setup
Here is how to get the project running on your local machine:

1. Clone the repository and navigate into the folder.
2. (Optional but recommended) Create and activate a virtual environment.
3. Install the required libraries:
   
   ```bash
   pip install -r requirements.txt
   ```

## How to Use It

You have two ways to run this project depending on what you want to do:

### 1. The Web Dashboard (Streamlit)
If you want to visually interact with the charts and settings, run the Streamlit app:
```bash
streamlit run app.py
```
This will open a dashboard in your browser. Just upload your CSV, pick your model and settings, and click "Run Forecast".

### 2. The Backend API (FastAPI)
If you want to use the models programmatically, run the API server:
```bash
uvicorn api:app --reload
```

**How to test the API:** 
Because the API requires a file upload, you can't just type the `http://.../predict` link directly into your search bar. Instead:
1. Go to **http://127.0.0.1:8000/docs** in your browser.
2. Click on the green `POST /predict` box, then click **Try it out**.
3. Upload your CSV file, adjust any parameters you want, and click **Execute**. 
4. The server will return a JSON response with your model's accuracy metrics and daily price predictions.
```