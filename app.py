import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    st.error("Please install tensorflow: pip install tensorflow-cpu")

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BTC Forecaster", layout="wide")
st.markdown("<style>.stApp{margin-top:-40px}.stButton>button{width:100%;border-radius:5px;font-weight:bold}</style>", unsafe_allow_html=True)
st.title("Bitcoin Price Forecasting")
st.markdown("Upload your Kaggle CSV to train ML, DL, and Statistical models on the fly.")


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    time_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "time" in col_lower or "date" in col_lower:
            time_col = col
            break

    price_cols = []
    for col in df.columns:
        if col.lower() in ["close", "open", "high", "low", "price"]:
            price_cols.append(col)

    if time_col is None or len(price_cols) == 0:
        return None, None

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)
    df = df.set_index(time_col)
    df = df.resample("D").ffill().bfill()
    df = df.dropna()

    return df, price_cols


def metrics(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

def z(conf):
    return 1.96 if conf == 95 else 1.28

def future_dates(df, horizon):
    return pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon)


def run_prophet(train, test, full, col, horizon, conf):
    def to_prophet(df): return df.reset_index().rename(columns={df.index.name: 'ds', col: 'y'})
    m = Prophet(interval_width=conf / 100).fit(to_prophet(train))
    preds = m.predict(m.make_future_dataframe(periods=len(test)))
    mae, rmse = metrics(test[col].values, preds['yhat'][-len(test):])
    m2 = Prophet(interval_width=conf / 100).fit(to_prophet(full))
    fc = m2.predict(m2.make_future_dataframe(periods=horizon))[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return fc, mae, rmse


def run_arima(train, test, full, col, horizon, conf):
    mae, rmse = metrics(test[col], ARIMA(train[col], order=(5, 1, 0)).fit().forecast(steps=len(test)))
    fc = ARIMA(full[col], order=(5, 1, 0)).fit().get_forecast(steps=horizon)
    ci = fc.conf_int(alpha=1 - conf / 100)
    return pd.DataFrame({'ds': future_dates(full, horizon), 'yhat': fc.predicted_mean.values,
                         'yhat_lower': ci.iloc[:, 0].values, 'yhat_upper': ci.iloc[:, 1].values}), mae, rmse


def run_ml_ensemble(train, test, full, col, horizon, conf):
    def fit_detrended(X, y):
        lr = LinearRegression().fit(X, y)
        trend = lr.predict(X)
        rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y - trend)
        return lr, rf

    def predict_detrended(lr, rf, X):
        return lr.predict(X) + rf.predict(X)

    # Backtest
    X_tr = np.arange(len(train)).reshape(-1, 1)
    X_te = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    lr, rf = fit_detrended(X_tr, train[col].values)
    mae, rmse = metrics(test[col].values, predict_detrended(lr, rf, X_te))

    # Forecast on full data
    X_full = np.arange(len(full)).reshape(-1, 1)
    X_fut  = np.arange(len(full), len(full) + horizon).reshape(-1, 1)
    lr2, rf2 = fit_detrended(X_full, full[col].values)
    yhat = predict_detrended(lr2, rf2, X_fut)

    residuals = full[col].values - predict_detrended(lr2, rf2, X_full)
    margin = z(conf) * np.std(residuals)
    return pd.DataFrame({'ds': future_dates(full, horizon), 'yhat': yhat,
                         'yhat_lower': yhat - margin, 'yhat_upper': yhat + margin}), mae, rmse


def run_lstm(train, test, full, col, horizon, conf):
    scaler, lookback = MinMaxScaler(), 30

    def make_sequences(data):
        s = scaler.fit_transform(data[[col]])
        X = [s[i - lookback:i, 0] for i in range(lookback, len(s))]
        return np.array(X).reshape(-1, lookback, 1), s[lookback:]

    def build_model():
        m = Sequential([LSTM(50, activation='relu', input_shape=(lookback, 1)), Dense(1)])
        m.compile(optimizer='adam', loss='mse')
        return m

    X_tr, y_tr = make_sequences(train)
    model = build_model()
    model.fit(X_tr, y_tr, epochs=5, batch_size=32, verbose=0)

    test_in = scaler.transform(pd.concat([train.iloc[-lookback:], test])[[col]])
    X_ts = np.array([test_in[i - lookback:i, 0] for i in range(lookback, len(test_in))]).reshape(-1, lookback, 1)
    mae, rmse = metrics(test[col].values, scaler.inverse_transform(model.predict(X_ts, verbose=0)).flatten())

    X_full, y_full = make_sequences(full)
    model.fit(X_full, y_full, epochs=3, batch_size=32, verbose=0)

    batch = scaler.transform(full[[col]])[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(horizon):
        p = model.predict(batch, verbose=0)[0]
        preds.append(p[0])
        batch = np.append(batch[:, 1:, :], [[p]], axis=1)

    yhat = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    margin = z(conf) * full[col].pct_change().std() * full[col].iloc[-1] * np.sqrt(np.arange(1, horizon + 1))
    return pd.DataFrame({'ds': future_dates(full, horizon), 'yhat': yhat,
                         'yhat_lower': yhat - margin, 'yhat_upper': yhat + margin}), mae, rmse


# Sidebar
with st.sidebar:
    st.header("Configuration")
    file = st.file_uploader("Upload CSV", type=['csv'])
    if file:
        df, p_cols = load_data(file)
        if df is not None:
            col = st.selectbox("Target Price", p_cols)
            model_type = st.selectbox("Algorithm", ["Prophet", "ARIMA", "ML Ensemble (LR+RF)", "Deep Learning (LSTM)"])
            horizon = st.slider("Horizon (Days)", 7, 90, 30)
            conf = st.selectbox("Confidence %", [80, 95], index=1)
            show_ma = st.toggle("Overlay Moving Averages")
            run_btn = st.button("Run Forecast", type="primary")

# Main
if file and df is not None:
    if run_btn:
        with st.spinner(f"Training {model_type}..."):
            split = int(len(df) * 0.8)
            train, test = df.iloc[:split], df.iloc[split:]

            runners = {
                "Prophet": run_prophet,
                "ARIMA": run_arima,
                "ML Ensemble (LR+RF)": run_ml_ensemble,
                "Deep Learning (LSTM)": run_lstm,
            }
            fc, mae, rmse = runners[model_type](train, test, df, col, horizon, conf)

            c1, c2 = st.columns(2)
            c1.metric("Test MAE", f"${mae:,.2f}")
            c2.metric("Test RMSE", f"${rmse:,.2f}")

            future = fc[fc['ds'] > df.index[-1]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name='Actual', line=dict(color='#F7931A', width=2)))
            if show_ma:
                fig.add_trace(go.Scatter(x=df.index, y=df[col].rolling(30).mean(), name='SMA 30',
                                         line=dict(color='white', dash='dot', width=1)))
            # Linear trend line over full historical data
            X_idx = np.arange(len(df)).reshape(-1, 1)
            trend_line = LinearRegression().fit(X_idx, df[col].values).predict(X_idx)
            fig.add_trace(go.Scatter(x=df.index, y=trend_line, name='Linear Trend',
                                     line=dict(color='#A78BFA', dash='dash', width=1.5)))

            fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], name='Forecast', line=dict(color='#00FF00', width=2)))
            fig.add_trace(go.Scatter(
                x=pd.concat([future['ds'], future['ds'][::-1]]),
                y=pd.concat([future['yhat_upper'], future['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(0,255,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                name=f'{conf}% Band'
            ))
            fig.add_vline(x=df.index[-1], line_width=2, line_dash="dash", line_color="#00FF00")
            fig.add_annotation(x=df.index[-1], y=df[col].max(), text="<b>Forecast Start</b>",
                               showarrow=True, arrowhead=2, ax=-70, ay=-30,
                               font=dict(color="black", size=12), bgcolor="#00FF00", borderpad=4)
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0),
                              legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig, use_container_width=True)

elif file and df is None:
    st.error("Invalid CSV. Could not find date/price columns.")
else:
    st.info("Upload a dataset to begin.")