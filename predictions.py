import os
import numpy as np
import pandas as pd
import joblib
from datetime import date, datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

from data_loader import fetch_stock_data
from feature_indicators import add_indicators
from data_processor import build_sequences
from config import TIME_STEPS, TICKER, START_DATE, SEED

LOG_FILE = "results_log.csv"


def log_result(row):
    """Appends one experiment result to results_log.csv (creates it if missing)."""
    pd.DataFrame([row]).to_csv(
        LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False
    )


def evaluate_and_predict(ticker=TICKER, start_date=START_DATE):
    print("Loading saved model, scalers, features and frozen timeline...")
    model = load_model("lstm_stock_model.keras")
    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    frozen_split_date = joblib.load("split_date.pkl")
    features = joblib.load("features.pkl")   # exactly what the model was trained on

    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching data from {start_date} to {today_str}...")
    df = add_indicators(fetch_stock_data(ticker, start_date, today_str))

    # Test data = everything from the frozen split date on.
    # We keep TIME_STEPS earlier rows so the first test window is complete.
    split_pos = df.index.searchsorted(pd.Timestamp(frozen_split_date))
    test_df = df.iloc[split_pos - TIME_STEPS:].copy()

    x_test, y_test_scaled = build_sequences(
        test_df, features, x_scaler, y_scaler, TIME_STEPS
    )

    # Predict, then convert back to real daily returns (e.g. 0.004 = +0.4%)
    scaled_predictions = model.predict(x_test, verbose=0)
    predicted = y_scaler.inverse_transform(scaled_predictions).ravel()
    actual = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    # --- Metrics ---
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    baseline_rmse = np.sqrt(mean_squared_error(actual, np.zeros_like(actual)))  # "return = 0"

    mda = np.mean(np.sign(actual) == np.sign(predicted)) * 100
    majority = max(np.mean(actual > 0), np.mean(actual < 0)) * 100
    n = len(actual)
    margin = 98 / np.sqrt(n)                      # ~95% "luck zone" in percentage points
    pct_pred_up = np.mean(predicted > 0) * 100    # does the model just say UP all the time?

    beats_rmse = rmse < baseline_rmse
    beats_direction = mda > majority + margin

    print("\n--------------------------------------------------")
    print(f" Stock: {ticker}  | Features: {features}")
    print(f" Test data since {frozen_split_date} ({n} days)")
    print(f" AI Model Error (RMSE):  {rmse:.5f}")
    print(f" Naive Baseline (RMSE):  {baseline_rmse:.5f}")
    print("--------------------------------------------------")
    print(" ✅ AI beats the baseline in magnitude." if beats_rmse
          else " ⚠️ AI does not beat the baseline in magnitude.")
    print(f" Directional Accuracy: {mda:.2f}%  "
          f"(majority guess: {majority:.2f}%, noise ±{margin:.1f}%)")
    print(" ✅ Direction beats the majority baseline by more than the noise margin."
          if beats_direction
          else " ⚠️ Direction is not distinguishable from guessing.")
    print(f" Model predicted UP on {pct_pred_up:.0f}% of test days")
    print("--------------------------------------------------")

    # --- Log this experiment ---
    log_result({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "features": "+".join(features),
        "seed": SEED,
        "split_date": frozen_split_date,
        "test_days": n,
        "rmse": round(float(rmse), 6),
        "baseline_rmse": round(float(baseline_rmse), 6),
        "mda": round(float(mda), 2),
        "majority": round(float(majority), 2),
        "noise_margin": round(float(margin), 2),
        "pct_pred_up": round(float(pct_pred_up), 1),
        "beats_rmse": bool(beats_rmse),
        "beats_direction": bool(beats_direction),
    })
    print(f" Result saved to {LOG_FILE}")

    # --- Experimental forecast for the next trading day ---
    # (yfinance's end date is exclusive, so the last row is the most recent closed day
    #  before today.) Treat this as a demo until the model beats the noise margin.
    last_window = df[features].tail(TIME_STEPS).to_numpy()
    x_tomorrow = x_scaler.transform(last_window).reshape(1, TIME_STEPS, len(features))

    scaled_tomorrow = model.predict(x_tomorrow, verbose=0)
    predicted_return = float(y_scaler.inverse_transform(scaled_tomorrow)[0][0])

    latest_close = float(df["Close"].iloc[-1])
    forecast_price = latest_close * (1 + predicted_return)
    direction = "📈 UP" if predicted_return > 0 else "📉 DOWN"

    print("\n🔮 EXPERIMENTAL FORECAST FOR THE NEXT TRADING DAY")
    print(f" Forecasted Return:      {predicted_return * 100:.2f}%")
    print(f" Forecasted Close Price: ₹{forecast_price:.2f}")
    print(f" Forecasted Movement:    {direction} (by ₹{abs(forecast_price - latest_close):.2f})\n")

    return predicted, actual


if __name__ == "__main__":
    evaluate_and_predict()