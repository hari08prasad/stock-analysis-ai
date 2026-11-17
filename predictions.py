import os
import numpy as np
import pandas as pd
import joblib
from datetime import date, datetime
from tensorflow.keras.models import load_model

from data_loader import fetch_all
from feature_indicators import add_indicators
from data_processor import build_sequences
from config import TIME_STEPS, START_DATE, SEED

# Dedicated log file for Classification metrics
LOG_FILE = "results_log_class.csv"

def log_result(row):
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

def evaluate_metrics(ticker, actual_binary, predicted_probs, features, frozen_split_date, is_combined=False):
    # Convert probabilities to binary guesses: > 50% means UP (1)
    predicted_binary = (predicted_probs > 0.5).astype(int).ravel()
    
    mda = np.mean(actual_binary == predicted_binary) * 100
    majority = max(np.mean(actual_binary == 1), np.mean(actual_binary == 0)) * 100
    n = len(actual_binary)
    margin = 98 / np.sqrt(n) 
    
    pct_pred_up = np.mean(predicted_binary == 1) * 100
    beats_direction = mda > majority + margin

    if is_combined:
        print("\n==================================================")
        print(f" 🌐 COMBINED CLASSIFICATION ({n} total test days)")
        print("==================================================")
    else:
        print(f"\n--- Stock: {ticker} ({n} test days) ---")
        
    print(f" Directional Accuracy: {mda:.2f}%  (majority: {majority:.2f}%, noise ±{margin:.1f}%)")
    print(f" Model Predicted UP on {pct_pred_up:.1f}% of days.")
    
    if beats_direction:
        print(" ✅ WIN: Direction beats majority baseline by more than noise margin.")
    else:
        print(" ⚠️ FAIL: Direction indistinguishable from guessing.")

    log_result({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "features": "+".join(features),
        "seed": SEED,
        "split_date": frozen_split_date,
        "test_days": n,
        "mda": round(float(mda), 2),
        "majority": round(float(majority), 2),
        "noise_margin": round(float(margin), 2),
        "pct_pred_up": round(float(pct_pred_up), 1),
        "beats_direction": bool(beats_direction),
    })
    
    return beats_direction

def evaluate_and_predict(start_date=START_DATE):
    model = load_model("lstm_stock_model.keras")
    x_scaler = joblib.load("x_scaler.pkl")
    frozen_split_date = joblib.load("split_date.pkl")
    features = joblib.load("features.pkl")
    trained_tickers = joblib.load("tickers.pkl") 

    today_str = date.today().strftime("%Y-%m-%d")
    stock_dict = fetch_all(trained_tickers, start_date, today_str)

    all_actuals = []
    all_predicteds = []
    stocks_beating_baseline = 0
    
    for ticker in trained_tickers:
        df = stock_dict[ticker]
        
        # Passes the ticker to avoid news cross-contamination
        df = add_indicators(df, ticker=ticker)

        split_pos = df.index.searchsorted(pd.Timestamp(frozen_split_date))
        test_df = df.iloc[split_pos - TIME_STEPS:].copy()

        # No y_scaler needed for classification
        x_test, y_test_binary = build_sequences(test_df, features, x_scaler, TIME_STEPS)

        # AI now outputs probabilities between 0 and 1
        predicted_probs = model.predict(x_test, verbose=0)

        all_actuals.append(y_test_binary)
        all_predicteds.append(predicted_probs)

        if evaluate_metrics(ticker, y_test_binary, predicted_probs, features, frozen_split_date):
            stocks_beating_baseline += 1

    combined_actuals = np.concatenate(all_actuals)
    combined_predicteds = np.concatenate(all_predicteds)
    evaluate_metrics("COMBINED", combined_actuals, combined_predicteds, features, frozen_split_date, is_combined=True)

    print("\n==================================================")
    print(f" 🏆 SCORE: {stocks_beating_baseline} out of {len(trained_tickers)} stocks beat the noise margin.")
    print("==================================================\n")

if __name__ == "__main__":
    evaluate_and_predict()