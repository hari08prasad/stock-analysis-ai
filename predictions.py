import numpy as np
import pandas as pd
import joblib
from datetime import date
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from data_loader import fetch_stock_data

# --- NEW: Import variables from master config ---
from config import TIME_STEPS, TICKER, START_DATE, SPLIT_DATE

def evaluate_and_predict(ticker=TICKER, start_date=START_DATE):
    print("Loading saved LSTM model and Scaler memory...")
    model = load_model("lstm_stock_model.keras")
    
    # 1. LOAD the exact scaler memory used during training
    scaler = joblib.load("scaler.pkl")

    # 2. Fetch data dynamically up to TODAY
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching data from {start_date} to {today_str}...")
    df = fetch_stock_data(ticker, start_date, today_str)
    
    # Calculate Returns (Market Momentum)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True) 
    
    # --- THE EXPERT FIX: FIXED DATE SPLIT ---
    # 3. Find the exact row index of the split date
    split_pos = df.index.searchsorted(SPLIT_DATE)
    test_df = df.iloc[split_pos - TIME_STEPS:].copy() 

    # 4. Process ONLY the test data using the LOADED scaler on RETURNS
    test_data = test_df['Return'].to_numpy().reshape(-1, 1)
    scaled_test = scaler.transform(test_data)

    x_test, y_test = [], []
    for i in range(TIME_STEPS, len(scaled_test)):
        x_test.append(scaled_test[i-TIME_STEPS:i, 0])
        y_test.append(scaled_test[i, 0])
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 5. Make predictions on the unseen test data
    scaled_predictions = model.predict(x_test, verbose=0)
    
    # Translate decimals back to actual Percentage Returns
    predicted_returns = scaler.inverse_transform(scaled_predictions)
    actual_returns = scaler.inverse_transform(y_test.reshape(-1, 1))

    # --- 6. CALCULATE METRICS & BASELINE ON RETURNS ---
    rmse = np.sqrt(mean_squared_error(actual_returns, predicted_returns))
    
    # Naive Baseline for Returns: Assuming tomorrow's return will be exactly 0.0
    baseline_predictions = np.zeros_like(actual_returns)
    baseline_rmse = np.sqrt(mean_squared_error(actual_returns, baseline_predictions))
    
    # --- THE EXPERT FIX: MAJORITY BASELINE & NOISE MARGIN ---
    actual_direction = np.sign(actual_returns)
    predicted_direction = np.sign(predicted_returns)
    
    mda = np.mean(actual_direction == predicted_direction) * 100
    majority = max(np.mean(actual_direction > 0), np.mean(actual_direction < 0)) * 100

    print(f"\n--------------------------------------------------")
    print(f" Stock: {ticker} (Test Data since {SPLIT_DATE})")
    print(f" AI Model Error (RMSE):      {rmse:.4f}")
    print(f" Naive Baseline Error:       {baseline_rmse:.4f}")
    print(f"--------------------------------------------------")

    # Evaluate Magnitude Edge
    if rmse < baseline_rmse:
        print(f" ✅ EDGE: AI beats the Baseline in exact magnitude.")
    else:
        print(f" ⚠️ LAGGING: AI loses to the Baseline in exact magnitude.")
        
    # Evaluate Directional Edge using Expert Logic
    print(f" Directional Accuracy (MDA): {mda:.2f}%  (always-guess-majority: {majority:.2f}%)")
    if mda > majority + 4:
        print(" ✅ Direction beats the majority baseline by more than the noise margin.")
    else:
        print(" ⚠️ Direction is not distinguishable from guessing.")
    print(f"--------------------------------------------------")

    # 7. Predict strictly for TOMORROW
    last_X_days = df['Return'].values[-TIME_STEPS:]
    scaled_last_X = scaler.transform(last_X_days.reshape(-1, 1))
    
    # Reshape for the AI: (1 sequence, TIME_STEPS, 1 feature)
    X_tomorrow = np.reshape(scaled_last_X, (1, TIME_STEPS, 1))
    
    # Predict and translate the return
    scaled_tomorrow = model.predict(X_tomorrow, verbose=0)
    predicted_tomorrow_return = scaler.inverse_transform(scaled_tomorrow)[0][0]
    
    # Convert predicted return back to an actual Rupee price
    latest_actual_close = float(df['Close'].values[-1][0])
    tomorrow_price = latest_actual_close * (1 + predicted_tomorrow_return)
    
    # Calculate Direction (UP or DOWN)
    price_difference = tomorrow_price - latest_actual_close
    direction = "📈 UP" if price_difference > 0 else "📉 DOWN"
    
    print(f"\n🔮 PREDICTION FOR NEXT TRADING DAY")
    print(f" Forecasted Return:      {predicted_tomorrow_return * 100:.2f}%")
    print(f" Forecasted Close Price: ₹{tomorrow_price:.2f}")
    print(f" Forecasted Movement:    {direction} (by ₹{abs(price_difference):.2f})\n")

    return predicted_returns, actual_returns

if __name__ == "__main__":
    evaluate_and_predict()