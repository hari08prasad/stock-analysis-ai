import numpy as np
import pandas as pd
import joblib
from datetime import date
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from data_loader import fetch_stock_data

def evaluate_and_predict(ticker="RELIANCE.NS", start_date="2021-01-01"):
    print("Loading saved LSTM model and Scaler memory...")
    model = load_model("lstm_stock_model.keras")
    
    # 1. LOAD the exact scaler memory used during training
    scaler = joblib.load("scaler.pkl")

    # 2. Fetch data dynamically up to TODAY
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching data from {start_date} to {today_str}...")
    df = fetch_stock_data(ticker, start_date, today_str)
    df.dropna(inplace=True) 
    
    # 3. Split data: Isolate the last 20% for unseen testing
    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size - 60:] # Keep previous 60 days to build the first test sequence

    # 4. Process test data using the LOADED scaler (transform only, never fit)
    test_data = test_df['Close'].to_numpy().reshape(-1, 1)
    scaled_test = scaler.transform(test_data)

    x_test, y_test = [], []
    for i in range(60, len(scaled_test)):
        x_test.append(scaled_test[i-60:i, 0])
        y_test.append(scaled_test[i, 0])
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 5. Make predictions on the 20% unseen test data
    scaled_predictions = model.predict(x_test, verbose=0)
    
    # Translate decimals back to INR (₹)
    predictions = scaler.inverse_transform(scaled_predictions)
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 6. Calculate Accuracy Metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    latest_actual_close = actuals[-1][0]

    print(f"\n--------------------------------------------------")
    print(f" Stock: {ticker} (Last 20% Unseen Test Data)")
    print(f" Root Mean Squared Error (RMSE): ₹{rmse:.2f}")
    print(f" Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f" Latest Test Predicted Close: ₹{predictions[-1][0]:.2f}")
    print(f" Latest Test Actual Close:    ₹{latest_actual_close:.2f}")
    print(f"--------------------------------------------------")

    # 7. Predict strictly for TOMORROW
    last_60_days = df['Close'].values[-60:]
    scaled_last_60 = scaler.transform(last_60_days.reshape(-1, 1))
    
    # Reshape for the AI: (1 sequence, 60 days, 1 feature)
    X_tomorrow = np.reshape(scaled_last_60, (1, 60, 1))
    
    # Predict and translate
    scaled_tomorrow = model.predict(X_tomorrow, verbose=0)
    tomorrow_price = scaler.inverse_transform(scaled_tomorrow)[0][0]
    
    # Calculate Direction (UP or DOWN)
    price_difference = tomorrow_price - latest_actual_close
    direction = "📈 UP" if price_difference > 0 else "📉 DOWN"
    
    print(f"\n🔮 PREDICTION FOR NEXT TRADING DAY")
    print(f" Forecasted Close Price: ₹{tomorrow_price:.2f}")
    print(f" Forecasted Movement:    {direction} (by ₹{abs(price_difference):.2f})\n")

    return predictions, actuals

if __name__ == "__main__":
    evaluate_and_predict()