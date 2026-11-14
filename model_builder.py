import os
import joblib
from datetime import date

os.environ["PATH"] = (
    r"C:\Windows\System32" + os.pathsep + os.environ.get("PATH", "")
)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_loader import fetch_stock_data
from data_processor import process_data

def build_lstm_model(input_shape):
    """
    Constructs and compiles a stacked LSTM neural network for stock price prediction.
    input_shape: tuple matching (time_steps, features), e.g., (60, 1)
    """
    model = Sequential()

    # Layer 1: First LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Layer 2: Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Layer 3: Standard Dense hidden layer
    model.add(Dense(units=25))

    # Layer 4: Output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(model, x_train, y_train, epochs=25, batch_size=32):
    """
    Trains the LSTM model on the prepared 3D sequences.
    """
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    return model, history

if __name__ == "__main__":
    # 1. Fetch and process the NSE data up to TODAY
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching data from 2021-01-01 to {today_str}...")
    
    df = fetch_stock_data("RELIANCE.NS", "2021-01-01", today_str)
    x_train, y_train, scaler = process_data(df, time_steps=60)

    # 2. Build the model using (60, 1) from x_train.shape[1:]
    input_shape = (x_train.shape[1], x_train.shape[2])
    print(f"\nBuilding LSTM model with input_shape={input_shape}...")
    model = build_lstm_model(input_shape)
    model.summary()

    # 3. Train the model
    print("\nStarting training...")
    model, history = train_model(model, x_train, y_train, epochs=25, batch_size=32)

    # 4. Save the trained brain AND the scaler memory
    model.save("lstm_stock_model.keras")
    joblib.dump(scaler, "scaler.pkl")
    
    print("\nModel saved successfully as 'lstm_stock_model.keras'!")
    print("Scaler memory saved successfully as 'scaler.pkl'!")