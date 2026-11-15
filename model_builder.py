import os
import random
import numpy as np
import tensorflow as tf
import joblib
from datetime import date

# --- NEW: Import master configuration ---
from config import TICKER, START_DATE, TIME_STEPS, SPLIT_DATE

# --- FREEZE RANDOMNESS ---
# 1. Lock Python's hash randomizer
os.environ['PYTHONHASHSEED'] = '42'
# 2. Lock standard Python random numbers
random.seed(42)
# 3. Lock Numpy's random math operations
np.random.seed(42)
# 4. Lock TensorFlow's neural network weight initialization
tf.random.set_seed(42)
# -------------------------

os.environ["PATH"] = (
    r"C:\Windows\System32" + os.pathsep + os.environ.get("PATH", "")
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_loader import fetch_stock_data
from data_processor import process_data

def build_lstm_model(input_shape):
    """
    Constructs a streamlined LSTM to prevent overfitting on return noise.
    """
    model = Sequential()

    # Layer 1: Single LSTM layer (simplified to 20 neurons)
    model.add(LSTM(units=20, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Layer 2: Direct output layer
    model.add(Dense(units=1))

    # Compile the model with a controlled learning rate
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

def train_model(model, x_train, y_train, epochs=25, batch_size=32):
    """
    Trains the LSTM model on the prepared sequences.
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
    # 1. Fetch the NSE data up to TODAY
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching data for {TICKER} from {START_DATE} to {today_str}...")
    
    df = fetch_stock_data(TICKER, START_DATE, today_str)

    # Calculate Returns (Market Momentum)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    # --- THE EXPERT FIX: FIXED DATE SPLIT ---
    # Find the exact row index of your dynamic 4-months-ago date
    split_pos = df.index.searchsorted(SPLIT_DATE)
    
    # Slice off only the data from before 4 months ago for training
    train_df = df.iloc[:split_pos].copy()
    
    print(f"Training strictly on data before {SPLIT_DATE} ({len(train_df)} days)...")

    # 2. Process ONLY the isolated training data
    x_train, y_train, scaler = process_data(train_df, time_steps=TIME_STEPS)

    # 3. Build the streamlined model
    input_shape = (x_train.shape[1], x_train.shape[2])
    print(f"\nBuilding simplified LSTM model with input_shape={input_shape}...")
    model = build_lstm_model(input_shape)
    model.summary()

    # 4. Train the model
    print("\nStarting training...")
    model, history = train_model(model, x_train, y_train, epochs=25, batch_size=32)

    # 5. Save the trained brain AND the safely isolated scaler memory
    model.save("lstm_stock_model.keras")
    joblib.dump(scaler, "scaler.pkl")
    
    print(f"\nModel saved successfully as 'lstm_stock_model.keras'!")
    print(f"Scaler memory saved successfully as 'scaler.pkl'! AI is safely isolated at {SPLIT_DATE}.")