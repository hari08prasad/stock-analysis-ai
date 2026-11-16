import os

# Windows workaround for missing DLL errors when TensorFlow loads (safe to keep)
os.environ["PATH"] = (
    r"C:\Windows\System32" + os.pathsep + os.environ.get("PATH", "")
)

import joblib
import pandas as pd
from datetime import date

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from config import TICKER, START_DATE, TIME_STEPS, SPLIT_DATE, FEATURE_SET, SEED
from data_loader import fetch_stock_data
from feature_indicators import add_indicators, FEATURE_COLUMNS
from data_processor import process_data

# --- FREEZE RANDOMNESS ---
# One call locks Python, NumPy and TensorFlow random numbers together.
tf.keras.utils.set_random_seed(SEED)


def build_lstm_model(input_shape):
    """
    Small LSTM: one layer of 20 memory units, dropout, one output number.
    input_shape = (time_steps, n_features), e.g. (20, 4)
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))   # predicts tomorrow's (scaled) return
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    """
    Trains with a validation split + early stopping:
      - the last 10% of the training windows are held out as a "practice test"
      - training stops when that practice score stops improving
      - the best weights are restored
    This prevents the model from memorising noise as more features are added.
    """
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    return model, history


if __name__ == "__main__":
    # Make sure every requested feature exists in feature_indicators.py
    missing = [f for f in FEATURE_SET if f not in FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"Unknown features in FEATURE_SET: {missing}. "
                         f"Available: {FEATURE_COLUMNS}")

    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching data for {TICKER} from {START_DATE} to {today_str}...")

    # 1. Price data -> features (Return, RSI, SMA_ratio, Volume_ratio), indexed by Date
    df = add_indicators(fetch_stock_data(TICKER, START_DATE, today_str))

    # 2. Training data = everything BEFORE the split date
    split_pos = df.index.searchsorted(pd.Timestamp(SPLIT_DATE))
    train_df = df.iloc[:split_pos].copy()
    print(f"Training strictly on data before {SPLIT_DATE} ({len(train_df)} days)")
    print(f"Features: {FEATURE_SET}")

    # 3. Fit scalers on training data only, build windows
    x_train, y_train, x_scaler, y_scaler = process_data(
        train_df, TIME_STEPS, FEATURE_SET
    )

    # 4. Build and train
    input_shape = (x_train.shape[1], x_train.shape[2])
    print(f"\nBuilding LSTM with input_shape={input_shape}...")
    model = build_lstm_model(input_shape)
    model.summary()

    print("\nStarting training...")
    model, history = train_model(model, x_train, y_train)

    # 5. Save EVERYTHING prediction needs, so the two scripts can never mismatch
    model.save("lstm_stock_model.keras")
    joblib.dump(x_scaler, "x_scaler.pkl")
    joblib.dump(y_scaler, "y_scaler.pkl")
    joblib.dump(SPLIT_DATE, "split_date.pkl")
    joblib.dump(FEATURE_SET, "features.pkl")

    print("\nSaved: lstm_stock_model.keras, x_scaler.pkl, y_scaler.pkl, "
          "split_date.pkl, features.pkl")
    print(f"Timeline frozen at {SPLIT_DATE}. Features: {FEATURE_SET}")