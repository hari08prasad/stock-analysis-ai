import os
os.environ["PATH"] = (r"C:\Windows\System32" + os.pathsep + os.environ.get("PATH", ""))

import joblib
import pandas as pd
from datetime import date
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from config import TICKERS, START_DATE, TIME_STEPS, SPLIT_DATE, FEATURE_SET, SEED
from data_loader import fetch_all
from feature_indicators import add_indicators, FEATURE_COLUMNS
from data_processor import process_data

tf.keras.utils.set_random_seed(SEED)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=20, return_sequences=False))
    model.add(Dropout(0.2))
    # Sigmoid activation outputs a probability (0.0 to 1.0)
    model.add(Dense(units=1, activation='sigmoid'))   
    
    # Binary Crossentropy punishes it for guessing the wrong category
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss="binary_crossentropy", 
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    # Calculate how heavily the AI should be penalized for missing a DOWN day (0)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}
    
    print(f"Applying Class Weights: 0 (DOWN) = {weights[0]:.2f}, 1 (UP) = {weights[1]:.2f}")

    history = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.1, 
        callbacks=[early_stop], 
        class_weight=class_weight_dict,
        verbose=1
    )
    return model, history

if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    stock_dict = fetch_all(TICKERS, START_DATE, today_str)

    train_dfs_dict = {}
    
    # Add features and chop off test data for EACH stock individually
    for ticker, df in stock_dict.items():
        df = add_indicators(df, ticker=ticker) 
        
        split_pos = df.index.searchsorted(pd.Timestamp(SPLIT_DATE))
        train_df = df.iloc[:split_pos].copy()
        
        # Saves the chopped dataframe so it can be combined later
        train_dfs_dict[ticker] = train_df 
        
    # Process without y_scaler
    x_train, y_train, x_scaler = process_data(train_dfs_dict, TIME_STEPS, FEATURE_SET)

    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_lstm_model(input_shape)
    model, history = train_model(model, x_train, y_train)

    # Save logic (No y_scaler)
    model.save("lstm_stock_model.keras")
    joblib.dump(x_scaler, "x_scaler.pkl")
    joblib.dump(SPLIT_DATE, "split_date.pkl")
    joblib.dump(FEATURE_SET, "features.pkl")
    joblib.dump(TICKERS, "tickers.pkl")

    print("\nSaved Classification Model and Scaler.")