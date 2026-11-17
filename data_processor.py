import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import StandardScaler

from config import TIME_STEPS, SPLIT_DATE, TICKERS, START_DATE, FEATURE_SET

def build_sequences(df, feature_columns, x_scaler, time_steps=TIME_STEPS):
    """Turns dataframe into LSTM windows. Target is now BINARY (1=UP, 0=DOWN)."""
    x_scaled = x_scaler.transform(df[feature_columns].to_numpy())
    
    # NEW: Convert numerical returns into 1s (Up) and 0s (Down)
    y_binary = (df["Return"] > 0).astype(int).to_numpy()

    x, y = [], []
    for i in range(time_steps, len(df)):
        x.append(x_scaled[i - time_steps:i, :])
        y.append(y_binary[i])
        
    return np.array(x), np.array(y)

def process_data(train_dfs_dict, time_steps=TIME_STEPS, feature_columns=FEATURE_SET):
    """Fits x_scaler globally, builds binary sequences."""
    combined_train_df = pd.concat(list(train_dfs_dict.values()), ignore_index=True)
    x_scaler = StandardScaler().fit(combined_train_df[feature_columns].to_numpy())

    all_x_train = []
    all_y_train = []
    
    for ticker, train_df in train_dfs_dict.items():
        x_train_stock, y_train_stock = build_sequences(
            train_df, feature_columns, x_scaler, time_steps
        )
        all_x_train.append(x_train_stock)
        all_y_train.append(y_train_stock)
        
    final_x_train = np.concatenate(all_x_train, axis=0)
    final_y_train = np.concatenate(all_y_train, axis=0)

    # We only return the x_scaler now
    return final_x_train, final_y_train, x_scaler

if __name__ == "__main__":
    from data_loader import fetch_all
    from feature_indicators import add_indicators

    today_str = date.today().strftime("%Y-%m-%d")
    stock_dict = fetch_all(TICKERS, START_DATE, today_str)
    
    train_dfs_dict = {}
    for ticker, df in stock_dict.items():
        df = add_indicators(df)
        split_pos = df.index.searchsorted(pd.Timestamp(SPLIT_DATE))
        train_dfs_dict[ticker] = df.iloc[:split_pos].copy()

    x_train, y_train, _ = process_data(train_dfs_dict)

    print(f"\nClassification Data Ready. y_train contains: {np.unique(y_train)}")