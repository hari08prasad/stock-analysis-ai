import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import StandardScaler

from config import TIME_STEPS, SPLIT_DATE, TICKER, START_DATE, FEATURE_SET


def build_sequences(df, feature_columns, x_scaler, y_scaler, time_steps=TIME_STEPS):
    """
    Turns a dataframe into LSTM-ready windows using ALREADY-FITTED scalers.
    Used for both training and testing, so the two can never drift apart.

    Each sample:
      x = the last `time_steps` days of ALL chosen features   (time_steps, n_features)
      y = the NEXT day's return (scaled)                      (single number)
    """
    # transform only: the scalers were fitted on training data
    x_scaled = x_scaler.transform(df[feature_columns].to_numpy())
    y_scaled = y_scaler.transform(df[["Return"]].to_numpy())

    x, y = [], []
    for i in range(time_steps, len(df)):
        x.append(x_scaled[i - time_steps:i, :])   # past days, every feature
        y.append(y_scaled[i, 0])                  # the day we want to predict
    return np.array(x), np.array(y)


def process_data(train_df, time_steps=TIME_STEPS, feature_columns=FEATURE_SET):
    """
    Fits the scalers on TRAINING data only, then builds training windows.

    Two scalers:
      x_scaler -> scales the input features (all chosen columns)
      y_scaler -> scales only the target (Return), so predictions can be
                  converted back to real % returns later
    """
    x_scaler = StandardScaler().fit(train_df[feature_columns].to_numpy())
    y_scaler = StandardScaler().fit(train_df[["Return"]].to_numpy())

    x_train, y_train = build_sequences(
        train_df, feature_columns, x_scaler, y_scaler, time_steps
    )
    return x_train, y_train, x_scaler, y_scaler


if __name__ == "__main__":
    from data_loader import fetch_stock_data
    from feature_indicators import add_indicators

    today_str = date.today().strftime("%Y-%m-%d")
    df = add_indicators(fetch_stock_data(TICKER, START_DATE, today_str))

    split_pos = df.index.searchsorted(pd.Timestamp(SPLIT_DATE))
    train_df = df.iloc[:split_pos].copy()

    x_train, y_train, _, _ = process_data(train_df)

    print(f"Features used: {FEATURE_SET}")
    print(f"X_train shape (samples, time_steps, features): {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Built from data before {SPLIT_DATE}.")