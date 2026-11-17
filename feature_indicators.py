import os
import numpy as np
import pandas as pd

# The columns the model will use as input.
FEATURE_COLUMNS = [
    "Return",        # daily % change: what happened today
    "RSI",           # momentum: is the stock stretched up or down?
    "SMA_ratio",     # trend: where is price vs its recent average?
    "Volume_ratio",  # activity: is today busier than normal?
    "Sentiment",     # news tone from FinBERT
]

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_sma_ratio(close, window=20):
    sma = close.rolling(window).mean()
    return close / sma - 1

def compute_volume_ratio(volume, window=20):
    volume = volume.replace(0, np.nan)
    return np.log(volume / volume.rolling(window).mean())

def add_sentiment(df, ticker, use_sentiment_flag):
    """
    Safely joins sentiment data. It ensures we don't accidentally give
    Reliance's news to TCS. If sentiment is toggled off, or the file 
    is missing for this specific ticker, it defaults to 0.0 (Neutral).
    """
    df = df.copy()
    
    # We expect a unique sentiment file per ticker, e.g., RELIANCE.NS_sentiment.csv
    # For now, since your partner only built Reliance, we hardcode that specific file.
    if ticker == "RELIANCE.NS" and use_sentiment_flag:
        path = "sentiment_daily.csv"
    else:
        path = f"{ticker}_sentiment.csv" # Placeholder for when they build the rest

    if use_sentiment_flag and os.path.exists(path):
        sentiment = pd.read_csv(path, index_col="Date", parse_dates=True)[["Sentiment"]]
        sentiment.index = pd.to_datetime(sentiment.index).tz_localize(None).normalize()
        df = df.join(sentiment, how="left")
        df["Sentiment"] = df["Sentiment"].fillna(0.0)
    else:
        # If toggled off or file doesn't exist, safely pad with zeros
        df["Sentiment"] = 0.0
        
    return df

def add_indicators(df, ticker="RELIANCE.NS"):
    """
    Takes the raw dataframe from fetch_stock_data and returns Close + FEATURES.
    Now requires the 'ticker' to prevent news cross-contamination.
    """
    # Import here to avoid circular logic
    from config import USE_SENTIMENT
    
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"

    df["Return"] = df["Close"].pct_change()
    df["RSI"] = compute_rsi(df["Close"])
    df["SMA_ratio"] = compute_sma_ratio(df["Close"])
    df["Volume_ratio"] = compute_volume_ratio(df["Volume"])

    # NEW: Pass the ticker to the sentiment function
    df = add_sentiment(df, ticker, USE_SENTIMENT)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    return df[["Close"] + FEATURE_COLUMNS]

if __name__ == "__main__":
    from datetime import date
    from data_loader import fetch_stock_data
    from config import START_DATE

    test_ticker = "RELIANCE.NS"
    raw = fetch_stock_data(test_ticker, START_DATE, date.today().strftime("%Y-%m-%d"))
    feats = add_indicators(raw, ticker=test_ticker)

    print(feats.tail())
    print("Days with non-zero sentiment:", int((feats["Sentiment"] != 0).sum()), "of", len(feats))