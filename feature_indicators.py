import os

import numpy as np
import pandas as pd

# Where sentiment_loader.py saves its output (must be in the project folder)
SENTIMENT_FILE = "sentiment_daily.csv"

# The columns the model will use as input.
# Return is the base feature (and also the prediction target, shifted 1 day).
FEATURE_COLUMNS = [
    "Return",        # daily % change: what happened today
    "RSI",           # momentum: is the stock stretched up or down?
    "SMA_ratio",     # trend: where is price vs its recent average?
    "Volume_ratio",  # activity: is today busier than normal?
    "Sentiment",     # news tone from FinBERT (sentiment_loader.py)
]


def compute_rsi(close, period=14):
    """
    RSI = Relative Strength Index. A number from 0 to 100.

    Idea: compare the average size of recent UP days with recent DOWN days.
      - Above ~70: mostly up days lately ("overbought", may cool off)
      - Below ~30: mostly down days lately ("oversold", may bounce)
      - Around 50: balanced

    Why it can help the model: it summarises the last 14 days of momentum
    in one number, and highlights "stretched" conditions where prices
    sometimes reverse.
    """
    delta = close.diff()                 # today's price change
    gain = delta.clip(lower=0)           # keep only the up moves
    loss = -delta.clip(upper=0)          # keep only the down moves (as positives)

    # Smoothed averages of gains and losses over `period` days
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss             # ratio of average gain to average loss
    return 100 - (100 / (1 + rs))        # squeeze into the 0-100 range


def compute_sma_ratio(close, window=20):
    """
    SMA ratio = how far today's price is from its 20-day average.

    Idea: the 20-day Simple Moving Average (SMA) is the "recent normal" price.
      - Positive value (e.g. +0.04): price is 4% ABOVE its average (uptrend)
      - Negative value: price is BELOW its average (downtrend)
      - Near 0: price is close to its average

    Why it can help the model: it tells the model about trend and about how
    far price has drifted from normal. We use a ratio instead of the raw
    average so the value stays in a stable range even when the stock
    reaches new highs (the same reason we switched to returns).
    """
    sma = close.rolling(window).mean()   # average of the last 20 closes
    return close / sma - 1               # % distance from that average


def compute_volume_ratio(volume, window=20):
    """
    Volume ratio = today's trading volume compared with its 20-day average.
    We take the log so that "twice as busy" and "half as busy" are
    equal-sized opposites (+0.69 and -0.69).

      - Positive: busier than usual (a big move today is more meaningful)
      - Negative: quieter than usual (a move today may be weak/noisy)

    Why it can help the model: it's the only feature that is NOT calculated
    from price. It adds a different kind of information, so it is the most
    likely of the three to give the model something new.
    """
    volume = volume.replace(0, np.nan)   # avoid divide-by-zero / log(0)
    return np.log(volume / volume.rolling(window).mean())


def add_sentiment(df, path=SENTIMENT_FILE):
    """
    Joins the daily sentiment score onto the price table by Date (the master key).
    Days with no news get 0.0 (neutral).

    If the sentiment file is missing, Sentiment is set to 0 for every day so
    the indicator-only experiments still run.
    """
    if os.path.exists(path):
        sentiment = pd.read_csv(path, index_col="Date", parse_dates=True)[["Sentiment"]]
        sentiment.index = pd.to_datetime(sentiment.index).tz_localize(None).normalize()
        df = df.join(sentiment, how="left")
        df["Sentiment"] = df["Sentiment"].fillna(0.0)
    else:
        print(f"WARNING: {path} not found. Sentiment set to 0 for all days.")
        df["Sentiment"] = 0.0
    return df


def add_indicators(df):
    """
    Takes the raw dataframe from fetch_stock_data (needs Close and Volume)
    and returns a dataframe with Close + FEATURE_COLUMNS, indexed by Date.
    """
    df = df.copy()

    # yfinance sometimes returns two-level column names; flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standard date index: YYYY-MM-DD, no timezone, named "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"

    # Calculate each feature
    df["Return"] = df["Close"].pct_change()
    df["RSI"] = compute_rsi(df["Close"])
    df["SMA_ratio"] = compute_sma_ratio(df["Close"])
    df["Volume_ratio"] = compute_volume_ratio(df["Volume"])

    # Merge the news sentiment (joined on the Date index)
    df = add_sentiment(df)

    # The first ~20 rows are NaN while the averages "warm up"; drop them
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)

    # Keep Close so a forecast can be converted back to a rupee price
    return df[["Close"] + FEATURE_COLUMNS]


if __name__ == "__main__":
    from datetime import date
    from data_loader import fetch_stock_data
    from config import TICKER, START_DATE

    raw = fetch_stock_data(TICKER, START_DATE, date.today().strftime("%Y-%m-%d"))
    feats = add_indicators(raw)

    print(feats.tail())
    print("\nShape:", feats.shape)
    print("Index name:", feats.index.name, "| tz:", feats.index.tz)
    print("Any NaN left?", feats.isna().any().any())
    print("Days with non-zero sentiment:", int((feats["Sentiment"] != 0).sum()),
          "of", len(feats))