import pandas as pd
from datetime import date

# ==========================================
# MASTER CONFIGURATION FILE
# ==========================================

# Step 1: 10 Liquid NSE Large Caps for cross-asset model
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
    "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "LT.NS", 
    "BHARTIARTL.NS", "HINDUNILVR.NS"
]
START_DATE = "2021-01-01"

# The lookback window (20 days = ~1 trading month of history per sample)
TIME_STEPS = 5

# Random seed: same seed + same data + same features = repeatable results.
SEED = 7

# ------------------------------------------
# TRAIN / TEST SPLIT DATE
# ------------------------------------------
FIXED_SPLIT_DATE = "2025-11-16"

if FIXED_SPLIT_DATE:
    SPLIT_DATE = FIXED_SPLIT_DATE
else:
    today = pd.to_datetime(date.today())
    SPLIT_DATE = (today - pd.DateOffset(months=12)).strftime("%Y-%m-%d")

# ------------------------------------------
# FEATURE SET & DATA TOGGLES
# ------------------------------------------
# Turn the sentiment toggle ON
USE_SENTIMENT = True

# Add "Sentiment" to the end of the feature list
FEATURE_SET = ["Return", "RSI", "SMA_ratio", "Volume_ratio", "Sentiment"]