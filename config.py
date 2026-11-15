import pandas as pd
from datetime import date

# ==========================================
# MASTER CONFIGURATION FILE
# ==========================================

TICKER = "RELIANCE.NS"
START_DATE = "2021-01-01"

# Automatically draw the line exactly 4 months ago from today.
# Everything before this date = Training Data
# The last 4 months = Testing Data (Unseen by the scaler and the AI during training)
today = pd.to_datetime(date.today())
SPLIT_DATE = (today - pd.DateOffset(months=4)).strftime("%Y-%m-%d")

# The lookback window (20 days = ~1 trading month of momentum)
TIME_STEPS = 20