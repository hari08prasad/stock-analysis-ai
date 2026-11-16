import pandas as pd
from datetime import date

# ==========================================
# MASTER CONFIGURATION FILE
# ==========================================

TICKER = "RELIANCE.NS"
START_DATE = "2021-01-01"

# The lookback window (20 days = ~1 trading month of history per sample)
TIME_STEPS = 20

# Random seed: same seed + same data + same features = repeatable results.
# To check that a result is not just luck, change this (42, 7, 123...) and re-run.
SEED = 7

# ------------------------------------------
# TRAIN / TEST SPLIT DATE
# ------------------------------------------
# Everything BEFORE this date = training data.
# Everything FROM this date on = test data (never seen by the scaler or the model).
#
# For fair experiments, FREEZE the date so every run uses the same test period.
# Set FIXED_SPLIT_DATE = None to go back to "exactly 12 months ago from today".
FIXED_SPLIT_DATE = "2025-11-16"

if FIXED_SPLIT_DATE:
    SPLIT_DATE = FIXED_SPLIT_DATE
else:
    today = pd.to_datetime(date.today())
    SPLIT_DATE = (today - pd.DateOffset(months=12)).strftime("%Y-%m-%d")

# ------------------------------------------
# FEATURE SET  <-- THE ONLY LINE YOU CHANGE FOR EACH EXPERIMENT
# ------------------------------------------
# Available: "Return", "RSI", "SMA_ratio", "Volume_ratio"
# (later you can add "Sentiment" once your partner's file is merged)
#
# Experiment order (change ONE thing at a time, retrain, evaluate, log):
#   Step 1: ["Return"]                                       <- baseline
#   Step 2: ["Return", "RSI"]
#   Step 3: ["Return", "RSI", "SMA_ratio"]
#   Step 4: ["Return", "RSI", "SMA_ratio", "Volume_ratio"]
FEATURE_SET = ["Return"]