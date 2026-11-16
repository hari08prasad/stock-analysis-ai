import pandas as pd

# 1. Read the log that predict.py has been appending to
log = pd.read_csv("results_log.csv")

# 2. Group the rows by feature set (e.g. "Return", "Return+RSI", ...)
#    and calculate the mean, spread and number of runs for each metric
summary = log.groupby("features")[["rmse", "mda", "pct_pred_up"]].agg(["mean", "std", "count"])

# 3. Print it neatly
print(summary.round(6))