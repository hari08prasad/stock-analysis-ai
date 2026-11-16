import pandas as pd

log = pd.read_csv("results_log.csv")
log["needed_to_beat"] = log["majority"] + log["noise_margin"]

summary = log.groupby("features").agg(
    runs=("seed", "count"),
    rmse=("rmse", "mean"),
    baseline_rmse=("baseline_rmse", "mean"),
    mda=("mda", "mean"),
    mda_std=("mda", "std"),
    majority=("majority", "mean"),
    needed_to_beat=("needed_to_beat", "mean"),
    pct_pred_up=("pct_pred_up", "mean"),
    test_days=("test_days", "mean"),
)

pd.set_option("display.width", 250, "display.max_columns", None)
print(summary.round(4).to_string())