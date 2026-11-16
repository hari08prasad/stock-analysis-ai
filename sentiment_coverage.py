import pandas as pd
from config import SPLIT_DATE

s = pd.read_csv("sentiment_daily.csv", index_col="Date", parse_dates=True)
has = s["Headline_count"] > 0

train, test = has[s.index < SPLIT_DATE], has[s.index >= SPLIT_DATE]
print(f"Train (before {SPLIT_DATE}): {train.mean()*100:.1f}% of {len(train)} days have headlines")
print(f"Test  (from {SPLIT_DATE}):   {test.mean()*100:.1f}% of {len(test)} days have headlines")

print("\nBy year (share of days with news, number of days):")
print(has.groupby(has.index.year).agg(["mean", "count"]).round(3))