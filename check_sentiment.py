import pandas as pd

s = pd.read_csv("sentiment_daily.csv", index_col="Date", parse_dates=True)["Sentiment"]
print(s[s != 0].describe())
print("Positive days:", (s > 0).sum(), "| Negative days:", (s < 0).sum())
print(s[s != 0].groupby(s[s != 0].index.year).agg(["count", "mean"]))

cache = pd.read_csv("headline_scores_cache.csv")
pd.set_option("display.max_colwidth", 90, "display.width", 200)
print("\nMost negative:\n", cache.nsmallest(5, "score").to_string(index=False))
print("\nMost positive:\n", cache.nlargest(5, "score").to_string(index=False))
print("\nRandom sample:\n", cache.sample(8, random_state=1).to_string(index=False))