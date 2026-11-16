from datetime import date
from data_loader import fetch_stock_data
from feature_indicators import add_indicators
from config import TICKER, START_DATE

raw = fetch_stock_data(TICKER, START_DATE, date.today().strftime("%Y-%m-%d"))
df = add_indicators(raw)

df.to_csv("merged_features.csv")
print(df.tail(10))
print("\nColumns:", list(df.columns))
print("Days with sentiment:", int((df["Sentiment"] != 0).sum()), "of", len(df))