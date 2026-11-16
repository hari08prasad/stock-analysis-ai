# sentiment_loader.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import TICKER, START_DATE
from data_loader import fetch_stock_data
from datetime import date

MODEL_NAME = "ProsusAI/finbert"
CACHE_FILE = "headline_scores_cache.csv"   # avoids re-running FinBERT every time
OUTPUT_FILE = "sentiment_daily.csv"

_tok = None
_model = None


def _load_finbert():
    global _tok, _model
    if _model is None:
        print("Loading FinBERT (first run downloads ~400 MB)...")
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    return _tok, _model


def score_headlines(texts, batch_size=32):
    """Returns a score in [-1, 1] per headline: P(positive) - P(negative)."""
    tok, model = _load_finbert()
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=64, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=1)
        # FinBERT label order: positive, negative, neutral
        scores += (probs[:, 0] - probs[:, 1]).tolist()
        print(f"  scored {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return scores


def load_news_csv(path, keyword=None):
    """Loads a headlines CSV. Auto-detects the date and headline columns."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    date_col = next((cols[c] for c in
                     ["published", "publish_date", "datetime", "date", "time", "timestamp"]
                     if c in cols), None)
    text_col = next((cols[c] for c in
                     ["headline", "title", "headlines", "text", "news"]
                     if c in cols), None)
    if date_col is None or text_col is None:
        raise ValueError(f"Could not find date/headline columns. Found: {list(df.columns)}")

    df = df[[date_col, text_col]].rename(columns={date_col: "published", text_col: "headline"})
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "headline"])

    # Normalize timezone: convert to IST, then drop tz info
    if df["published"].dt.tz is not None:
        df["published"] = df["published"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    if keyword:
        df = df[df["headline"].str.contains(keyword, case=False, na=False)]

    print(f"Loaded {len(df)} headlines "
          f"({df['published'].min().date()} to {df['published'].max().date()})")
    return df.reset_index(drop=True)


def build_daily_sentiment(news_df, trading_dates):
    """
    news_df: columns ['published', 'headline']
    trading_dates: tz-naive DatetimeIndex of NSE trading days
    Returns a DataFrame indexed by trading Date with columns:
      Sentiment       average FinBERT score of the headlines assigned to that day
      Headline_count  number of headlines assigned to that day
    Headlines published after the close (3:30 PM IST) or on non-trading days
    are assigned to the NEXT trading day, so nothing leaks from the future.
    """
    news_df = news_df.copy()

    # Score with caching
    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE).drop_duplicates("headline").set_index("headline")["score"]
    else:
        cache = pd.Series(dtype=float)

    todo = news_df.loc[~news_df["headline"].isin(cache.index), "headline"].drop_duplicates().tolist()
    if todo:
        print(f"Scoring {len(todo)} new headlines...")
        new_scores = pd.Series(score_headlines(todo), index=todo)
        cache = pd.concat([cache, new_scores])
        cache.rename("score").rename_axis("headline").reset_index().to_csv(CACHE_FILE, index=False)

    news_df["score"] = news_df["headline"].map(cache)

    # --- Leakage control: first day the news could have been traded on ---
    pub = news_df["published"]
    date_only = (pub.dt.hour == 0).all() and (pub.dt.minute == 0).all()
    if date_only:
        after_close = pd.Series(True, index=news_df.index)   # no time info: be conservative
    else:
        after_close = (pub.dt.hour > 15) | ((pub.dt.hour == 15) & (pub.dt.minute >= 30))

    news_df["date"] = pub.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")

    # Snap each headline's date forward to the next trading day
    pos = trading_dates.searchsorted(news_df["date"])
    keep = pos < len(trading_dates)
    news_df, pos = news_df[keep].copy(), pos[keep]
    news_df["trading_date"] = trading_dates[pos]

    g = news_df.groupby("trading_date")["score"].agg(["mean", "count"])

    out = pd.DataFrame({
        "Sentiment": g["mean"],
        "Headline_count": g["count"],
    }).reindex(trading_dates)

    out["Sentiment"] = out["Sentiment"].fillna(0.0)             # no news = neutral
    out["Headline_count"] = out["Headline_count"].fillna(0).astype(int)
    out.index.name = "Date"
    return out


if __name__ == "__main__":
    import sys

    # Usage: python sentiment_loader.py reliance_news.csv
    news_path = sys.argv[1] if len(sys.argv) > 1 else "reliance_news.csv"

    today_str = date.today().strftime("%Y-%m-%d")
    prices = fetch_stock_data(TICKER, START_DATE, today_str)
    trading_dates = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()

    news = load_news_csv(news_path, keyword="Reliance")
    sentiment = build_daily_sentiment(news, trading_dates)

    sentiment.to_csv(OUTPUT_FILE)
    print(f"\nSaved {OUTPUT_FILE}")
    print(sentiment.tail(10))
    print(f"\nTrading days: {len(sentiment)}, "
          f"days with news: {(sentiment['Headline_count'] > 0).sum()}")