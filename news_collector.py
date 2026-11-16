# news_collector.py
import time
import urllib.parse
from datetime import date, timedelta

import feedparser
import pandas as pd

QUERY = '"Reliance Industries" OR "Reliance Jio" OR "RIL shares"'
START = date(2021, 1, 1)
STEP_DAYS = 7
OUTPUT = "reliance_news.csv"


def fetch_window(start, end):
    q = f"{QUERY} after:{start} before:{end}"
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
           + "&hl=en-IN&gl=IN&ceid=IN:en")
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries:
        if not getattr(e, "published_parsed", None):
            continue
        ts = pd.Timestamp(*e.published_parsed[:6], tz="UTC").tz_convert("Asia/Kolkata")
        rows.append({"published": ts.tz_localize(None), "headline": e.title})
    return rows


if __name__ == "__main__":
    all_rows = []
    cur = START
    today = date.today()
    while cur < today:
        nxt = min(cur + timedelta(days=STEP_DAYS), today + timedelta(days=1))
        try:
            rows = fetch_window(cur, nxt)
        except Exception as ex:
            print(f"{cur}: error {ex}")
            rows = []
        all_rows += rows
        print(f"{cur} -> {nxt}: {len(rows)} headlines (total {len(all_rows)})")

        # save progress regularly so a crash doesn't lose everything
        if len(all_rows) % 200 < len(rows) or nxt >= today:
            pd.DataFrame(all_rows).drop_duplicates("headline").to_csv(OUTPUT, index=False)

        cur = nxt
        time.sleep(1.5)   # be polite, avoid getting blocked

    df = pd.DataFrame(all_rows).drop_duplicates("headline").sort_values("published")
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} headlines to {OUTPUT}")
    print(df.groupby(df["published"].dt.year).size())