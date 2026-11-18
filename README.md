# Predicting NSE Stock Direction with LSTM, Technical Indicators and News Sentiment

A BCA project that asks one question: **can an LSTM neural network, given recent price history, technical indicators and news sentiment, predict whether an NSE large-cap stock will close up or down the next trading day?**

We built the pipeline in stages and tested every version on a frozen, never-seen 12-month window, judging results against a majority-guess baseline and a statistical noise margin.

> **Full write-up:** [`report/Stock_Prediction_Project_Report.pdf`](report/Stock_Prediction_Project_Report.pdf) (9 pages: how the project developed, all results, limitations, lessons).

## Result in one paragraph

**No model reliably beat guessing.** On Reliance alone, directional accuracy rose from 45.9% (price only) to 54.1% (three indicators), but this stayed inside the noise margin and did not repeat on 10 stocks (47.7% pooled, below the 52.2% majority guess). News sentiment (FinBERT on Google News headlines) added no measurable improvement. Switching from regression to up/down classification stopped the model predicting "UP" every day, but pooled accuracy stayed at 48.9 to 51.9%, which is what a no-skill model with the same UP rate would score. The value of the project is its method: leakage-free scaling, a frozen out-of-sample window, baselines, noise margins, repeated seeds and an experiment log.

## How results are judged

| Item | Setting |
|---|---|
| Stocks | RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, ITC, SBIN, LT, BHARTIARTL, HINDUNILVR |
| Training data | 2021-01-01 to 2025-11-15 |
| Test data | 2025-11-16 onward (227 days per stock, 2,270 pooled) |
| Baseline | Always guess the more common direction in the test year |
| Noise margin | 98 / sqrt(n) percentage points (about ±6.5 per stock, ±2.1 pooled; the pooled value is a lower bound because stocks move together) |
| A result "wins" only if | accuracy > majority guess + noise margin |

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | Tickers, dates, window length, seed, frozen split date, feature list, sentiment toggle |
| `data_loader.py` | Downloads prices from Yahoo Finance and caches one CSV per stock |
| `feature_indicators.py` | Return, RSI, SMA ratio, volume ratio; joins daily sentiment by date (Reliance only) |
| `data_processor.py` | Builds 3D LSTM windows per stock; scaler fitted on training rows only; binary up/down target |
| `model_builder.py` | Trains the LSTM (sigmoid output, binary cross-entropy, class weights, early stopping) and saves the model, scaler, split date, features and tickers |
| `predictions.py` | Evaluates per stock and pooled against the baseline and noise margin; appends each run to `results_log_class.csv` |
| `news_collector.py` | Collects Reliance headlines from Google News RSS |
| `sentiment_loader.py` | Scores headlines with FinBERT and builds `sentiment_daily.csv` |
| `summarize_results.py` | Averages a results log by feature set |
| `sentiment_coverage.py`, `check_sentiment.py`, `check_merge.py` | Optional sanity checks on the sentiment data |
| `results_log.csv` | Reliance regression experiments (6 feature sets x 3 seeds) |
| `results_log_multi.csv` | 10-stock regression experiments (4 feature sets x 3 seeds) |
| `results_log_class.csv` | 10-stock classification experiments |
| `report/` | The project report (PDF) |

The two regression logs come from earlier versions of the code, which you can find in the git history. The current code is the classification pipeline.

## Setup and run

```bash
pip install -r requirements.txt
```

1. Open `config.py` and set `FEATURE_SET`, `SEED`, `TIME_STEPS` and `USE_SENTIMENT`.
2. Train the model:
   ```bash
   python model_builder.py
   ```
   This creates `lstm_stock_model.keras`, `x_scaler.pkl`, `split_date.pkl`, `features.pkl` and `tickers.pkl`. These are ignored by git and regenerated on every run.
3. Evaluate it:
   ```bash
   python predictions.py
   ```
   Always run this straight after training, without changing `config.py` in between. It prints per-stock and pooled results and appends them to `results_log_class.csv`.

The first run downloads prices and saves them as `<TICKER>_cache.csv` (ignored by git). Delete those files to download fresh prices; exact numbers depend on the download date.

### Rebuilding the news data (optional, slow)

The repository already contains `sentiment_daily.csv`, so you do not need this to run experiments.

```bash
python news_collector.py                       # several minutes; overwrites reliance_news.csv
python sentiment_loader.py reliance_news.csv   # downloads FinBERT (about 400 MB) on first run
```

Google News does not return identical headlines each time, so re-collecting changes the sentiment data and the results that depend on it. Back up the existing CSVs first.

## Known limitations

- Sentiment data exists for **Reliance only**; the other nine stocks receive 0.
- The classification results are **single runs (seed 7)**, and the sentiment comparison has not been run as a clean with/without test over several seeds.
- One test year, in which only 47.5% of days were UP.
- Early stopping holds out the last 10% of the stacked training array, which is roughly the last stock.
- The logs do not record the window length or class weights of each run.

Details are in sections 5.3 and 7 of the report.

## Disclaimer

This is an educational project. Nothing here is investment advice, and the model should not be used to trade.
