import yfinance as yf
import pandas as pd
from datetime import date
import os

from config import TICKERS, START_DATE

def fetch_stock_data(ticker, start_date, end_date):
    # Download historical market data
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Flatten two-level column names
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Drop any missing values to ensure clean data
    stock_data.dropna(inplace=True)

    return stock_data

def fetch_all(tickers, start_date, end_date):
    """
    Fetches data for all tickers and caches them as CSVs to avoid 
    re-downloading and getting IP-blocked by yfinance.
    Returns a dictionary of dataframes {ticker: dataframe}.
    """
    all_data = {}
    
    for ticker in tickers:
        cache_file = f"{ticker}_cache.csv"
        
        # Check if we already downloaded this stock recently
        if os.path.exists(cache_file):
            print(f"Loading {ticker} from local cache...")
            # Read from CSV and restore the Date index
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            print(f"Downloading {ticker} from Yahoo Finance...")
            df = fetch_stock_data(ticker, start_date, end_date)
            df.to_csv(cache_file)
            
        all_data[ticker] = df
        
    return all_data

if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    
    print(f"Fetching data for {len(TICKERS)} stocks...")
    
    # Load all stocks into a dictionary
    stock_dict = fetch_all(TICKERS, START_DATE, today_str)

    print("\nAll data successfully loaded and cached!")
    print(f"Total Reliance records: {len(stock_dict['RELIANCE.NS'])}")
    print(f"Total TCS records: {len(stock_dict['TCS.NS'])}")