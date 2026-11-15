import yfinance as yf
import pandas as pd
from datetime import date

# Import master variables for the local testing block
from config import TICKER, START_DATE

def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker}...")
    # Download historical market data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Drop any missing values to ensure clean data for the neural network
    stock_data.dropna(inplace=True)
    
    return stock_data

if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    
    # Using master config variables instead of hardcoded strings
    df = fetch_stock_data(TICKER, START_DATE, today_str)
    
    print("\nData successfully loaded!")
    print(df.head())
    print(f"\nTotal records: {len(df)}")