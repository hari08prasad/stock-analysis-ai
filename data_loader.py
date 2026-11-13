import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker}...")
    # Download historical market data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Drop any missing values to ensure clean data for the neural network
    stock_data.dropna(inplace=True)
    
    return stock_data

if __name__ == "__main__":
    # Using Reliance Industries (.NS suffix for NSE)
    df = fetch_stock_data("RELIANCE.NS", "2021-01-01", "2024-01-01")
    
    print("\nData successfully loaded!")
    print(df.head())
    print(f"\nTotal records: {len(df)}")