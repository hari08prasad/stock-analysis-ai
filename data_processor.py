import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from data_loader import fetch_stock_data

# --- NEW: Import master configuration ---
from config import TIME_STEPS, SPLIT_DATE, TICKER, START_DATE

def process_data(train_df, time_steps=TIME_STEPS):
    # 1. Isolate the 'Return' column instead of raw prices
    # This prevents the AI from breaking when the stock hits new all-time highs
    data = train_df['Return'].to_numpy().reshape(-1, 1)
    
    # 2. Scale the data
    # Because train_df strictly ends 4 months ago, the scaler memory is safely isolated from test data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Build the sequences
    x_train, y_train = [], []
    for i in range(time_steps, len(scaled_data)):
        x_train.append(scaled_data[i-time_steps:i, 0])
        y_train.append(scaled_data[i, 0])
        
    # 4. Convert and reshape for the LSTM (samples, time_steps, features)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    df = fetch_stock_data(TICKER, START_DATE, today_str)
    
    # Calculate returns while the timeline is fully connected
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    
    # --- THE EXPERT FIX: FIXED DATE SPLIT ---
    # Find the exact row index of your dynamic 4-months-ago date
    split_pos = df.index.searchsorted(SPLIT_DATE)
    
    # Slice off only the data from before 4 months ago for training
    train_df = df.iloc[:split_pos].copy()
    
    # Process only the training data
    x_train, y_train, scaler = process_data(train_df, time_steps=TIME_STEPS)
    
    print(f"X_train shape (3D): {x_train.shape}")
    print(f"Data Processor successfully built isolated return sequences using data before {SPLIT_DATE}!")