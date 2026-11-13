import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import fetch_stock_data

def process_data(df, time_steps=60):
  # 1. Isolate the target column and force it into a 2D mathematical shape
    data = df['Close'].to_numpy().reshape(-1, 1)
    
    # 2. Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Build the sequences
    x_train, y_train = [], []
    for i in range(time_steps, len(scaled_data)):
        x_train.append(scaled_data[i-time_steps:i, 0])
        y_train.append(scaled_data[i, 0])
        
    # 4. Convert and reshape
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

if __name__ == "__main__":
    df = fetch_stock_data("RELIANCE.NS", "2021-01-01", "2024-01-01")
    x_train, y_train, scaler = process_data(df, time_steps=60)
    print(f"X_train shape (3D): {x_train.shape}")