import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def download_data(ticker, period="5y"):
    """Downloads historical stock data from Yahoo Finance."""
    print(f"Downloading data for {ticker}...")
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # In newer versions of yfinance, the columns might be multi-index. Let's flatten if needed.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            # Rename columns to standard names by removing the trailing ticker part
            data.rename(columns=lambda x: x.split('_')[0], inplace=True)
            
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def clean_data(df):
    """Cleans data by handling missing values."""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    return df

def add_features(df):
    """Adds necessary features for predictions."""
    # We will predict 'Close' based on 'Open', 'High', 'Low', 'Close', 'Volume'
    # Adding a 'Previous_Close' feature
    df['Previous_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    return df

def preprocess_for_lstm(df, sequence_length=60, prediction_days=5):
    """
    Prepares data for LSTM model (sequence to sequence/point).
    We use sequence_length days to predict the next prediction_days days.
    """
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    data = df[features].values
    
    scaler_feature = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_feature.fit_transform(data)
    
    # We need a separate scaler for 'Close' (target column) so we can inverse transform predictions later
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit_transform(df[['Close']].values)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_days + 1):
        X.append(scaled_data[i-sequence_length:i])
        # predict the next 'prediction_days' Close prices
        # Close is the last index (4) in the features list
        y.append(scaled_data[i:i+prediction_days, 4])
        
    X, y = np.array(X), np.array(y)
    
    # Train-test split (80% train, 20% test)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler_feature, scaler_target, features

def preprocess_for_sklearn(df, prediction_days=5):
    """
    Prepares data for Random Forest / Linear Regression models.
    """
    features = ['Open', 'High', 'Low', 'Volume', 'Previous_Close']
    
    data = df.copy()
    
    # We want to predict the next 5 days. For simple sklearn models, we might predict them one by one
    # or use multi-output regression.
    # Let's create target columns: Close_T+1, Close_T+2, ..., Close_T+5
    for i in range(1, prediction_days + 1):
        data[f'Target_{i}'] = data['Close'].shift(-i)
        
    data.dropna(inplace=True)
    
    X = data[features].values
    y = data[[f'Target_{i}' for i in range(1, prediction_days + 1)]].values
    
    # Scale X
    scaler_X = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler_X.fit_transform(X)
    
    # Scale y (optional for RF/LR, but good practice, though we can skip to keep it simple)
    # Let's skip scaling y to make interpretation easier.
    
    split = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler_X, features

if __name__ == "__main__":
    # Test the script locally
    df = download_data("AAPL", "1y")
    if df is not None:
        df = clean_data(df)
        df = add_features(df)
        print("Data shape after cleaning and features:", df.shape)
        X_train, X_test, y_train, y_test, _, _, _ = preprocess_for_lstm(df)
        print("LSTM X_train shape:", X_train.shape)
        print("LSTM y_train shape:", y_train.shape)
