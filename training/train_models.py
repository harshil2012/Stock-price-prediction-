import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from training.data_processor import (
    download_data, clean_data, add_features,
    preprocess_for_lstm, preprocess_for_sklearn
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def build_and_train_lstm(
        X_train,
        y_train,
        X_test,
        y_test,
        sequence_length,
        prediction_days,
        num_features):
    print("Training LSTM model...")
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=prediction_days)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    return model


def build_and_train_rf(X_train, y_train):
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_and_train_lr(X_train, y_train):
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_pred, model_name):
    # Flatten arrays for multi-output regression (predicting 5 days)
    y_t_flat = y_true.flatten()
    y_p_flat = y_pred.flatten()

    rmse = root_mean_squared_error(y_t_flat, y_p_flat)
    mae = mean_absolute_error(y_t_flat, y_p_flat)
    r2 = r2_score(y_t_flat, y_p_flat)

    # Simulated accuracy score based on directional correctness
    # Did the prediction go up or down matching the truth?
    # This is rough and simple
    diff_true = np.diff(y_t_flat)
    diff_pred = np.diff(y_p_flat)

    # Avoid division by zero warnings by replacing 0 with small eps if needed,
    # or just sign
    sign_true = np.sign(diff_true)
    sign_pred = np.sign(diff_pred)

    directional_accuracy = np.mean(
        sign_true == sign_pred) * 100 if len(sign_true) > 0 else 0

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "directional_accuracy": float(directional_accuracy)
    }

    print(f"--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    return metrics


def train_pipeline(ticker="AAPL", sequence_length=60, prediction_days=5):
    print(f"Starting training pipeline for {ticker}")
    df = download_data(ticker, period="5y")
    if df is None:
        raise ValueError("Failed to download data.")

    df = clean_data(df)
    df = add_features(df)

    # 1. Train LSTM
    (X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm,
     scaler_feat, scaler_targ, lstm_features) = preprocess_for_lstm(
        df, sequence_length, prediction_days)

    lstm_model = build_and_train_lstm(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
        sequence_length, prediction_days, len(lstm_features)
    )
    lstm_model.save(os.path.join(MODELS_DIR, f"{ticker}_lstm.h5"))
    joblib.dump(
        scaler_feat,
        os.path.join(
            MODELS_DIR,
            f"{ticker}_scaler_feat.joblib"))
    joblib.dump(
        scaler_targ,
        os.path.join(
            MODELS_DIR,
            f"{ticker}_scaler_targ.joblib"))

    # Evaluate LSTM
    y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
    # Inverse transform
    y_pred_lstm = scaler_targ.inverse_transform(y_pred_lstm_scaled)
    y_test_unscaled = scaler_targ.inverse_transform(y_test_lstm)
    lstm_metrics = evaluate_model(y_test_unscaled, y_pred_lstm, "LSTM")

    # 2. Train Sklearn Models
    (X_train_sk, X_test_sk, y_train_sk, y_test_sk,
     scaler_X_sk, sk_features) = preprocess_for_sklearn(
        df, prediction_days)

    # RF
    rf_model = build_and_train_rf(X_train_sk, y_train_sk)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, f"{ticker}_rf.joblib"))

    # Evaluate RF
    y_pred_rf = rf_model.predict(X_test_sk)
    rf_metrics = evaluate_model(y_test_sk, y_pred_rf, "Random Forest")

    # LR
    lr_model = build_and_train_lr(X_train_sk, y_train_sk)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, f"{ticker}_lr.joblib"))

    # Evaluate LR
    y_pred_lr = lr_model.predict(X_test_sk)
    lr_metrics = evaluate_model(y_test_sk, y_pred_lr, "Linear Regression")

    joblib.dump(
        scaler_X_sk,
        os.path.join(
            MODELS_DIR,
            f"{ticker}_scaler_X_sk.joblib"))

    print("Training finished successfully.")

    return {
        "lstm": lstm_metrics,
        "rf": rf_metrics,
        "lr": lr_metrics
    }


if __name__ == "__main__":
    train_pipeline("AAPL")
