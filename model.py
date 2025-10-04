import os
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.keras import metrics
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Directory to save models
SAVE_DIR = "models"
n_steps = 3
n_features = 1

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X),array(y)

def compute(var):
    # ==== training dataSet normalization =====
    train_norm = np.asarray(x_train[var]).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler.fit_transform(train_norm)
    train_norm = train_norm[train_norm != 0]

    #  ==== testing dataSet normalization =====
    test_norm = np.asarray(x_test[var]).reshape(-1, 1)
    test_norm = scaler.transform(test_norm)
    test_norm = test_norm[test_norm != 0]

    # ===== Slice the sequence =====
    X_split_train, y_split_train = split_sequence(train_norm, n_steps)
    X_split_train = X_split_train.reshape((X_split_train.shape[0], X_split_train.shape[1], n_features))
    X_split_test,  y_split_test  = split_sequence(test_norm,  n_steps)
    X_split_test  = X_split_test.reshape((X_split_test.shape[0],  X_split_test.shape[1],  n_features))

    # ===== build the model =====
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[metrics.MeanAbsoluteError(name='mae'),
                           metrics.RootMeanSquaredError(name='rmse')])

    # ===== training =====
    hist = model.fit(X_split_train, y_split_train,
                     validation_data=(X_split_test, y_split_test),
                     epochs=10, verbose=1)

    # ===== evalution =====
    yhat = model.predict(X_split_test)
    mse = mean_squared_error(y_split_test, yhat)
    print(f"{var} → Test MSE: {mse:.6f}")

    # ===== save the file =====
    base   = f"{var}_lstm"
    model_path  = os.path.join(SAVE_DIR, f"{base}.keras")   # Keras v3 建議用 .keras
    scaler_path = os.path.join(SAVE_DIR, f"{base}.scaler")
    config_path = os.path.join(SAVE_DIR, f"{base}.json")
    hist_path   = os.path.join(SAVE_DIR, f"{base}_history.npy")

    model.save(model_path)                 
    joblib.dump(scaler, scaler_path)       
    np.save(hist_path, hist.history)       

    config = {
        "var": var,
        "n_steps": n_steps,
        "n_features": n_features,
        "train_len": int(len(train_norm)),
        "test_len": int(len(test_norm)),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"已儲存：\n  model   -> {model_path}\n  scaler  -> {scaler_path}\n  config  -> {config_path}\n  history -> {hist_path}")

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "config_path": config_path,
        "history_path": hist_path,
    }

if __name__ == "__main__":
    # create directory if not exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # load data
    df = pd.read_csv('data/data_221_2022-2025/locationid221_2022-2025.csv')
    df = df.dropna()
    print("df.shape: ", df.shape)

    # split into train and test sets
    train_size = int(len(df) * 0.8)
    x_train = df[:train_size]
    x_test = df[train_size:]
    print(f"total data amount: {len(df)}")
    print(f"training data amount: {len(x_train)}")
    print(f"testing data amont: {len(x_test)}")

    vars_to_train = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    artifacts = {}
    for v in vars_to_train:
        artifacts[v] = compute(v)