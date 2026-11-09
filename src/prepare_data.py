"""
Tahap 4: scaling bebas leakage + persiapan input model.

Input : data/train.csv, data/test.csv, data/features.json
Output: data/prepared.npz       (array siap latih)
        models/scaler_X.joblib  (scaler fitur, di-fit HANYA di train)
        models/scaler_y.joblib  (scaler target, di-fit HANYA di train)
"""
import json, os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

WINDOW = 60          # LSTM melihat 60 hari terakhir
VAL_RATIO = 0.10     # 10% akhir data train (berurutan) untuk early stopping LSTM

os.makedirs("models", exist_ok=True)
cfg = json.load(open("data/features.json"))
FEATURES, TARGET = cfg["features"], cfg["target"]

train = pd.read_csv("data/train.csv", parse_dates=["Date"])
test = pd.read_csv("data/test.csv", parse_dates=["Date"])

# --- 1. Scaling: fit HANYA di train, lalu transform train dan test ---
sx, sy = MinMaxScaler(), MinMaxScaler()
X_train = sx.fit_transform(train[FEATURES])
X_test = sx.transform(test[FEATURES])
y_train = sy.fit_transform(train[[TARGET]]).ravel()
y_test = sy.transform(test[[TARGET]]).ravel()
joblib.dump(sx, "models/scaler_X.joblib")
joblib.dump(sy, "models/scaler_y.joblib")

# --- 2. Window untuk LSTM: (sampel, 60 hari, 18 fitur) ---
def make_windows(X, y, w):
    Xw = np.stack([X[i - w + 1: i + 1] for i in range(w - 1, len(X))])
    return Xw, y[w - 1:]

Xw_train_all, yw_train_all = make_windows(X_train, y_train, WINDOW)

# Test: tambahkan 59 hari terakhir train sebagai konteks awal window
X_ctx = np.vstack([X_train[-(WINDOW - 1):], X_test])
y_ctx = np.concatenate([np.zeros(WINDOW - 1), y_test])  # nilai dummy, dibuang oleh make_windows
Xw_test, yw_test = make_windows(X_ctx, y_ctx, WINDOW)

# --- 3. Validasi LSTM: potongan AKHIR dari train (berurutan, tidak diacak) ---
n_val = int(len(Xw_train_all) * VAL_RATIO)
Xw_train, yw_train = Xw_train_all[:-n_val], yw_train_all[:-n_val]
Xw_val, yw_val = Xw_train_all[-n_val:], yw_train_all[-n_val:]

# --- 4. Pemeriksaan keselarasan ---
assert Xw_test.shape[0] == len(test), "Jumlah window test harus sama dengan jumlah baris test"
assert np.allclose(yw_test, y_test), "Target LSTM dan model ML harus identik"
assert X_train.min() >= 0 and X_train.max() <= 1, "Train harus 0..1 (di-fit di train)"

np.savez("data/prepared.npz",
         X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
         Xw_train=Xw_train, yw_train=yw_train, Xw_val=Xw_val, yw_val=yw_val,
         Xw_test=Xw_test, yw_test=yw_test,
         y_test_rupiah=test[TARGET].values, dates_test=test["Date"].dt.strftime("%Y-%m-%d").values)

print(f"ML    : X_train {X_train.shape} | X_test {X_test.shape}")
print(f"LSTM  : train {Xw_train.shape} | val {Xw_val.shape} | test {Xw_test.shape}")
print(f"Fitur di test di luar rentang train (nilai <0 atau >1): "
      f"{((X_test < 0) | (X_test > 1)).any(axis=1).mean():.1%} baris")
print(f"Target test di luar rentang train: {((y_test < 0) | (y_test > 1)).mean():.1%} baris")