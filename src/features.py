"""
Feature engineering + split train/test (80:20 berurutan, tanpa shuffle).
Indikator dihitung manual dengan pandas/numpy (tanpa library tambahan)
supaya environment Docker tetap ringan dan hasilnya transparan.

Input : data/dataset_BBCA_clean.csv
Output: data/train.csv, data/test.csv (BELUM di-scale), data/features.json
Scaling dilakukan di script training, fit HANYA pada data train.
"""
import json
import numpy as np
import pandas as pd

CLEAN = "data/dataset_BBCA_clean.csv"
TRAIN_RATIO = 0.8

df = pd.read_csv(CLEAN, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
c = df["Close"]

# --- Moving average & EMA ---
for w in (5, 10, 20):
    df[f"MA_{w}"] = c.rolling(w).mean()
df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
df["EMA_26"] = c.ewm(span=26, adjust=False).mean()

# --- RSI 14 (Wilder smoothing) ---
delta = c.diff()
gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
df["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

# --- MACD (12, 26, 9) ---
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

# --- Bollinger Bands (20, 2) ---
ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
df["BB_upper"] = ma20 + 2 * sd20
df["BB_lower"] = ma20 - 2 * sd20
df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / ma20

# --- Return harian ---
df["Return_1d"] = c.pct_change()

# --- Target: Close hari berikutnya ---
df["target"] = c.shift(-1)

# Baris awal (indikator belum terbentuk) dan baris terakhir (tanpa target) dibuang
df = df.dropna().reset_index(drop=True)

FEATURES = ["Open", "High", "Low", "Close", "Volume",
            "MA_5", "MA_10", "MA_20", "EMA_12", "EMA_26", "RSI_14",
            "MACD", "MACD_signal", "MACD_hist",
            "BB_upper", "BB_lower", "BB_width", "Return_1d"]

# --- Split kronologis, tanpa shuffle ---
split = int(len(df) * TRAIN_RATIO)
train, test = df.iloc[:split], df.iloc[split:]

cols = ["Date"] + FEATURES + ["target"]
train[cols].to_csv("data/train.csv", index=False)
test[cols].to_csv("data/test.csv", index=False)
json.dump({"features": FEATURES, "target": "target"}, open("data/features.json", "w"), indent=2)

# --- Ringkasan & sanity check ---
print(f"Total baris : {len(df)}")
print(f"Train       : {len(train)}  ({train.Date.min().date()} s/d {train.Date.max().date()})")
print(f"Test        : {len(test)}  ({test.Date.min().date()} s/d {test.Date.max().date()})")
assert train.Date.max() < test.Date.min(), "Train dan test tumpang tindih!"
print(f"Jumlah fitur: {len(FEATURES)}")
print(f"Close train  : {train.Close.min():.0f} - {train.Close.max():.0f}")
print(f"Close test   : {test.Close.min():.0f} - {test.Close.max():.0f}")

# Baseline naif: prediksi besok = harga hari ini
rmse = np.sqrt(((test["target"] - test["Close"]) ** 2).mean())
mae = (test["target"] - test["Close"]).abs().mean()
print(f"Baseline naif (test): MAE={mae:.1f}  RMSE={rmse:.1f}")