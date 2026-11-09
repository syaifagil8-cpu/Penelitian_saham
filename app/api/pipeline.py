"""Perhitungan fitur untuk aplikasi. HARUS identik dengan src/features.py
(diuji: hasilnya sama persis dengan data/train.csv dan data/test.csv)."""
import json
import numpy as np
import pandas as pd


def load_feature_list(path):
    return json.load(open(path))["features"]


def load_clean(path):
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def compute_features(df):
    out = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    c = out["Close"]
    for w in (5, 10, 20):
        out[f"MA_{w}"] = c.rolling(w).mean()
    out["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = c.ewm(span=26, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["BB_upper"] = ma20 + 2 * sd20
    out["BB_lower"] = ma20 - 2 * sd20
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / ma20
    out["Return_1d"] = c.pct_change()
    return out