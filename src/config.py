"""
Tahap 5: konfigurasi eksperimen. Satu-satunya tempat mengatur seed,
jumlah thread, dan hyperparameter. Semua script training mengimpor file ini.
"""
SEED = 42
N_THREADS = 1          # sama untuk SEMUA model agar waktu training/inference adil
WINDOW = 60
N_REPEATS = 5          # jumlah pengulangan pengukuran waktu (Tahap 6)
CV_SPLITS = 5          # TimeSeriesSplit untuk model ML (hanya di data train)

# ---------- Model ML: dituning dengan GridSearchCV + TimeSeriesSplit ----------
ML_GRIDS = {
    "LinearRegression": {},   # baseline, tanpa hyperparameter
    "RandomForest": {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5],
    },
    "XGBoost": {
        "n_estimators": [200, 500],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
    },
    "SVR": {
        "kernel": ["rbf"],
        "C": [1, 10, 100],
        "gamma": ["scale", 0.01, 0.1],
        "epsilon": [0.001, 0.01],
    },
}

# ---------- LSTM: grid kecil, dinilai pada validasi (203 sampel akhir train) ----------
LSTM_GRID = {
    "units": [32, 64],
    "dropout": [0.0, 0.2],
}
LSTM_FIXED = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_patience": 10,   # pantau val_loss, restore best weights
    "optimizer": "adam",
    "loss": "mse",
}