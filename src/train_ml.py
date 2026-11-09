"""
Tahap 6: latih 4 model ML (Linear Regression, Random Forest, XGBoost, SVR).
Tuning di data train (TimeSeriesSplit) -> latih ulang berulang kali untuk ukur waktu
-> evaluasi di data test dalam Rupiah.

Input : data/prepared.npz, data/test.csv, models/scaler_y.joblib, src/config.py
Output: models/<Model>.joblib, results/ml_results.csv,
        results/predictions_ml.csv, results/best_params_ml.json
"""
import json, os, sys, time, threading
import numpy as np
import pandas as pd
import joblib, psutil
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR

sys.path.insert(0, os.path.dirname(__file__))
import config as C

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
proc = psutil.Process()

d = np.load("data/prepared.npz", allow_pickle=True)
X_train, y_train, X_test = d["X_train"], d["y_train"], d["X_test"]
y_test_rp = d["y_test_rupiah"]
sy = joblib.load("models/scaler_y.joblib")
test = pd.read_csv("data/test.csv")
close_today = test["Close"].values          # untuk akurasi arah


def make_models():
    m = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=C.SEED, n_jobs=C.N_THREADS),
        "SVR": SVR(),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(random_state=C.SEED, n_jobs=C.N_THREADS,
                                    tree_method="hist", verbosity=0)
    except ImportError:
        print("PERINGATAN: xgboost belum terpasang, XGBoost dilewati.")
    return m


def peak_rss_during(fn):
    """Jalankan fn(), kembalikan (hasil, waktu_detik, kenaikan_puncak_RSS_MB)."""
    base = proc.memory_info().rss
    peak, stop = [base], threading.Event()
    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)
    th = threading.Thread(target=poll); th.start()
    t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set(); th.join()
    return out, dt, max(peak[0] - base, 0) / 1e6


def to_rp(y_scaled):
    return sy.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": r2_score(y_true, y_pred),
        "Dir_Acc_%": float(np.mean(np.sign(y_pred - close_today) == np.sign(y_true - close_today)) * 100),
    }


rows, preds, best = [], {"Date": d["dates_test"], "Actual": y_test_rp}, {}

# Baseline naif: besok = hari ini
naive = metrics(y_test_rp, close_today); naive["Dir_Acc_%"] = np.nan   # arah tidak terdefinisi untuk baseline naif
rows.append({"Model": "NaiveBaseline", **naive})

for name, model in make_models().items():
    print(f"\n=== {name} ===")
    grid = C.ML_GRIDS[name]

    # 1) Tuning (hanya data train, validasi silang berurutan)
    tune_time = 0.0
    if grid:
        gs = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=C.CV_SPLITS),
                          scoring="neg_root_mean_squared_error", n_jobs=1, refit=False)
        t0 = time.perf_counter(); gs.fit(X_train, y_train); tune_time = time.perf_counter() - t0
        params = gs.best_params_
    else:
        params = {}
    best[name] = params
    print("Hyperparameter terbaik:", params, f"| waktu tuning {tune_time:.1f}s")
    final = clone(model).set_params(**params)

    # 2) Waktu training & RAM: ulang N_REPEATS kali
    t_train, ram = [], []
    for r in range(C.N_REPEATS):
        m = clone(final)
        _, dt, mb = peak_rss_during(lambda: m.fit(X_train, y_train))
        t_train.append(dt); ram.append(mb)
    # m = model hasil fit terakhir (seed tetap -> hasil identik tiap ulangan)

    # 3) Waktu inference: pemanasan sekali, lalu rata-rata
    m.predict(X_test[:1])
    t_batch, t_single = [], []
    for r in range(C.N_REPEATS):
        t0 = time.perf_counter(); m.predict(X_test); t_batch.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for i in range(100): m.predict(X_test[i:i + 1])
        t_single.append((time.perf_counter() - t0) / 100)

    # 4) Evaluasi di data test (Rupiah)
    y_pred = to_rp(m.predict(X_test))
    preds[name] = y_pred
    path = f"models/{name}.joblib"
    joblib.dump(m, path)

    rows.append({"Model": name, **metrics(y_test_rp, y_pred),
                 "Train_s_mean": np.mean(t_train), "Train_s_std": np.std(t_train),
                 "Infer_batch_ms": np.mean(t_batch) * 1e3,
                 "Infer_1sample_ms_mean": np.mean(t_single) * 1e3,
                 "Infer_1sample_ms_std": np.std(t_single) * 1e3,
                 "RAM_train_MB": np.mean(ram),
                 "Model_size_KB": os.path.getsize(path) / 1024,
                 "Tuning_s": tune_time})
    print({k: round(v, 3) for k, v in rows[-1].items() if k != "Model"})

res = pd.DataFrame(rows)
res.to_csv("results/ml_results.csv", index=False)
pd.DataFrame(preds).to_csv("results/predictions_ml.csv", index=False)
json.dump(best, open("results/best_params_ml.json", "w"), indent=2)
print("\n", res[["Model", "MAE", "RMSE", "MAPE_%", "R2", "Dir_Acc_%", "Train_s_mean", "Infer_1sample_ms_mean"]].round(3).to_string(index=False))