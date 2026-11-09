"""
Tahap 7: latih LSTM dengan protokol pengukuran yang sama seperti model ML.
Tuning kecil (grid 4 kombinasi) di data validasi -> latih ulang N_REPEATS kali
(seed berbeda) -> evaluasi di data test dalam Rupiah. CPU saja.

Input : data/prepared.npz, data/test.csv, models/scaler_y.joblib, src/config.py
Output: models/LSTM.keras, results/lstm_results.csv, results/predictions_lstm.csv,
        results/best_params_lstm.json, results/all_results.csv (gabungan dengan ML)
"""
import os, sys, json, time, threading, itertools
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # paksa CPU agar adil dengan model ML
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import joblib, psutil
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
import config as C

tf.config.threading.set_intra_op_parallelism_threads(C.N_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(C.N_THREADS)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
proc = psutil.Process()

d = np.load("data/prepared.npz", allow_pickle=True)
Xtr, ytr, Xva, yva, Xte = d["Xw_train"], d["yw_train"], d["Xw_val"], d["yw_val"], d["Xw_test"]
y_test_rp = d["y_test_rupiah"]
sy = joblib.load("models/scaler_y.joblib")
close_today = pd.read_csv("data/test.csv")["Close"].values
FX = C.LSTM_FIXED


def build_model(units, dropout):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=Xtr.shape[1:]),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(FX["learning_rate"]), loss=FX["loss"])
    return m


def fit_model(m):
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=FX["early_stopping_patience"],
                                          restore_best_weights=True)
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=FX["max_epochs"],
              batch_size=FX["batch_size"], callbacks=[es], verbose=0, shuffle=False)
    return h.history


def peak_rss_during(fn):
    base = proc.memory_info().rss
    peak, stop = [base], threading.Event()
    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.01)
    th = threading.Thread(target=poll); th.start()
    t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
    peak[0] = max(peak[0], proc.memory_info().rss); stop.set(); th.join()
    return out, dt, max(peak[0] - base, 0) / 1e6


def to_rp(y):
    return sy.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()


def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "R2": r2_score(y_true, y_pred),
            "Dir_Acc_%": float(np.mean(np.sign(y_pred - close_today) == np.sign(y_true - close_today)) * 100)}


# ---------- 1) Tuning: pilih kombinasi dengan val_loss terbaik ----------
print("=== Tuning LSTM (validasi 203 sampel akhir train) ===")
keys = list(C.LSTM_GRID)
tune_t0 = time.perf_counter()
best_val, best = np.inf, None
for combo in itertools.product(*C.LSTM_GRID.values()):
    p = dict(zip(keys, combo))
    tf.keras.utils.set_random_seed(C.SEED)
    hist = fit_model(build_model(**p))
    v = float(np.min(hist["val_loss"]))
    print(f"  {p} -> val_loss {v:.6f} ({len(hist['loss'])} epoch)")
    if v < best_val:
        best_val, best = v, p
tune_time = time.perf_counter() - tune_t0
print("Hyperparameter terbaik:", best, f"| waktu tuning {tune_time:.0f}s")

# ---------- 2) Latih ulang N_REPEATS kali (seed berbeda) ----------
runs, t_train, ram, epochs, first = [], [], [], [], None
for r in range(C.N_REPEATS):
    tf.keras.utils.set_random_seed(C.SEED + r)
    m = build_model(**best)
    hist, dt, mb = peak_rss_during(lambda: fit_model(m))
    t_train.append(dt); ram.append(mb); epochs.append(len(hist["loss"]))
    pred = to_rp(m.predict(Xte, verbose=0, batch_size=len(Xte)))
    runs.append(metrics(y_test_rp, pred))
    print(f"  ulangan {r + 1}/{C.N_REPEATS}: {dt:.0f}s, {epochs[-1]} epoch, RMSE {runs[-1]['RMSE']:.1f}")
    if r == 0:
        first = (m, pred)
model, y_pred = first          # model dan prediksi ulangan pertama (SEED) disimpan

# ---------- 3) Waktu inference (model ulangan pertama) ----------
model.predict(Xte[:1], verbose=0)                      # pemanasan
t_batch, t_single, t_direct = [], [], []
model(Xte[:1], training=False)                         # pemanasan panggilan langsung
for _ in range(C.N_REPEATS):
    t0 = time.perf_counter(); model.predict(Xte, verbose=0, batch_size=len(Xte)); t_batch.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for i in range(100):
        model.predict(Xte[i:i + 1], verbose=0)
    t_single.append((time.perf_counter() - t0) / 100)
    # panggilan langsung model(x): tanpa overhead pipeline predict() Keras
    t0 = time.perf_counter()
    for i in range(100):
        model(Xte[i:i + 1], training=False)
    t_direct.append((time.perf_counter() - t0) / 100)

# ---------- 4) Simpan ----------
model.save("models/LSTM.keras")
mean = {k: float(np.mean([x[k] for x in runs])) for k in runs[0]}
std = {k + "_std": float(np.std([x[k] for x in runs])) for k in runs[0]}
row = {"Model": "LSTM", **mean, **std,
       "Train_s_mean": np.mean(t_train), "Train_s_std": np.std(t_train),
       "Infer_batch_ms": np.mean(t_batch) * 1e3,
       "Infer_1sample_ms_mean": np.mean(t_single) * 1e3, "Infer_1sample_ms_std": np.std(t_single) * 1e3,
       "Infer_1sample_direct_ms_mean": np.mean(t_direct) * 1e3, "Infer_1sample_direct_ms_std": np.std(t_direct) * 1e3,
       "RAM_train_MB": np.mean(ram), "Model_size_KB": os.path.getsize("models/LSTM.keras") / 1024,
       "Tuning_s": tune_time, "Epochs_mean": np.mean(epochs)}
pd.DataFrame([row]).to_csv("results/lstm_results.csv", index=False)
pd.DataFrame({"Date": d["dates_test"], "Actual": y_test_rp, "LSTM": y_pred}).to_csv("results/predictions_lstm.csv", index=False)
json.dump({"LSTM": best, "epochs_per_run": epochs}, open("results/best_params_lstm.json", "w"), indent=2)

if os.path.exists("results/ml_results.csv"):
    allr = pd.concat([pd.read_csv("results/ml_results.csv"), pd.DataFrame([row])], ignore_index=True)
    allr.to_csv("results/all_results.csv", index=False)
    print("\nTabel gabungan ->", "results/all_results.csv")
    print(allr[["Model", "MAE", "RMSE", "MAPE_%", "R2", "Dir_Acc_%", "Train_s_mean", "Infer_1sample_ms_mean", "Model_size_KB"]].round(3).to_string(index=False))
else:
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})