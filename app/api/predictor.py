"""Logika prediksi: memuat model SEKALI saat startup, lalu melayani prediksi.
Tidak ada training di sini (dataset dan model terkunci)."""
import json, os, sys, threading, time
import numpy as np
import pandas as pd
import joblib
from .pipeline import compute_features, load_clean, load_feature_list

sys.path.insert(0, "src")
try:
    import config as _cfg
    WINDOW = _cfg.WINDOW
except Exception:
    WINDOW = 60

ML_MODELS = ["LinearRegression", "SVR", "RandomForest", "XGBoost"]
FIGURES = ["fig_actual_vs_pred.png", "fig_tradeoff.png", "fig_error_by_range.png"]


def _records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))   # NaN -> null


class Predictor:
    def __init__(self, base="."):
        self.base = base
        p = lambda *a: os.path.join(base, *a)
        self.features = load_feature_list(p("data", "features.json"))
        self.clean = load_clean(p("data", "dataset_BBCA_clean.csv"))
        valid = compute_features(self.clean).dropna(subset=self.features)
        self.valid = valid.reset_index().rename(columns={"index": "clean_pos"})
        self.sx = joblib.load(p("models", "scaler_X.joblib"))
        self.sy = joblib.load(p("models", "scaler_y.joblib"))
        self.scaled = self.sx.transform(self.valid[self.features])
        self.dates = self.valid["Date"].values
        self.train = pd.read_csv(p("data", "train.csv"), usecols=["Date"], parse_dates=["Date"])
        self.test = pd.read_csv(p("data", "test.csv"), usecols=["Date", "Close"], parse_dates=["Date"])
        self.train_end = self.train["Date"].max()
        self.lock = threading.Lock()
        self.models, self.unavailable = {}, {}
        for name in ML_MODELS:
            path = p("models", f"{name}.joblib")
            try:
                self.models[name] = joblib.load(path)
            except Exception as e:
                self.unavailable[name] = f"{type(e).__name__}: {e}"[:200]
        try:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            self.models["LSTM"] = tf.keras.models.load_model(p("models", "LSTM.keras"))
        except Exception as e:
            self.unavailable["LSTM"] = f"{type(e).__name__}: {e}"[:200]
        self._load_predictions()

    # ---- prediksi historis di data uji (untuk rentang kesalahan dan backtest) ----
    def _load_predictions(self):
        p = lambda f: os.path.join(self.base, "results", f)
        self.pred_test, self.rel_q = None, {}
        try:
            pr = pd.read_csv(p("predictions_ml.csv"))
            if os.path.exists(p("predictions_lstm.csv")):
                pr = pr.merge(pd.read_csv(p("predictions_lstm.csv"))[["Date", "LSTM"]], on="Date", how="left")
            self.pred_test = pr
            for m in self.models:
                if m in pr.columns and pr[m].notna().all():
                    rel = (pr["Actual"] - pr[m]) / pr[m]
                    self.rel_q[m] = (float(rel.quantile(0.10)), float(rel.quantile(0.90)))
        except Exception:
            pass

    def available_models(self):
        return list(self.models)

    def date_range(self):
        pos_min = WINDOW - 1
        return str(pd.Timestamp(self.dates[pos_min]).date()), str(pd.Timestamp(self.dates[-1]).date())

    def _locate(self, as_of):
        pos_min = WINDOW - 1
        if as_of is None:
            return len(self.dates) - 1
        t = np.datetime64(pd.Timestamp(as_of))
        pos = int(np.searchsorted(self.dates, t, side="right")) - 1
        if pos < pos_min:
            raise ValueError(f"Tanggal terlalu awal. Mulai dari {self.date_range()[0]}.")
        return pos

    def predict(self, model, as_of=None):
        if model not in self.models:
            raise KeyError(model)
        pos = self._locate(as_of)
        row = self.valid.iloc[pos]
        as_of_date = pd.Timestamp(row["Date"])
        close_today = float(row["Close"])
        with self.lock:
            if model == "LSTM":
                x = self.scaled[pos - WINDOW + 1: pos + 1][None, :, :]
                t0 = time.perf_counter()
                ps = float(np.asarray(self.models[model].predict(x, verbose=0, batch_size=1)).ravel()[0])
            else:
                x = self.scaled[pos: pos + 1]
                t0 = time.perf_counter()
                ps = float(self.models[model].predict(x)[0])
            infer_ms = (time.perf_counter() - t0) * 1e3
        pred = float(self.sy.inverse_transform([[ps]])[0, 0])
        cp = int(row["clean_pos"])
        actual = actual_date = None
        if cp + 1 < len(self.clean):
            actual = float(self.clean["Close"].iloc[cp + 1])
            actual_date = str(self.clean["Date"].iloc[cp + 1].date())
        q = self.rel_q.get(model)
        return {"model": model, "as_of_date": str(as_of_date.date()), "close_today": close_today,
                "predicted_close": round(pred, 1),
                "predicted_change_pct": round((pred / close_today - 1) * 100, 3),
                "baseline_close": close_today,
                "interval80": None if q is None else [round(pred * (1 + q[0]), 1), round(pred * (1 + q[1]), 1)],
                "actual_next_close": actual, "actual_next_date": actual_date,
                "error": None if actual is None else round(pred - actual, 1),
                "in_training_data": bool(as_of_date <= self.train_end),
                "inference_ms": round(infer_ms, 3)}

    def predict_all(self, as_of=None):
        return [self.predict(m, as_of) for m in self.models]

    def history(self, n=90, end=None):
        df = self.clean
        if end is not None:
            df = df[df["Date"] <= pd.Timestamp(end)]
        d = df.tail(n)[["Date", "Close"]].copy()
        d["Date"] = d["Date"].dt.strftime("%Y-%m-%d")
        return _records(d)

    def backtest(self, model):
        if self.pred_test is None or model not in self.pred_test.columns:
            raise KeyError(model)
        d = self.pred_test[["Date", "Actual", model]].rename(columns={model: "Pred"})
        d["Naive"] = self.test["Close"].values
        return _records(d)

    def _csv(self, name):
        f = os.path.join(self.base, "results", name)
        return _records(pd.read_csv(f)) if os.path.exists(f) else None

    def metrics(self):
        return {"accuracy": self._csv("table_accuracy.csv"), "compute": self._csv("table_compute.csv"),
                "significance": self._csv("table_significance.csv"), "by_range": self._csv("table_by_price_range.csv")}

    def results(self):
        """Hasil eksperimen dalam angka (bukan teks berformat), untuk grafik di tampilan."""
        return self._csv("all_results.csv")

    def environment(self):
        f = os.path.join(self.base, "results", "environment.json")
        return {"environment": json.load(open(f)) if os.path.exists(f) else None,
                "stage_times": self._csv("stage_times.csv")}

    def dataset_info(self):
        lo, hi = self.date_range()
        return {"rows_clean": len(self.clean), "period": [str(self.clean["Date"].min().date()), str(self.clean["Date"].max().date())],
                "n_train": len(self.train), "n_test": len(self.test),
                "train_end": str(self.train_end.date()), "test_start": str(self.test["Date"].min().date()),
                "features": self.features, "window_lstm": WINDOW, "as_of_min": lo, "as_of_max": hi,
                "models_available": self.available_models(), "models_unavailable": self.unavailable}