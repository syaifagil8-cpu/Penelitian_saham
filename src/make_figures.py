"""
Gambar untuk laporan: (1) arsitektur dan alur pelatihan LSTM, (2) kurva pelatihan LSTM,
(3) perbandingan lima model. Semua dibuat dari hasil eksperimenmu sendiri.

Jalankan dari folder utama:   python src/make_figures.py
Tanpa melatih ulang LSTM:      python src/make_figures.py --skip-train

Input : results/all_results.csv, results/predictions_lstm.csv, results/best_params_lstm.json,
        data/prepared.npz, data/test.csv, models/scaler_y.joblib, src/config.py
Output: results/fig_lstm_arsitektur, fig_lstm_training, fig_perbandingan_5_model (.png dan .pdf)
        results/lstm_history.csv  (riwayat loss per epoch, dibuat sekali lalu dipakai ulang)
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
import config as C

R = "results"
SKIP_TRAIN = "--skip-train" in sys.argv
ORDER = ["LinearRegression", "SVR", "LSTM", "RandomForest", "XGBoost"]
COL = {"LinearRegression": "#0072B2", "SVR": "#009E73", "LSTM": "#D55E00", "RandomForest": "#CC79A7", "XGBoost": "#E69F00"}
INK, GRAY = "#1B1F2A", "#7F7F7F"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{R}/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  tersimpan: {R}/{name}.png dan .pdf")


def id_num(n):
    return f"{n:,}".replace(",", ".")


# ====================================================================== 1. ARSITEKTUR
def fig_architecture():
    best = {"units": 64, "dropout": 0.2}
    if os.path.exists(f"{R}/best_params_lstm.json"):
        best = json.load(open(f"{R}/best_params_lstm.json"))["LSTM"]
    n_all = n_train = n_val = n_test = None
    if os.path.exists("data/prepared.npz"):
        d = np.load("data/prepared.npz", allow_pickle=True)
        n_train, n_val, n_test = len(d["Xw_train"]), len(d["Xw_val"]), len(d["Xw_test"])
    fx = C.LSTM_FIXED

    fig, ax = plt.subplots(figsize=(15, 7.4))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.4); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, bold=False, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14", fc=fc, ec=ec, lw=1.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.4)

    def arrow(p1, p2, color=INK, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=17, color=color, lw=1.7, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    X = [0.8 + i * 2.8 for i in range(5)]
    W = 2.3
    # ---- baris 1: persiapan data
    y1, h = 5.55, 1.2
    t1 = ["Data harian BBCA\n2.642 hari bursa\n(setelah pembersihan)", "18 fitur teknikal\n+ target Close besok",
          "Pembagian kronologis\n80% latih : 20% uji", "Skala Min-Max\n(fit hanya di data latih)",
          f"Window {C.WINDOW} hari\nbentuk ({C.WINDOW} × 18)\nper sampel"]
    for i, t in enumerate(t1):
        box(X[i], y1, W, h, t, "#ECEEF2", "#3E4656")
        if i:
            arrow((X[i - 1] + W, y1 + h / 2), (X[i], y1 + h / 2))
    # ---- baris 2: arsitektur
    y2 = 3.35
    t2 = [f"Masukan\n({C.WINDOW} hari × 18 fitur)", f"Lapisan LSTM\n{best['units']} unit", f"Dropout\n{best['dropout']}".replace(".", ","),
          "Dense\n1 unit (keluaran)", "Prediksi\nClose besok\n(kembali ke Rp)"]
    for i, t in enumerate(t2):
        box(X[i], y2, W, h, t, "#E1F0FA" if i not in (1,) else "#CFE6F7", "#00507C", bold=(i == 1))
        if i:
            arrow((X[i - 1] + W, y2 + h / 2), (X[i], y2 + h / 2))
    # sambungan baris 1 -> baris 2 (siku)
    xm, xb = X[4] + W / 2, X[0] + W / 2
    ax.plot([xm, xm], [y1, 4.95], color=INK, lw=1.7)
    ax.plot([xm, xb], [4.95, 4.95], color=INK, lw=1.7)
    arrow((xb, 4.95), (xb, y2 + h))
    # ---- baris 3: pelatihan
    y3, h3 = 0.55, 1.55
    n_txt = ""
    if n_train:
        n_txt = f"Sampel: latih {id_num(n_train)} · validasi {id_num(n_val)} (bagian akhir data latih) · uji {id_num(n_test)}\n"
    box(X[0], y3, X[4] + W - X[0], h3,
        f"{n_txt}Optimizer {fx['optimizer'].capitalize()} (learning rate {str(fx['learning_rate']).replace('.', ',')}) · batch {fx['batch_size']} · "
        f"loss {fx['loss'].upper()} · maksimum {fx['max_epochs']} epoch\n"
        f"Early stopping: pantau val_loss, sabar {fx['early_stopping_patience']} epoch, pulihkan bobot terbaik · tanpa shuffle (urutan waktu dijaga)",
        "#FBE6D8", "#8A3A00", fs=10.3)
    arrow((X[4] + W / 2, y2), (X[4] + W / 2, y3 + h3), color="#8A3A00")
    ax.text(X[4] + W / 2 + 0.15, 2.75, "bandingkan dengan\nharga sebenarnya (loss)", fontsize=9.2, color="#8A3A00", va="center")
    arrow((X[1] + W / 2, y3 + h3), (X[1] + W / 2, y2), color="#8A3A00", ls="--")
    ax.text(X[1] + W / 2 + 0.15, 2.75, "perbarui bobot\n(backpropagation)", fontsize=9.2, color="#8A3A00", va="center")
    # label tahap (vertikal di tepi kiri)
    for y, txt in [(y1 + h / 2, "PERSIAPAN DATA"), (y2 + h / 2, "ARSITEKTUR"), (y3 + h3 / 2, "PELATIHAN")]:
        ax.text(0.3, y, txt, rotation=90, ha="center", va="center", fontsize=9, color=GRAY, fontweight="bold")
    ax.set_title("Arsitektur dan alur pelatihan model LSTM", fontsize=14, fontweight="bold", color=INK, pad=6)
    save(fig, "fig_lstm_arsitektur")


# ====================================================================== 2. KURVA PELATIHAN
def get_history():
    path = f"{R}/lstm_history.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    if SKIP_TRAIN:
        return None
    print("  melatih LSTM sekali (seed", C.SEED, ") untuk merekam kurva loss. Butuh sekitar 1 menit...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    import joblib
    tf.config.threading.set_intra_op_parallelism_threads(C.N_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(C.N_THREADS)
    d = np.load("data/prepared.npz", allow_pickle=True)
    best = json.load(open(f"{R}/best_params_lstm.json"))["LSTM"]
    fx = C.LSTM_FIXED
    tf.keras.utils.set_random_seed(C.SEED)
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=d["Xw_train"].shape[1:]), tf.keras.layers.LSTM(best["units"]),
                             tf.keras.layers.Dropout(best["dropout"]), tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(fx["learning_rate"]), loss=fx["loss"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=fx["early_stopping_patience"], restore_best_weights=True)
    h = m.fit(d["Xw_train"], d["yw_train"], validation_data=(d["Xw_val"], d["yw_val"]), epochs=fx["max_epochs"],
              batch_size=fx["batch_size"], callbacks=[es], verbose=0, shuffle=False)
    hist = pd.DataFrame(h.history); hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist.to_csv(path, index=False)
    # pemeriksaan: apakah rekaman ini sama dengan model ulangan pertama yang dipakai di eksperimen?
    sy = joblib.load("models/scaler_y.joblib")
    pred = sy.inverse_transform(m.predict(d["Xw_test"], verbose=0, batch_size=len(d["Xw_test"])).reshape(-1, 1)).ravel()
    rmse_now = float(np.sqrt(np.mean((d["y_test_rupiah"] - pred) ** 2)))
    if os.path.exists(f"{R}/predictions_lstm.csv"):
        p = pd.read_csv(f"{R}/predictions_lstm.csv")
        rmse_saved = float(np.sqrt(np.mean((p["Actual"] - p["LSTM"]) ** 2)))
        status = "SAMA (kurva ini mewakili model ulangan pertama)" if abs(rmse_now - rmse_saved) < 0.5 else "BERBEDA (kurva hanya ilustrasi pelatihan)"
        print(f"  RMSE ulangan ini {rmse_now:.1f} vs RMSE di predictions_lstm.csv {rmse_saved:.1f}: {status}")
    return hist


def fig_training(hist):
    has_pred = os.path.exists(f"{R}/predictions_lstm.csv") and os.path.exists("data/test.csv")
    ncols = (1 if hist is not None else 0) + (1 if has_pred else 0)
    if ncols == 0:
        print("  dilewati: tidak ada riwayat pelatihan maupun prediksi LSTM"); return
    fig, axes = plt.subplots(1, ncols, figsize=(6.8 * ncols, 4.9), squeeze=False)
    axes = list(axes[0]); k = 0
    if hist is not None:
        ax = axes[k]; k += 1
        ax.plot(hist["epoch"], hist["loss"], color=COL["LinearRegression"], lw=2, label="Loss latih")
        ax.plot(hist["epoch"], hist["val_loss"], color=COL["LSTM"], lw=2, label="Loss validasi")
        eb = int(hist["epoch"].iloc[hist["val_loss"].values.argmin()])
        ax.axvline(eb, color=GRAY, ls="--", lw=1.2)
        ax.annotate(f"epoch terbaik: {eb}", (eb, hist["val_loss"].min()), textcoords="offset points", xytext=(10, 22),
                    fontsize=9.5, arrowprops=dict(arrowstyle="-", color=GRAY))
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (data diskalakan 0 sampai 1, skala log)")
        ax.set_title(f"Kurva pelatihan LSTM (berhenti di epoch {int(hist['epoch'].iloc[-1])})", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.25)
    if has_pred:
        ax = axes[k]
        p = pd.read_csv(f"{R}/predictions_lstm.csv"); t = pd.read_csv("data/test.csv")
        x = pd.to_datetime(p["Date"])
        ax.plot(x, p["Actual"], color="black", lw=1.6, label="Aktual")
        ax.plot(x, t["Close"].values, color=GRAY, lw=1, alpha=0.8, label="Baseline naif (= hari ini)")
        ax.plot(x, p["LSTM"], color=COL["LSTM"], lw=1.2, label="Prediksi LSTM")
        rm = float(np.sqrt(np.mean((p["Actual"] - p["LSTM"]) ** 2)))
        ax.set_title(f"Prediksi LSTM pada data uji (RMSE {rm:.1f})", fontweight="bold")
        ax.set_ylabel("Harga Close (Rp)"); ax.legend(fontsize=9); ax.grid(alpha=0.25)
        for lab in ax.get_xticklabels(): lab.set_rotation(20)
    fig.tight_layout()
    save(fig, "fig_lstm_training")


# ====================================================================== 3. PERBANDINGAN 5 MODEL
def fig_compare():
    df = pd.read_csv(f"{R}/all_results.csv").set_index("Model")
    models = [m for m in ORDER if m in df.index]
    naive = df.loc["NaiveBaseline"] if "NaiveBaseline" in df.index else None
    panels = [("RMSE", "RMSE (Rp)\nlebih kecil lebih baik", False, "RMSE_std", None if naive is None else naive["RMSE"], "baseline naif"),
              ("MAE", "MAE (Rp)\nlebih kecil lebih baik", False, "MAE_std", None if naive is None else naive["MAE"], "baseline naif"),
              ("Dir_Acc_%", "Akurasi arah (%)\n50% = tebakan acak", False, "Dir_Acc_%_std", 50, "tebakan acak"),
              ("Train_s_mean", "Waktu training (detik, skala log)\nlebih kecil lebih baik", True, "Train_s_std", None, None),
              ("Infer_1sample_ms_mean", "Inferensi 1 sampel (ms, skala log)\nlebih kecil lebih baik", True, "Infer_1sample_ms_std", None, None),
              ("Model_size_KB", "Ukuran model (KB, skala log)\nlebih kecil lebih baik", True, None, None, None)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, (col, title, log, stdc, ref, reflab) in zip(axes.ravel(), panels):
        vals = df.loc[models, col].astype(float)
        err = df.loc[models, stdc].astype(float).fillna(0).values if stdc and stdc in df.columns else None
        bars = ax.bar(models, vals.values, color=[COL[m] for m in models], yerr=err, capsize=4, ecolor=INK, zorder=3)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        if ref is not None:
            ax.axhline(ref, color=INK, ls="--", lw=1.3, zorder=4, label=f"{reflab} ({ref:g})".replace(".", ",") if col != "Dir_Acc_%" else reflab)
            ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        for b, v in zip(bars, vals.values):
            ax.annotate(f"{v:,.3f}".rstrip("0").rstrip(".").replace(".", ",") if v < 10 else f"{v:,.0f}".replace(",", "."),
                        (b.get_x() + b.get_width() / 2, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.6,
                        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        if col in ("RMSE", "MAE"): ax.set_ylim(0, vals.max() * 1.18)
        if col == "Dir_Acc_%": ax.set_ylim(0, 100)
    fig.suptitle("Perbandingan lima model: akurasi dan biaya komputasi (data uji BBCA, 525 hari)", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, "LSTM: rata-rata ± simpangan baku dari 5 ulangan. Waktu training dan inferensi: rata-rata ± simpangan baku dari 5 ulangan di Docker.",
             ha="center", fontsize=9, color=GRAY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig_perbandingan_5_model")


if __name__ == "__main__":
    os.makedirs(R, exist_ok=True)
    print("1/3 Arsitektur dan alur pelatihan LSTM"); fig_architecture()
    print("2/3 Kurva pelatihan LSTM"); fig_training(get_history())
    print("3/3 Perbandingan lima model"); fig_compare()
    print("Selesai.")