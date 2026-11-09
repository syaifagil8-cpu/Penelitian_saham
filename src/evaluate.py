"""
Tahap 8: evaluasi akhir dan analisis trade-off.

Input : results/all_results.csv, results/predictions_ml.csv, results/predictions_lstm.csv,
        data/train.csv, data/test.csv
Output: results/table_accuracy.csv, results/table_compute.csv, results/table_significance.csv,
        results/table_pairwise_pvalues.csv, results/table_by_price_range.csv,
        results/fig_actual_vs_pred.png, results/fig_tradeoff.png, results/fig_error_by_range.png
"""
import math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

R = "results"
res = pd.read_csv(f"{R}/all_results.csv")
pred = pd.read_csv(f"{R}/predictions_ml.csv").merge(
    pd.read_csv(f"{R}/predictions_lstm.csv")[["Date", "LSTM"]], on="Date")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
assert len(pred) == len(test), "Jumlah baris prediksi harus sama dengan data test"
pred["NaiveBaseline"] = test["Close"].values
close_today = test["Close"].values
actual = pred["Actual"].values
models = [m for m in ["NaiveBaseline", "LinearRegression", "SVR", "LSTM", "RandomForest", "XGBoost"] if m in pred.columns]
n = len(pred)

# ---------------- Tabel 1: akurasi ----------------
def fmt(row, k):
    s = f"{row[k]:.1f}" if k in ("MAE", "RMSE") else f"{row[k]:.3f}"
    if f"{k}_std" in row and pd.notna(row.get(f"{k}_std")):
        s += f" ± {row[f'{k}_std']:.{1 if k in ('MAE', 'RMSE') else 3}f}"
    return s
acc = pd.DataFrame([{
    "Model": r["Model"], "MAE (Rp)": fmt(r, "MAE"), "RMSE (Rp)": fmt(r, "RMSE"),
    "MAPE (%)": f"{r['MAPE_%']:.2f}", "R2": fmt(r, "R2"),
    "Akurasi arah (%)": "-" if pd.isna(r["Dir_Acc_%"]) else f"{r['Dir_Acc_%']:.1f}"}
    for _, r in res.iterrows()])
acc.to_csv(f"{R}/table_accuracy.csv", index=False)

# ---------------- Tabel 2: komputasi ----------------
comp = res[res["Model"] != "NaiveBaseline"].copy()
comp_t = pd.DataFrame({
    "Model": comp["Model"],
    "Training (s)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(comp["Train_s_mean"], comp["Train_s_std"])],
    "Inferensi 1 sampel (ms)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(comp["Infer_1sample_ms_mean"], comp["Infer_1sample_ms_std"])],
    "Inferensi batch 525 (ms)": comp["Infer_batch_ms"].round(2),
    "Ukuran model (KB)": comp["Model_size_KB"].round(1),
    "Waktu tuning (s)": comp["Tuning_s"].round(1),
    "RAM training (MB, perkiraan)": comp["RAM_train_MB"].round(1)})
comp_t.to_csv(f"{R}/table_compute.csv", index=False)

# ---------------- Uji signifikansi ----------------
def p_norm(z):
    return math.erfc(abs(z) / math.sqrt(2))

def dm_test(e1, e2, lag=5):
    """Diebold-Mariano, loss kuadrat, varians Newey-West. z>0: model 2 lebih baik dari model 1."""
    d = e1 ** 2 - e2 ** 2
    dc = d - d.mean()
    var = np.mean(dc * dc)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.mean(dc[k:] * dc[:-k])
    z = d.mean() / math.sqrt(var / len(d))
    return z, p_norm(z)

err = {m: actual - pred[m].values for m in models}
rows = []
for m in models:
    if m == "NaiveBaseline":
        continue
    z, p = dm_test(err["NaiveBaseline"], err[m])
    k = int(np.sum(np.sign(pred[m].values - close_today) == np.sign(actual - close_today)))
    pd_ = p_norm((k - n / 2) / math.sqrt(n / 4))
    rows.append({"Model": m,
                 "RMSE (Rp)": round(math.sqrt(np.mean(err[m] ** 2)), 1),
                 "Selisih MSE vs naif": round(np.mean(err["NaiveBaseline"] ** 2) - np.mean(err[m] ** 2), 0),
                 "DM z": round(z, 2), "p (vs naif)": round(p, 4),
                 "Hasil vs naif": ("lebih baik" if z > 0 else "lebih buruk") if p < 0.05 else "tidak berbeda nyata",
                 "Akurasi arah (%)": round(100 * k / n, 1), "p (arah vs 50%)": round(pd_, 4)})
sig = pd.DataFrame(rows); sig.to_csv(f"{R}/table_significance.csv", index=False)

pw = pd.DataFrame(index=models, columns=models, dtype=float)
for a in models:
    for b in models:
        pw.loc[a, b] = np.nan if a == b else round(dm_test(err[a], err[b])[1], 4)
pw.to_csv(f"{R}/table_pairwise_pvalues.csv")

# ---------------- Error per rentang harga ----------------
cap = train["target"].max()
hi = actual > cap
byr = pd.DataFrame([{"Model": m,
                     f"MAE harga <= {cap:.0f} (n={int((~hi).sum())})": round(np.abs(err[m][~hi]).mean(), 1),
                     f"MAE harga > {cap:.0f} (n={int(hi.sum())})": round(np.abs(err[m][hi]).mean(), 1)} for m in models])
byr.to_csv(f"{R}/table_by_price_range.csv", index=False)

# ---------------- Grafik ----------------
COL = {"NaiveBaseline": "#7F7F7F", "LinearRegression": "#0072B2", "SVR": "#009E73",
       "LSTM": "#D55E00", "RandomForest": "#CC79A7", "XGBoost": "#E69F00"}
dates = pd.to_datetime(pred["Date"])

fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
ax[0].plot(dates, actual, color="black", lw=1.6, label="Aktual")
for m in ["NaiveBaseline", "LinearRegression", "SVR", "LSTM"]:
    if m in models: ax[0].plot(dates, pred[m], lw=1, color=COL[m], label=m, alpha=0.9)
ax[0].set_title("Aktual vs prediksi: baseline, Linear Regression, SVR, LSTM"); ax[0].legend(ncol=5, fontsize=8)
ax[1].plot(dates, actual, color="black", lw=1.6, label="Aktual")
for m in ["RandomForest", "XGBoost"]:
    if m in models: ax[1].plot(dates, pred[m], lw=1.2, color=COL[m], label=m)
ax[1].axhline(cap, color="gray", ls="--", lw=1); ax[1].text(dates.iloc[0], cap + 60, f"harga tertinggi data train (Rp {cap:,.0f})".replace(",", "."), fontsize=8, color="gray")
ax[1].set_title("Aktual vs prediksi: model pohon (tertahan di batas data train)"); ax[1].legend(fontsize=8)
for a in ax: a.set_ylabel("Harga Close (Rp)"); a.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{R}/fig_actual_vs_pred.png", dpi=150); plt.close(fig)

ml = res[res["Model"] != "NaiveBaseline"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (x, lab) in zip(ax, [("Train_s_mean", "Waktu training (detik, skala log)"),
                            ("Infer_1sample_ms_mean", "Inferensi 1 sampel (ms, skala log)"),
                            ("Model_size_KB", "Ukuran model (KB, skala log)")]):
    for _, r in ml.iterrows():
        a.scatter(r[x], r["RMSE"], s=90, color=COL[r["Model"]], zorder=3)
        a.annotate(r["Model"], (r[x], r["RMSE"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    a.axhline(float(res.loc[res.Model == "NaiveBaseline", "RMSE"].iloc[0]), color="gray", ls="--", lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(lab); a.set_ylabel("RMSE (Rp, skala log), lebih kecil lebih baik"); a.grid(alpha=0.25, which="major")
    a.set_yticks([150, 200, 300, 500]); a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); a.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter());
ax[0].text(0.02, 0.95, "garis putus abu-abu: baseline naif", transform=ax[0].transAxes, fontsize=8, color="gray", va="top")
fig.suptitle("Trade-off akurasi vs biaya komputasi (pojok kiri-bawah = terbaik)"); fig.tight_layout()
fig.savefig(f"{R}/fig_tradeoff.png", dpi=150); plt.close(fig)

fig, a = plt.subplots(figsize=(9, 4.6)); w = 0.38; x = np.arange(len(models))
c1, c2 = byr.columns[1], byr.columns[2]
a.bar(x - w / 2, byr[c1], w, label=c1, color="#56B4E9"); a.bar(x + w / 2, byr[c2], w, label=c2, color="#D55E00")
a.set_xticks(x); a.set_xticklabels(models, rotation=20, fontsize=9); a.set_ylabel("MAE (Rp)"); a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)
a.set_title("Error per rentang harga: di mana model pohon gagal"); fig.tight_layout()
fig.savefig(f"{R}/fig_error_by_range.png", dpi=150); plt.close(fig)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("\n== AKURASI ==\n", acc.to_string(index=False))
print("\n== KOMPUTASI ==\n", comp_t.to_string(index=False))
print("\n== SIGNIFIKANSI (uji Diebold-Mariano, alpha 0,05) ==\n", sig.to_string(index=False))
print("\n== ERROR PER RENTANG HARGA ==\n", byr.to_string(index=False))
print("\nTabel dan grafik tersimpan di folder results/")