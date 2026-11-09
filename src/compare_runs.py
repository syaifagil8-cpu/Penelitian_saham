"""Uji reproducibility: bandingkan hasil dua kali eksekusi.
Pemakaian:  python src/compare_runs.py runs/run1 runs/run2"""
import sys, os
import numpy as np
import pandas as pd

d1, d2 = sys.argv[1], sys.argv[2]
a, b = (pd.read_csv(f"{d}/all_results.csv").set_index("Model") for d in (d1, d2))
ACC = ["MAE", "RMSE", "MAPE_%", "R2", "Dir_Acc_%"]
TIME = ["Train_s_mean", "Infer_1sample_ms_mean", "Infer_batch_ms", "Model_size_KB"]
rows = []
for m in a.index:
    r = {"Model": m}
    r["Selisih maks metrik akurasi"] = float(np.nanmax(np.abs(a.loc[m, ACC].astype(float).values - b.loc[m, ACC].astype(float).values)))
    for t in TIME:
        x, y = a.loc[m, t], b.loc[m, t]
        r[f"{t} selisih %"] = None if pd.isna(x) or x == 0 else round(abs(x - y) / abs(x) * 100, 1)
    rows.append(r)
rep = pd.DataFrame(rows)

pred = {}
for f in ["predictions_ml.csv", "predictions_lstm.csv"]:
    p1, p2 = pd.read_csv(f"{d1}/{f}"), pd.read_csv(f"{d2}/{f}")
    for c in p1.columns:
        if c not in ("Date", "Actual"):
            pred[c] = float(np.max(np.abs(p1[c].values - p2[c].values)))
rep["Selisih maks prediksi (Rp)"] = rep["Model"].map(pred)

out = os.path.join(os.path.dirname(d1.rstrip("/\\")) or ".", "reproducibility_report.csv")
rep.to_csv(out, index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(rep.round(4).to_string(index=False))
ml = rep[rep.Model.isin(["LinearRegression", "RandomForest", "SVR", "XGBoost"])]
print("\nModel ML: akurasi", "IDENTIK" if (ml["Selisih maks metrik akurasi"] < 1e-9).all() else "BERBEDA (lihat tabel)")
l = rep[rep.Model == "LSTM"]["Selisih maks metrik akurasi"].iloc[0]
print(f"LSTM: selisih maks metrik akurasi = {l:.6f}", "(IDENTIK)" if l < 1e-9 else "")
print("Laporan disimpan ke", out)