"""
Preprocessing dataset BBCA (cleaning saja, TANPA scaling).
Scaling dilakukan nanti SETELAH split train/test supaya tidak ada data leakage.

Input : data/dataset_BBCA_2015_2025.csv   (raw, jangan diubah)
Output: data/dataset_BBCA_clean.csv       (dataset terkunci untuk semua model)
"""
import pandas as pd

RAW = "data/dataset_BBCA_2015_2025.csv"
OUT = "data/dataset_BBCA_clean.csv"

# 1. Baca: pemisah ';' dan semua kolom dibaca sebagai teks dulu
df = pd.read_csv(RAW, sep=";", dtype=str)
n_raw = len(df)

# 2. Tanggal format dd/mm/yyyy, urutkan, hapus duplikat
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

# 3. Kolom harga & volume -> numerik
for c in ["Open", "High", "Low", "Close", "Volume"]:
    df[c] = pd.to_numeric(df[c], errors="raise")


# 4. Perbaiki Adj Close: titik ribuan menyisip di angka desimal
#    (mis. '2.058.509.521.484.370' seharusnya 2058.509521484370).
#    Hapus titik, lalu cari posisi desimal yang membuat nilainya
#    wajar (<= Close dan dekat dengan Close).
def fix_adj(raw, close):
    digits = raw.replace(".", "")
    best = None
    for k in range(1, len(digits)):
        v = float(digits[:k] + "." + digits[k:])
        if 0.3 * close < v <= 1.001 * close:
            if best is None or abs(v - close) < abs(best - close):
                best = v
    return best

raw_adj = pd.read_csv(RAW, sep=";", dtype=str)["Adj Close"]
raw_adj = raw_adj.loc[df.index] if len(raw_adj) == len(df) else raw_adj
df["Adj Close"] = [fix_adj(a, c) for a, c in zip(raw_adj, df["Close"])]

# 5. Hapus hari non-trading yang diisi Yahoo (Volume = 0, O=H=L=C)
n_before = len(df)
df = df[df["Volume"] > 0].reset_index(drop=True)
n_zero = n_before - len(df)

# 6. Validasi konsistensi OHLC
assert df.isna().sum().sum() == 0, "Masih ada nilai kosong"
assert (df["High"] >= df["Low"]).all()
assert ((df["Close"] <= df["High"]) & (df["Close"] >= df["Low"])).all()
assert ((df["Open"] <= df["High"]) & (df["Open"] >= df["Low"])).all()
assert df["Date"].is_monotonic_increasing

df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
df.to_csv(OUT, index=False)

print(f"Baris raw           : {n_raw}")
print(f"Hari volume=0 dibuang: {n_zero}")
print(f"Baris bersih        : {len(df)}")
print(f"Periode             : {df['Date'].min().date()} s/d {df['Date'].max().date()}")
print(f"Disimpan ke         : {OUT}")