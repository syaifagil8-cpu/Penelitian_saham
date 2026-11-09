import pandas as pd
from pathlib import Path

files = [
    "results/predictions_ml.csv",
    "results/predictions_lstm.csv"
]

for file in files:
    path = Path(file)

    print("\n" + "=" * 50)
    print("FILE:", file)
    print("=" * 50)

    if not path.exists():
        print("STATUS: File tidak ditemukan!")
        continue

    df = pd.read_csv(path)

    print("Jumlah baris:", len(df))
    print("Jumlah kolom:", len(df.columns))
    print("Nama kolom:", df.columns.tolist())
    print("\n5 data pertama:")
    print(df.head().to_string(index=False))
    print("\nJumlah nilai kosong:", df.isna().sum().sum())

    if len(df) > 0:
        print("\nSTATUS: File berisi data.")
    else:
        print("\nSTATUS: File kosong.")