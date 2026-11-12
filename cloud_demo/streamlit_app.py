from pathlib import Path
import json
import os

import pandas as pd
import streamlit as st
import requests
# =========================================================
# KONFIGURASI
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
API_URL = os.getenv(
    "API_URL",
    "https://bbca-cloud-api-2318c.containers.snapdeploy.app"
).rstrip("/")

st.set_page_config(
    page_title="Prediksi Saham BBCA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



# =========================================================
# FUNGSI MEMBACA DATA
# =========================================================

@st.cache_data(ttl=300)
def api_get(endpoint):
    try:
        response = requests.get(
            f"{API_URL}/{endpoint.lstrip('/')}",
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"Gagal mengakses API ({endpoint}): {error}")
        return None


@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(ttl=300)
def load_predictions():
    data = api_get("predictions?model=all&limit=525")

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    return df.sort_values("Date")


# Data dari API online
predictions = load_predictions()

results_data = api_get("results")
all_results = (
    pd.DataFrame(results_data)
    if isinstance(results_data, list)
    else pd.DataFrame()
)

metrics_data = api_get("metrics") or {}

accuracy = pd.DataFrame(
    metrics_data.get("accuracy", [])
)

compute = pd.DataFrame(
    metrics_data.get("computation", [])
)

significance = pd.DataFrame(
    metrics_data.get("significance", [])
)

price_range = pd.DataFrame(
    metrics_data.get("price_range", [])
)

environment = api_get("environment") or {}

# Dataset dan fitur tetap dari repository
dataset = load_csv(DATA_DIR / "dataset_BBCA_clean.csv")
features_info = load_json(DATA_DIR / "features.json")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📈 BBCA Prediction")
st.sidebar.caption("Perbandingan Machine Learning dan LSTM")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Beranda",
        "Prediksi Historis",
        "Perbandingan Model",
        "Analisis Eksperimen",
        "Informasi Dataset"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Demo berbasis hasil eksperimen tersimpan.")
st.sidebar.caption("Periode data: 2015–2025")


# =========================================================
# HEADER
# =========================================================

st.title("Sistem Prediksi Harga Saham BBCA")
st.caption(
    "Perbandingan model Machine Learning dan Long Short-Term Memory "
    "untuk prediksi harga penutupan saham BBCA."
)

st.info(
    "Demo online ini menampilkan hasil eksperimen yang telah disimpan. "
    "Proses pelatihan model dan prediksi melalui API tetap dijalankan "
    "pada aplikasi lokal."
)


# =========================================================
# BERANDA
# =========================================================

if menu == "Beranda":

    st.header("Ringkasan Penelitian")

    if not dataset.empty:
        dataset["Date"] = pd.to_datetime(dataset["Date"])

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Jumlah Data", f"{len(dataset):,}")
        c2.metric(
            "Awal Periode",
            dataset["Date"].min().strftime("%d-%m-%Y")
        )
        c3.metric(
            "Akhir Periode",
            dataset["Date"].max().strftime("%d-%m-%Y")
        )
        c4.metric(
            "Jumlah Fitur",
            str(len(features_info.get("features", [])))
        )

        st.subheader("Pergerakan Harga Penutupan BBCA")

        if "Close" in dataset.columns:
            chart_data = dataset.set_index("Date")[["Close"]]
            st.line_chart(chart_data, height=350)

    st.subheader("Model yang Dibandingkan")

    model_names = [
        "Linear Regression",
        "Random Forest",
        "Support Vector Regression (SVR)",
        "XGBoost",
        "Long Short-Term Memory (LSTM)"
    ]

    cols = st.columns(5)

    for col, name in zip(cols, model_names):
        col.markdown(f"**{name}**")

    st.subheader("Tahapan Penelitian")

    st.markdown("""
    1. Pengumpulan data historis saham BBCA.
    2. Pembersihan dan preprocessing data.
    3. Ekstraksi 18 fitur.
    4. Pembagian data training dan testing secara kronologis.
    5. Pelatihan lima model prediksi.
    6. Evaluasi dan perbandingan hasil prediksi.
    """)


# =========================================================
# PREDIKSI HISTORIS
# =========================================================

elif menu == "Prediksi Historis":

    st.header("Hasil Prediksi Historis")

    if predictions.empty:
        st.error("File prediksi tidak ditemukan.")
        st.stop()

    model_options = {
        "Linear Regression": "LinearRegression",
        "Random Forest": "RandomForest",
        "SVR": "SVR",
        "XGBoost": "XGBoost",
        "LSTM": "LSTM"
    }

    selected_label = st.selectbox(
        "Pilih Model",
        list(model_options.keys())
    )

    selected_model = model_options[selected_label]

    dates = predictions["Date"].dt.strftime("%Y-%m-%d").tolist()

    selected_date = st.select_slider(
        "Pilih Tanggal Pengujian",
        options=dates,
        value=dates[-1]
    )

    row = predictions[
        predictions["Date"].dt.strftime("%Y-%m-%d") == selected_date
    ].iloc[0]

    actual = float(row["Actual"])
    predicted = float(row[selected_model])
    error = abs(actual - predicted)
    error_pct = (error / actual * 100) if actual else 0

    c1, c2, c3 = st.columns(3)

    c1.metric("Harga Aktual", f"Rp {actual:,.2f}")
    c2.metric("Hasil Prediksi", f"Rp {predicted:,.2f}")
    c3.metric("Absolute Error", f"Rp {error:,.2f}")

    st.caption(f"Tanggal pengujian: {selected_date}")
    st.caption(f"Persentase absolute error: {error_pct:.2f}%")

    st.subheader("Grafik Aktual dan Prediksi")

    chart = predictions.set_index("Date")[
        ["Actual", selected_model]
    ].rename(columns={
        "Actual": "Harga Aktual",
        selected_model: "Harga Prediksi"
    })

    st.line_chart(chart, height=400)

    st.subheader("Data Prediksi")

    display = predictions[
        ["Date", "Actual", selected_model]
    ].copy()

    display.columns = [
        "Tanggal",
        "Harga Aktual",
        "Harga Prediksi"
    ]

    st.dataframe(
        display.sort_values("Tanggal", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PERBANDINGAN MODEL
# =========================================================

elif menu == "Perbandingan Model":

    st.header("Perbandingan Kinerja Model")

    if not accuracy.empty:
        st.subheader("Tabel Evaluasi Model")

        st.dataframe(
            accuracy,
            use_container_width=True,
            hide_index=True
        )

    elif not all_results.empty:
        st.dataframe(
            all_results,
            use_container_width=True,
            hide_index=True
        )

    if not all_results.empty:
        required = {"Model", "RMSE"}

        if required.issubset(all_results.columns):
            st.subheader("Perbandingan RMSE")

            rmse_data = all_results[["Model", "RMSE"]].copy()
            rmse_data["RMSE"] = pd.to_numeric(
                rmse_data["RMSE"], errors="coerce"
            )
            rmse_data = rmse_data.dropna().set_index("Model")

            st.bar_chart(rmse_data, height=350)

        if {"Model", "MAE"}.issubset(all_results.columns):
            st.subheader("Perbandingan MAE")

            mae_data = all_results[["Model", "MAE"]].copy()
            mae_data["MAE"] = pd.to_numeric(
                mae_data["MAE"], errors="coerce"
            )
            mae_data = mae_data.dropna().set_index("Model")

            st.bar_chart(mae_data, height=350)

    st.subheader("Grafik Hasil Eksperimen")

    figures = [
        ("fig_actual_vs_pred.png", "Aktual vs Prediksi"),
        ("fig_error_by_range.png", "Error Berdasarkan Rentang Harga"),
        ("fig_perbandingan_5_model.png", "Perbandingan Lima Model"),
        ("fig_tradeoff.png", "Trade-off Kinerja Model")
    ]

    for filename, caption in figures:
        image_url = f"{API_URL}/figure/{filename}"

    st.image(
        image_url,
        caption=caption,
        use_container_width=True
    )


# =========================================================
# ANALISIS EKSPERIMEN
# =========================================================

elif menu == "Analisis Eksperimen":

    st.header("Analisis Eksperimen")

    if not compute.empty:
        st.subheader("Waktu Komputasi dan Ukuran Model")
        st.dataframe(
            compute,
            use_container_width=True,
            hide_index=True
        )

    if not significance.empty:
        st.subheader("Uji Signifikansi")
        st.dataframe(
            significance,
            use_container_width=True,
            hide_index=True
        )

    if not price_range.empty:
        st.subheader("Evaluasi Berdasarkan Rentang Harga")
        st.dataframe(
            price_range,
            use_container_width=True,
            hide_index=True
        )

    if environment:
        st.subheader("Lingkungan Eksperimen")

        env_display = {
            "Python": environment.get("python"),
            "Sistem Operasi": environment.get("os"),
            "CPU": environment.get("cpu"),
            "Jumlah Thread": environment.get("n_thread_model"),
            "Seed": environment.get("seed"),
            "Jumlah Pengulangan": environment.get("n_pengulangan"),
            "Di dalam Docker": environment.get("di_dalam_docker")
        }

        st.json(env_display)

    history_path = RESULTS_DIR / "lstm_history.csv"

    if history_path.exists():
        history = load_csv(history_path)

        st.subheader("Riwayat Pelatihan LSTM")
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = history.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            st.line_chart(
                history[numeric_columns],
                height=300
            )


# =========================================================
# INFORMASI DATASET
# =========================================================

elif menu == "Informasi Dataset":

    st.header("Informasi Dataset dan Fitur")

    if not dataset.empty:
        st.subheader("Preview Dataset")

        st.dataframe(
            dataset.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Statistik Deskriptif")
        st.dataframe(
            dataset.describe(),
            use_container_width=True
        )

    st.subheader("Daftar 18 Fitur")

    feature_list = features_info.get("features", [])

    if feature_list:
        feature_df = pd.DataFrame({
            "No": range(1, len(feature_list) + 1),
            "Nama Fitur": feature_list
        })

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

    st.write(
        "**Target prediksi:**",
        features_info.get("target", "Tidak tersedia")
    )

    st.subheader("Informasi Pembagian Data")

    train = load_csv(DATA_DIR / "train.csv")
    test = load_csv(DATA_DIR / "test.csv")

    c1, c2 = st.columns(2)
    c1.metric("Data Training", f"{len(train):,}")
    c2.metric("Data Testing", f"{len(test):,}")

    st.caption(
        "Pembagian data dilakukan secara kronologis "
        "untuk mempertahankan urutan waktu."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Proyek penelitian perbandingan Machine Learning dan LSTM "
    "untuk prediksi harga saham BBCA. "
    "Hasil yang ditampilkan merupakan hasil eksperimen tersimpan, "
    "bukan rekomendasi investasi."
)
