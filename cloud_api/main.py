
import csv
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(
    title="API Demo Prediksi Saham BBCA",
    description="API untuk menampilkan hasil eksperimen prediksi saham BBCA.",
    version="1.0.0"
)

DATA_DIR = Path("/app/data")
RESULTS_DIR = Path("/app/results")


def read_csv(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} tidak ditemukan."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} tidak ditemukan."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/")
def home():
    return {
        "message": "API Demo Prediksi Saham BBCA",
        "status": "online",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "BBCA Prediction API"
    }


@app.get("/results")
def results():
    return read_csv("all_results.csv")


@app.get("/metrics")
def metrics():
    return {
        "accuracy": read_csv("table_accuracy.csv"),
        "computation": read_csv("table_compute.csv"),
        "significance": read_csv("table_significance.csv"),
        "price_range": read_csv("table_by_price_range.csv")
    }


@app.get("/predictions")
def predictions(model: str = "all", limit: int = 100):
    ml = read_csv("predictions_ml.csv")
    lstm = read_csv("predictions_lstm.csv")

    lstm_by_date = {row["Date"]: row for row in lstm}

    combined = []

    for row in ml:
        date = row["Date"]
        lstm_row = lstm_by_date.get(date, {})

        item = {
            "Date": date,
            "Actual": row.get("Actual"),
            "LinearRegression": row.get("LinearRegression"),
            "RandomForest": row.get("RandomForest"),
            "SVR": row.get("SVR"),
            "XGBoost": row.get("XGBoost"),
            "LSTM": lstm_row.get("LSTM")
        }

        if model.lower() == "all":
            combined.append(item)
        elif model in item:
            combined.append({
                "Date": date,
                "Actual": item["Actual"],
                model: item[model]
            })
        else:
            raise HTTPException(
                status_code=400,
                detail="Model tidak dikenal."
            )

    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit minimal 1.")

    return combined[-min(limit, 525):]


@app.get("/dataset")
def dataset():
    path = DATA_DIR / "dataset_BBCA_clean.csv"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset tidak ditemukan.")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)

    features = read_json("features.json")

    return {
        "total_rows": len(rows),
        "columns": columns,
        "first_date": rows[0].get("Date") if rows else None,
        "last_date": rows[-1].get("Date") if rows else None,
        "features": features.get("features", []),
        "target": features.get("target")
    }


@app.get("/environment")
def environment():
    path = RESULTS_DIR / "environment.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Environment tidak ditemukan.")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/figure/{name}")
def figure(name: str):
    allowed = {
        "fig_actual_vs_pred.png",
        "fig_tradeoff.png",
        "fig_error_by_range.png",
        "fig_perbandingan_5_model.png",
        "fig_lstm_arsitektur.png",
        "fig_lstm_training.png"
    }

    if name not in allowed:
        raise HTTPException(status_code=404, detail="Gambar tidak tersedia.")

    path = RESULTS_DIR / name

    if not path.exists():
        raise HTTPException(status_code=404, detail="File gambar tidak ditemukan.")

    return FileResponse(path, media_type="image/png")
