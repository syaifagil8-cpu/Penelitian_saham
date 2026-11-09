"""API FastAPI. Hanya merutekan permintaan; logika ada di predictor.py (prediksi) dan records.py (CRUD)."""
import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .predictor import FIGURES, Predictor
from api.records import RecordStore
state = {}


@asynccontextmanager
async def lifespan(app):
    base = os.environ.get("APP_BASE", ".")
    state["p"] = Predictor(base)                                   # model dimuat SEKALI di sini
    state["db"] = RecordStore(os.path.join(os.environ.get("APP_DB_DIR", os.path.join(base, "appdata")), "records.db"))
    yield


app = FastAPI(title="API Prediksi Saham BBCA (penelitian)", lifespan=lifespan)


def P():
    return state["p"]


def DB():
    return state["db"]


def _bad(e):
    return HTTPException(status_code=400, detail=str(e))


class RecordIn(BaseModel):
    model: str
    date: Optional[str] = None
    note: Optional[str] = ""
    tag: Optional[str] = "umum"


class RecordUpdate(BaseModel):
    model: Optional[str] = None
    date: Optional[str] = None
    note: Optional[str] = None
    tag: Optional[str] = None


# ------------------------------------------------------------------ prediksi
@app.get("/health")
def health():
    return {"status": "ok", "models": P().available_models()}


@app.get("/models")
def models():
    return {"available": P().available_models(), "unavailable": P().unavailable}


@app.get("/dataset")
def dataset():
    return P().dataset_info()


@app.get("/predict")
def predict(model: str, date: Optional[str] = None):
    try:
        return P().predict(model, date)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model}' tidak tersedia")
    except ValueError as e:
        raise _bad(e)


@app.get("/predict-all")
def predict_all(date: Optional[str] = None):
    try:
        return P().predict_all(date)
    except ValueError as e:
        raise _bad(e)


@app.get("/history")
def history(n: int = Query(90, ge=5, le=1000), end: Optional[str] = None):
    return P().history(n, end)


@app.get("/backtest")
def backtest(model: str):
    try:
        return P().backtest(model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prediksi uji untuk '{model}' tidak tersedia")


@app.get("/metrics")
def metrics():
    return P().metrics()


@app.get("/results")
def results():
    return P().results() or []


@app.get("/environment")
def environment():
    return P().environment()


@app.get("/figure/{name}")
def figure(name: str):
    if name not in FIGURES:
        raise HTTPException(status_code=404, detail="Gambar tidak ditemukan")
    path = os.path.join(P().base, "results", name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Gambar belum dibuat (jalankan eksperimen dulu)")
    return FileResponse(path)


# --------------------------------------------------------- CRUD riwayat prediksi
# Nilai prediksi selalu dihitung ULANG oleh server, bukan diterima dari klien, supaya catatan tidak bisa dipalsukan.
def _predict_for(model, date):
    try:
        return P().predict(model, date)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model}' tidak tersedia")
    except ValueError as e:
        raise _bad(e)


@app.post("/records", status_code=201)                                    # CREATE
def create_record(body: RecordIn):
    pred = _predict_for(body.model, body.date)
    try:
        return DB().create(pred, body.note, body.tag)
    except ValueError as e:
        raise _bad(e)


@app.get("/records")                                                      # READ (daftar)
def list_records(model: Optional[str] = None, tag: Optional[str] = None, q: Optional[str] = None,
                 limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), order: str = "desc"):
    return DB().list(model, tag, q, limit, offset, order)


@app.get("/records/stats")
def records_stats():
    return DB().stats()


@app.get("/records/export")
def export_records():
    return Response(content=DB().export_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=riwayat_prediksi.csv"})


@app.get("/records/{rid}")                                                # READ (satu)
def get_record(rid: int):
    try:
        return DB().get(rid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Catatan tidak ditemukan")


@app.put("/records/{rid}")                                                # UPDATE
def update_record(rid: int, body: RecordUpdate):
    try:
        current = DB().get(rid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Catatan tidak ditemukan")
    pred = None
    if body.model is not None or body.date is not None:                  # model atau tanggal berubah -> hitung ulang
        pred = _predict_for(body.model or current["model"], body.date or current["as_of_date"])
    try:
        return DB().update(rid, pred, body.note, body.tag)
    except ValueError as e:
        raise _bad(e)


@app.delete("/records/{rid}")                                             # DELETE
def delete_record(rid: int):
    try:
        DB().delete(rid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Catatan tidak ditemukan")
    return {"deleted": rid}