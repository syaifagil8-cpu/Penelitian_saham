"""Penyimpanan CRUD riwayat prediksi (SQLite).
Terpisah dari dataset, model, dan hasil eksperimen yang terkunci (hanya-baca)."""
import csv, io, os, sqlite3
from contextlib import closing
from datetime import datetime, timezone

FIELDS = ["id", "created_at", "updated_at", "model", "as_of_date", "close_today", "predicted_close",
          "interval_low", "interval_high", "baseline_close", "actual_next_close", "actual_next_date",
          "error", "in_training_data", "inference_ms", "note", "tag"]
MAX_NOTE, MAX_TAG = 500, 30

SCHEMA = """CREATE TABLE IF NOT EXISTS records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  model TEXT NOT NULL, as_of_date TEXT NOT NULL,
  close_today REAL NOT NULL, predicted_close REAL NOT NULL,
  interval_low REAL, interval_high REAL, baseline_close REAL NOT NULL,
  actual_next_close REAL, actual_next_date TEXT, error REAL,
  in_training_data INTEGER NOT NULL DEFAULT 0, inference_ms REAL,
  note TEXT NOT NULL DEFAULT '', tag TEXT NOT NULL DEFAULT 'umum')"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_note(x):
    x = (x or "").strip()
    if len(x) > MAX_NOTE:
        raise ValueError(f"Catatan maksimal {MAX_NOTE} karakter")
    return x


def clean_tag(x):
    x = (x or "").strip() or "umum"
    if len(x) > MAX_TAG:
        raise ValueError(f"Label maksimal {MAX_TAG} karakter")
    return x


def _pred_values(p):
    iv = p.get("interval80") or [None, None]
    return (p["model"], p["as_of_date"], p["close_today"], p["predicted_close"], iv[0], iv[1], p["baseline_close"],
            p.get("actual_next_close"), p.get("actual_next_date"), p.get("error"),
            int(bool(p.get("in_training_data"))), p.get("inference_ms"))


def _out(row):
    d = dict(row)
    d["in_training_data"] = bool(d["in_training_data"])
    return d


class RecordStore:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with closing(self._conn()) as c, c:
            c.execute(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # ---------------- Create ----------------
    def create(self, pred, note="", tag="umum"):
        note, tag, now = clean_note(note), clean_tag(tag), _now()
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO records (created_at, updated_at, model, as_of_date, close_today, predicted_close,"
                " interval_low, interval_high, baseline_close, actual_next_close, actual_next_date, error,"
                " in_training_data, inference_ms, note, tag) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (now, now) + _pred_values(pred) + (note, tag))
            rid = cur.lastrowid
        return self.get(rid)

    # ---------------- Read ----------------
    def get(self, rid):
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM records WHERE id = ?", (rid,)).fetchone()
        if row is None:
            raise KeyError(rid)
        return _out(row)

    def list(self, model=None, tag=None, q=None, limit=50, offset=0, order="desc"):
        where, args = [], []
        if model:
            where.append("model = ?"); args.append(model)
        if tag:
            where.append("tag = ?"); args.append(tag)
        if q:
            esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("(note LIKE ? ESCAPE '\\' OR tag LIKE ? ESCAPE '\\')"); args += [f"%{esc}%", f"%{esc}%"]
        w = ("WHERE " + " AND ".join(where)) if where else ""
        direction = "ASC" if str(order).lower() == "asc" else "DESC"
        limit, offset = max(1, min(int(limit), 500)), max(0, int(offset))
        with closing(self._conn()) as c:
            total = c.execute(f"SELECT COUNT(*) FROM records {w}", args).fetchone()[0]
            rows = c.execute(f"SELECT * FROM records {w} ORDER BY id {direction} LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return {"total": total, "items": [_out(r) for r in rows]}

    def stats(self):
        with closing(self._conn()) as c:
            r = c.execute("SELECT COUNT(*) n, SUM(actual_next_close IS NOT NULL) w, AVG(ABS(error)) mae,"
                          " SUM(in_training_data) tr FROM records").fetchone()
            by = c.execute("SELECT model, COUNT(*) n FROM records GROUP BY model ORDER BY n DESC").fetchall()
        return {"total": r["n"], "with_actual": int(r["w"] or 0),
                "mean_abs_error": None if r["mae"] is None else round(r["mae"], 1),
                "in_training_data": int(r["tr"] or 0), "by_model": {b["model"]: b["n"] for b in by}}

    # ---------------- Update ----------------
    def update(self, rid, pred=None, note=None, tag=None):
        self.get(rid)                                   # KeyError kalau tidak ada
        sets, args = ["updated_at = ?"], [_now()]
        if pred is not None:
            sets += ["model=?", "as_of_date=?", "close_today=?", "predicted_close=?", "interval_low=?", "interval_high=?",
                     "baseline_close=?", "actual_next_close=?", "actual_next_date=?", "error=?", "in_training_data=?", "inference_ms=?"]
            args += list(_pred_values(pred))
        if note is not None:
            sets.append("note = ?"); args.append(clean_note(note))
        if tag is not None:
            sets.append("tag = ?"); args.append(clean_tag(tag))
        with closing(self._conn()) as c, c:
            c.execute(f"UPDATE records SET {', '.join(sets)} WHERE id = ?", args + [rid])
        return self.get(rid)

    # ---------------- Delete ----------------
    def delete(self, rid):
        self.get(rid)
        with closing(self._conn()) as c, c:
            c.execute("DELETE FROM records WHERE id = ?", (rid,))

    # ---------------- Ekspor ----------------
    def export_csv(self):
        items = self.list(limit=500)["items"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=FIELDS)
        w.writeheader(); w.writerows(items)
        return buf.getvalue()