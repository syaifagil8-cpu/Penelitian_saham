"""Tampilan Streamlit. Hanya menampilkan; semua perhitungan ada di API."""
import os
from html import escape

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
INK, BLUE, ORANGE, GRAY = "#1B1F2A", "#0072B2", "#D55E00", "#7F7F7F"
# Palet ramah buta warna (Okabe-Ito)
MODEL_COLORS = {"LinearRegression": "#0072B2", "SVR": "#009E73", "LSTM": "#D55E00", "RandomForest": "#CC79A7", "XGBoost": "#E69F00"}

st.set_page_config(page_title="Prediksi BBCA | ML vs LSTM", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;700&family=DM+Mono:wght@500&display=swap');
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1240px; }
footer { visibility: hidden; }
h1, h2, h3, h4, .hero, .phead, .brand { font-family: 'DM Sans', system-ui, sans-serif; }
.hero { background: linear-gradient(120deg, #12304f 0%, #0b5a8a 100%); color: #fff; border-radius: 16px;
        padding: 30px 34px 24px 34px; margin-bottom: 22px; box-shadow: 0 4px 14px rgba(18,48,79,.18); }
.hero h1 { color: #fff; font-size: 2.05rem; line-height: 1.2; margin: 0 0 8px 0; padding: 0; }
.hero p { color: #d7e6f2; margin: 0; font-size: 1.02rem; }
.hero .chip { display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.32);
              color: #fff; border-radius: 999px; padding: 3px 13px; margin: 14px 8px 0 0; font-size: .82rem; }
.phead { margin: 0 0 14px 0; padding-bottom: 12px; border-bottom: 3px solid #0072B2; }
.phead h2 { margin: 0; padding: 0; font-size: 1.75rem; color: #1B1F2A; }
.phead p { margin: 4px 0 0 0; color: #5B6475; }
.pchip { display: inline-block; border-radius: 999px; padding: 2px 12px; margin: 0 8px 8px 0; font-size: .82rem; font-weight: 600; }
.pchip.blue { background: #E1F0FA; color: #00507C; }
.pchip.orange { background: #FBE6D8; color: #8A3A00; }
.pchip.gray { background: #ECEEF2; color: #3E4656; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #DCE1E9; border-radius: 12px; padding: 14px 16px;
                           box-shadow: 0 1px 2px rgba(20,30,50,.06); }
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
.brand { font-size: 1.25rem; font-weight: 700; color: #12304f; margin: 4px 0 0 0; }
.brand-sub { font-size: .85rem; color: #5B6475; margin: 0 0 14px 0; }
.disc { margin-top: 28px; padding-top: 12px; border-top: 1px solid #DCE1E9; font-size: .82rem; color: #5B6475; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- pembantu
def rp(x):
    return "-" if x is None else "Rp " + f"{x:,.0f}".replace(",", ".")


def hero(title, subtitle, chips=()):
    ch = "".join(f'<span class="chip">{escape(c)}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>{ch}</div>', unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f'<div class="phead"><h2>{escape(title)}</h2>' + (f"<p>{escape(subtitle)}</p>" if subtitle else "") + "</div>",
                unsafe_allow_html=True)


def chips(*items):
    st.markdown("".join(f'<span class="pchip {kind}">{escape(text)}</span>' for kind, text in items), unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def get(path, **params):
    r = requests.get(f"{API}{path}", params=params or None, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_image(name):
    r = requests.get(f"{API}/figure/{name}", timeout=60)
    return r.content if r.ok else None


def api(method, path, **kw):
    """Panggilan API tanpa cache (untuk CRUD). Mengembalikan (berhasil, data_atau_pesan_error)."""
    try:
        r = requests.request(method, f"{API}{path}", timeout=60, **kw)
    except Exception as e:
        return False, f"API tidak dapat dihubungi: {e}"
    if r.ok:
        return True, (r.json() if "json" in r.headers.get("content-type", "") else r.content)
    try:
        return False, r.json().get("detail", r.text)
    except Exception:
        return False, r.text


def wib(ts):
    return (pd.to_datetime(ts) + pd.Timedelta(hours=7)).strftime("%d-%m-%Y %H:%M")

def tanpa_pm(df):
    """Ubah teks seperti '0.004 ± 0.003' menjadi satu angka (rata-rata saja)."""
    df = df.copy()
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.contains("±").any():
            df[c] = pd.to_numeric(s.str.split("±").str[0].str.strip(), errors="coerce")
    return df


try:
    info = get("/dataset")
except Exception as e:
    st.error(f"API tidak dapat dihubungi di {API}. Pastikan layanan api berjalan. ({e})")
    st.stop()

with st.sidebar:
    st.markdown('<p class="brand">Prediksi BBCA</p><p class="brand-sub">Machine learning vs LSTM · Docker</p>', unsafe_allow_html=True)
    page = st.radio("Menu", ["Beranda", "Prediksi", "Riwayat Prediksi", "Perbandingan Model", "Environment", "Tentang"],
                    label_visibility="collapsed")
    st.caption("Penelitian: perbandingan model untuk memprediksi harga Close BBCA, dijalankan di Docker.")

flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)

# ------------------------------------------------------------------- BERANDA
if page == "Beranda":
    hero("Prediksi harga saham BBCA: machine learning vs LSTM",
         "Perbandingan akurasi, biaya komputasi, dan reproducibility dalam satu kerangka, dijalankan di Docker.",
         ["BBCA.JK", f"{len(info['models_available'])} model + baseline", "Dataset terkunci", "Reproducible"])
    k = st.columns(4)
    k[0].metric("Hari bursa (setelah dibersihkan)", f"{info['rows_clean']:,}".replace(",", "."))
    k[1].metric("Data latih / uji", f"{info['n_train']:,} / {info['n_test']}".replace(",", "."))
    k[2].metric("Fitur", len(info["features"]))
    k[3].metric("Window LSTM", f"{info['window_lstm']} hari")

    st.subheader("Temuan utama")
    try:
        r = pd.DataFrame(get("/results"))
        naive = float(r.loc[r.Model == "NaiveBaseline", "RMSE"].iloc[0])
        ml = r[r.Model != "NaiveBaseline"]
        best = ml.loc[ml.RMSE.idxmin()]
        trees = ml[ml.Model.isin(["RandomForest", "XGBoost"])]
        fast, slow = ml.loc[ml.Train_s_mean.idxmin()], ml.loc[ml.Train_s_mean.idxmax()]
        c1, c2, c3 = st.columns(3)
        with c1.container(border=True):
            st.markdown("**Tebakan sederhana sulit dikalahkan**")
            st.write(f"Baseline \"besok = hari ini\" menghasilkan RMSE {naive:.1f}. Model terbaik ({best.Model}) {best.RMSE:.1f}, selisih yang sangat kecil.")
        with c2.container(border=True):
            st.markdown("**Model pohon gagal di luar rentang latih**")
            if len(trees):
                st.write(f"RMSE {trees.RMSE.min():.0f} sampai {trees.RMSE.max():.0f}, karena harga uji melewati harga tertinggi data latih.")
            else:
                st.write("Hasil model pohon belum tersedia.")
        with c3.container(border=True):
            st.markdown("**Biaya komputasi berbeda jauh**")
            st.write(f"Waktu latih dari {fast.Train_s_mean:.3f} detik ({fast.Model}) sampai {slow.Train_s_mean:.0f} detik ({slow.Model}).")
    except Exception:
        st.info("Ringkasan hasil belum tersedia. Jalankan eksperimen Docker dulu agar folder results terisi.")

    st.subheader("Mulai dari mana")
    g1, g2, g3 = st.columns(3)
    with g1.container(border=True):
        st.markdown("**Prediksi**")
        st.write("Coba tiap model, lihat baseline dan rentang kesalahannya.")
    with g2.container(border=True):
        st.markdown("**Perbandingan Model**")
        st.write("Grafik dan tabel akurasi, biaya komputasi, dan uji signifikansi.")
    with g3.container(border=True):
        st.markdown("**Riwayat Prediksi**")
        st.write("Simpan, lihat, ubah, dan hapus catatan prediksi (CRUD).")
    st.caption("Pilih halaman lewat menu di sebelah kiri.")

# ------------------------------------------------------------------ PREDIKSI
elif page == "Prediksi":
    page_header("Prediksi harga Close", "Perkiraan harga penutupan hari bursa berikutnya, dengan baseline dan rentang kesalahan.")
    st.info("Hasil ini adalah **perkiraan penelitian**, bukan sinyal beli atau jual. Pada data uji, tidak ada model yang "
            "lebih baik dari tebakan sederhana \"besok = hari ini\" secara meyakinkan, dan akurasi arah semua model sekitar 50%.")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.4, 1])
        model = c1.selectbox("Model", info["models_available"],
                             index=info["models_available"].index("SVR") if "SVR" in info["models_available"] else 0)
        mode = c2.radio("Data yang dipakai", ["Hari bursa berikutnya (data terakhir)", "Uji mundur (pilih tanggal)"])
        date = None
        if mode.startswith("Uji"):
            date = c3.date_input("Tanggal acuan", value=pd.Timestamp(info["test_start"]),
                                 min_value=pd.Timestamp(info["as_of_min"]), max_value=pd.Timestamp(info["as_of_max"])).isoformat()
        for name, why in info["models_unavailable"].items():
            st.caption(f"Model {name} tidak dimuat: {why}")
        run = st.button("Prediksi", type="primary")
    if run:
        try:
            st.session_state["res"] = get("/predict", model=model, **({"date": date} if date else {}))
            st.session_state["all"] = get("/predict-all", **({"date": date} if date else {}))
        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")
    res = st.session_state.get("res")
    if res:
        with st.container(border=True):
            tags = [("blue", res["model"]), ("gray", f"Acuan {res['as_of_date']}")]
            if res["in_training_data"]:
                tags.append(("orange", "Termasuk data latih"))
            chips(*tags)
            if res["in_training_data"]:
                st.warning("Tanggal ini termasuk data latih. Model sudah \"melihat\" periode ini, jadi hasilnya lebih optimistis daripada kenyataan. "
                           "Untuk uji yang adil, pilih tanggal setelah " + info["train_end"] + ".")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Close terakhir", rp(res["close_today"]))
            m2.metric("Prediksi berikutnya", rp(res["predicted_close"]), f"{res['predicted_change_pct']:+.2f}%", delta_color="off")
            m3.metric("Baseline naif (= hari ini)", rp(res["baseline_close"]))
            m4.metric("Rentang kesalahan 80%", "-" if not res["interval80"] else f"{res['interval80'][0]:,.0f} - {res['interval80'][1]:,.0f}".replace(",", "."))
            st.caption(f"Waktu inferensi {res['inference_ms']:.2f} ms. Rentang diambil dari sebaran selisih prediksi dan harga sebenarnya "
                       "pada data uji (80% selisih historis jatuh di dalamnya), bukan jaminan.")
            if res["actual_next_close"] is not None:
                st.success(f"Harga sebenarnya pada {res['actual_next_date']}: **{rp(res['actual_next_close'])}**. "
                           f"Selisih prediksi: {res['error']:+,.0f} Rupiah.".replace(",", "."))
        with st.container(border=True):
            h = pd.DataFrame(get("/history", n=90, end=res["as_of_date"]))
            h["Date"] = pd.to_datetime(h["Date"])
            nxt = pd.Timestamp(res["actual_next_date"]) if res["actual_next_date"] else h["Date"].max() + pd.offsets.BDay(1)
            line = alt.Chart(h).mark_line(color=GRAY, strokeWidth=2).encode(
                x=alt.X("Date:T", title=None), y=alt.Y("Close:Q", scale=alt.Scale(zero=False), title="Harga Close (Rp)"))
            layers = [line, alt.Chart(pd.DataFrame({"y": [res["close_today"]]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(y="y:Q")]
            if res["interval80"]:
                band = pd.DataFrame({"Date": [nxt], "low": [res["interval80"][0]], "high": [res["interval80"][1]]})
                layers.append(alt.Chart(band).mark_rule(color=BLUE, strokeWidth=8, opacity=0.25).encode(x="Date:T", y="low:Q", y2="high:Q"))
            pred_pt = pd.DataFrame({"Date": [nxt], "Close": [res["predicted_close"]]})
            layers.append(alt.Chart(pred_pt).mark_point(size=200, filled=True, color=BLUE).encode(x="Date:T", y="Close:Q"))
            if res["actual_next_close"] is not None:
                act = pd.DataFrame({"Date": [nxt], "Close": [res["actual_next_close"]]})
                layers.append(alt.Chart(act).mark_point(size=200, shape="diamond", filled=True, color=ORANGE).encode(x="Date:T", y="Close:Q"))
            st.altair_chart(alt.layer(*layers).properties(height=320, width="container"))
            st.caption("Garis abu-abu: harga 90 hari terakhir (garis putus = harga acuan). Titik biru: prediksi, dengan batang transparan sebagai "
                       "rentang kesalahan 80%" + (". Belah ketupat oranye: harga sebenarnya." if res["actual_next_close"] is not None else "."))
        with st.expander("Bandingkan semua model pada tanggal yang sama"):
            t = pd.DataFrame(st.session_state["all"])[["model", "predicted_close", "predicted_change_pct", "interval80", "inference_ms"]]
            t.columns = ["Model", "Prediksi (Rp)", "Perubahan (%)", "Rentang 80%", "Inferensi (ms)"]
            st.dataframe(t, hide_index=True)
        with st.expander("Simpan prediksi ini ke riwayat"):
            with st.form("simpan_cepat"):
                n_cepat = st.text_input("Catatan (opsional)", max_chars=500)
                t_cepat = st.text_input("Label", value="umum", max_chars=30)
                if st.form_submit_button("Simpan"):
                    ok, data = api("POST", "/records", json={"model": res["model"], "date": res["as_of_date"], "note": n_cepat, "tag": t_cepat})
                    st.success(f"Tersimpan sebagai catatan #{data['id']}. Lihat di halaman Riwayat Prediksi.") if ok else st.error(data)

# ---------------------------------------------------------- RIWAYAT (CRUD)
elif page == "Riwayat Prediksi":
    page_header("Riwayat prediksi", "Create, Read, Update, Delete atas catatan prediksi.")
    st.caption("Catatan disimpan di database terpisah; dataset, model, dan hasil eksperimen tidak ikut berubah. "
               "Nilai prediksi selalu dihitung oleh server, bukan diketik manual.")
    t_lihat, t_tambah, t_ubah, t_hapus = st.tabs(["Lihat", "Tambah", "Ubah", "Hapus"])
    ok_all, allrec = api("GET", "/records", params={"limit": 500})
    items = allrec["items"] if ok_all else []
    by_id = {r["id"]: r for r in items}

    def label(rid):
        r = by_id[rid]
        return f"#{rid} | {r['model']} | {r['as_of_date']} | {r['tag']}"

    def short(r):
        return pd.DataFrame({"Nilai": [wib(r["created_at"]), r["model"], r["as_of_date"], rp(r["close_today"]), rp(r["predicted_close"]),
                                       rp(r["actual_next_close"]), r["tag"], r["note"] or "-"]},
                            index=["Dibuat (WIB)", "Model", "Tanggal acuan", "Close terakhir", "Prediksi", "Harga sebenarnya", "Label", "Catatan"]).astype(str)

    with t_lihat:
        c1, c2, c3 = st.columns(3)
        f_model = c1.selectbox("Filter model", ["Semua"] + info["models_available"])
        f_tag = c2.text_input("Filter label (persis)")
        f_q = c3.text_input("Cari di catatan atau label")
        params = {"limit": 500}
        if f_model != "Semua":
            params["model"] = f_model
        if f_tag.strip():
            params["tag"] = f_tag.strip()
        if f_q.strip():
            params["q"] = f_q.strip()
        ok, data = api("GET", "/records", params=params)
        ok_s, stt = api("GET", "/records/stats")
        if not ok:
            st.error(data)
        else:
            if ok_s:
                m1, m2, m3 = st.columns(3)
                m1.metric("Jumlah catatan", stt["total"])
                m2.metric("Sudah ada harga sebenarnya", stt["with_actual"])
                m3.metric("Rata-rata selisih mutlak", rp(stt["mean_abs_error"]))
            if data["items"]:
                df = pd.DataFrame(data["items"])
                df["created_at"] = df["created_at"].map(wib)
                df["in_training_data"] = df["in_training_data"].map({True: "ya", False: "tidak"})
                view = df[["id", "created_at", "model", "as_of_date", "close_today", "predicted_close", "interval_low", "interval_high",
                           "actual_next_close", "error", "in_training_data", "tag", "note"]]
                view.columns = ["ID", "Dibuat (WIB)", "Model", "Tanggal acuan", "Close terakhir", "Prediksi", "Rentang bawah", "Rentang atas",
                                "Harga sebenarnya", "Selisih", "Data latih?", "Label", "Catatan"]
                st.caption(f"Menampilkan {len(df)} dari {data['total']} catatan.")
                st.dataframe(view, hide_index=True)
                ok_e, csv_bytes = api("GET", "/records/export")
                if ok_e:
                    st.download_button("Unduh semua catatan (CSV)", csv_bytes, file_name="riwayat_prediksi.csv", mime="text/csv")
            else:
                st.info("Belum ada catatan yang cocok. Tambahkan lewat tab Tambah, atau simpan dari halaman Prediksi.")

    with t_tambah:
        with st.form("tambah"):
            a, b = st.columns(2)
            m_new = a.selectbox("Model", info["models_available"])
            mode_new = b.radio("Data yang dipakai", ["Data terakhir", "Pilih tanggal"], horizontal=True)
            d_new = st.date_input("Tanggal acuan (dipakai jika memilih tanggal)", value=pd.Timestamp(info["test_start"]),
                                  min_value=pd.Timestamp(info["as_of_min"]), max_value=pd.Timestamp(info["as_of_max"]))
            note_new = st.text_area("Catatan", max_chars=500)
            tag_new = st.text_input("Label", value="umum", max_chars=30)
            if st.form_submit_button("Hitung dan simpan", type="primary"):
                ok, data = api("POST", "/records", json={"model": m_new, "date": d_new.isoformat() if mode_new == "Pilih tanggal" else None,
                                                         "note": note_new, "tag": tag_new})
                if ok:
                    st.session_state["flash"] = f"Catatan #{data['id']} dibuat: {data['model']} memprediksi {rp(data['predicted_close'])} (acuan {data['as_of_date']})."
                    st.rerun()
                else:
                    st.error(data)

    with t_ubah:
        if not items:
            st.info("Belum ada catatan untuk diubah.")
        else:
            rid = st.selectbox("Pilih catatan", list(by_id), format_func=label, key="ubah_pilih")
            r = by_id[rid]
            with st.form(f"ubah_{rid}"):
                a, b = st.columns(2)
                m_up = a.selectbox("Model", info["models_available"], index=info["models_available"].index(r["model"]) if r["model"] in info["models_available"] else 0)
                d_up = b.date_input("Tanggal acuan", value=pd.Timestamp(r["as_of_date"]),
                                    min_value=pd.Timestamp(info["as_of_min"]), max_value=pd.Timestamp(info["as_of_max"]))
                note_up = st.text_area("Catatan", value=r["note"], max_chars=500)
                tag_up = st.text_input("Label", value=r["tag"], max_chars=30)
                st.caption("Mengubah model atau tanggal akan menghitung ulang prediksi. Mengubah catatan atau label tidak.")
                if st.form_submit_button("Simpan perubahan", type="primary"):
                    body = {"note": note_up, "tag": tag_up}
                    if m_up != r["model"]:
                        body["model"] = m_up
                    if d_up.isoformat() != r["as_of_date"]:
                        body["date"] = d_up.isoformat()
                    ok, data = api("PUT", f"/records/{rid}", json=body)
                    if ok:
                        st.session_state["flash"] = f"Catatan #{rid} diperbarui."
                        st.rerun()
                    else:
                        st.error(data)

    with t_hapus:
        if not items:
            st.info("Belum ada catatan untuk dihapus.")
        else:
            rid_d = st.selectbox("Pilih catatan", list(by_id), format_func=label, key="hapus_pilih")
            st.table(short(by_id[rid_d]))
            sure = st.checkbox("Saya yakin ingin menghapus catatan ini (tidak bisa dibatalkan)", key=f"yakin_{rid_d}")
            if st.button("Hapus catatan", disabled=not sure, type="primary"):
                ok, data = api("DELETE", f"/records/{rid_d}")
                if ok:
                    st.session_state["flash"] = f"Catatan #{rid_d} dihapus."
                    st.rerun()
                else:
                    st.error(data)

# ------------------------------------------------------------ PERBANDINGAN
elif page == "Perbandingan Model":
    page_header("Perbandingan model",
                f"Data uji: {info['n_test']} hari bursa ({info['test_start']} sampai {info['period'][1]}). Semua model memakai dataset, fitur, pembagian data, dan environment yang sama.")
    m = get("/metrics")
    if m["accuracy"] is None:
        st.warning("Hasil eksperimen belum tersedia. Jalankan eksperimen Docker dulu.")
        st.stop()
    tab_v, tab_t, tab_g = st.tabs(["Ringkasan visual", "Tabel lengkap", "Grafik dari eksperimen"])

    with tab_v:
        try:
            rdf = pd.DataFrame(get("/results"))
            naive = float(rdf.loc[rdf.Model == "NaiveBaseline", "RMSE"].iloc[0])
            ml = rdf[rdf.Model != "NaiveBaseline"].copy()
            cscale = alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))
            with st.container(border=True):
                st.markdown("**Akurasi: RMSE per model** (lebih kecil lebih baik)")
                bars = alt.Chart(ml).mark_bar().encode(
                    x=alt.X("Model:N", sort="y", title=None), y=alt.Y("RMSE:Q", title="RMSE (Rp)"),
                    color=alt.condition(alt.datum.RMSE > naive, alt.value(ORANGE), alt.value(BLUE)),
                    tooltip=["Model:N", alt.Tooltip("RMSE:Q", format=".1f"), alt.Tooltip("MAE:Q", format=".1f")])
                rule = alt.Chart(pd.DataFrame({"y": [naive]})).mark_rule(color=INK, strokeDash=[5, 4]).encode(y="y:Q")
                st.altair_chart((bars + rule).properties(height=300, width="container"))
                st.caption(f"Garis putus: baseline naif (RMSE {naive:.1f}). Oranye: lebih buruk dari baseline. Biru: setara atau lebih baik.")
            with st.container(border=True):
                st.markdown("**Trade-off: akurasi vs biaya komputasi** (pojok kiri-bawah = terbaik)")
                opt = {"Waktu training (detik)": "Train_s_mean", "Inferensi 1 sampel (ms)": "Infer_1sample_ms_mean", "Ukuran model (KB)": "Model_size_KB"}
                xlab = st.radio("Sumbu X", list(opt), horizontal=True)
                col = opt[xlab]
                base = alt.Chart(ml).encode(x=alt.X(f"{col}:Q", scale=alt.Scale(type="log"), title=xlab),
                                            y=alt.Y("RMSE:Q", scale=alt.Scale(type="log"), title="RMSE (Rp, skala log)"))
                dots = base.mark_circle(size=260, opacity=0.9).encode(color=alt.Color("Model:N", scale=cscale),
                                                                       tooltip=["Model:N", alt.Tooltip("RMSE:Q", format=".1f"), alt.Tooltip(f"{col}:Q", format=",.3f")])
                labels = base.mark_text(dy=-16, fontSize=12).encode(text="Model:N")
                rule2 = alt.Chart(pd.DataFrame({"y": [naive]})).mark_rule(color=GRAY, strokeDash=[5, 4]).encode(y="y:Q")
                st.altair_chart((dots + labels + rule2).properties(height=340, width="container"))
                st.caption("Kedua sumbu berskala log. Garis putus abu-abu: baseline naif.")
        except Exception as e:
            st.warning(f"Grafik interaktif tidak dapat dibuat: {e}")

    with tab_t:
        st.subheader("Akurasi")
        st.dataframe(tanpa_pm(pd.DataFrame(m["accuracy"])), hide_index=True)
        st.caption("Baris NaiveBaseline = tebakan \"besok = hari ini\". R² hampir selalu tinggi pada harga saham sehingga kurang informatif; "
                   "bandingkan MAE dan RMSE dengan baseline.")
        st.subheader("Biaya komputasi")
        st.dataframe(tanpa_pm(pd.DataFrame(m["compute"])), hide_index=True)
        st.caption("RAM training hanya perkiraan kasar. Ukuran model dan waktu lebih dapat diandalkan.")
        st.subheader("Apakah selisih terhadap baseline bermakna? (uji Diebold-Mariano)")
        st.dataframe(tanpa_pm(pd.DataFrame(m["significance"])), hide_index=True)
        st.subheader("Error menurut rentang harga")
        st.dataframe(tanpa_pm(pd.DataFrame(m["by_range"])), hide_index=True)

    with tab_g:
        shown = False
        for name, cap in [("fig_actual_vs_pred.png", "Aktual vs prediksi"), ("fig_tradeoff.png", "Trade-off akurasi dan biaya komputasi"),
                          ("fig_error_by_range.png", "Error per rentang harga")]:
            img = get_image(name)
            if img:
                st.image(img, caption=cap)
                shown = True
        if not shown:
            st.info("Gambar belum tersedia di folder results.")

# --------------------------------------------------------------- ENVIRONMENT
elif page == "Environment":
    page_header("Environment dan reproducibility", "Bukti bahwa eksperimen berjalan di lingkungan yang terkontrol.")
    a, b = st.columns(2)
    with a.container(border=True):
        st.subheader("Dataset (terkunci)")
        st.table(pd.DataFrame(
            {"Nilai": ["BBCA.JK (Yahoo Finance)", f"{info['period'][0]} s/d {info['period'][1]}", info["rows_clean"],
                       f"{info['n_train']} (sampai {info['train_end']})", f"{info['n_test']} (mulai {info['test_start']})",
                       len(info["features"]), f"{info['window_lstm']} hari", "Close hari bursa berikutnya"]},
            index=["Saham", "Periode", "Baris setelah pembersihan", "Train", "Test", "Jumlah fitur", "Window LSTM", "Target"]).astype(str))
    env = get("/environment")
    with b.container(border=True):
        st.subheader("Kontainer eksperimen")
        e = env["environment"]
        if e:
            st.table(pd.DataFrame({"Nilai": [e["python"], e["os"], e["cpu"], e["batas_cpu_kontainer"], e["batas_ram_kontainer_GB"],
                                              e["seed"], e["n_thread_model"], e["n_pengulangan"]]},
                                  index=["Python", "OS", "CPU", "Batas CPU kontainer", "Batas RAM kontainer (GB)", "Seed", "Thread per model", "Pengulangan ukur"]).astype(str))
            with st.expander("Versi library"):
                st.json(e["library"])
        else:
            st.warning("environment.json belum ada. Jalankan eksperimen Docker dulu.")
    if env["stage_times"]:
        with st.container(border=True):
            st.subheader("Waktu tiap tahap pipeline")
            st.dataframe(pd.DataFrame(env["stage_times"]), hide_index=True)
            st.caption("Puncak RAM kontainer bersifat kumulatif untuk seluruh pipeline, bukan per model.")

# -------------------------------------------------------------------- TENTANG
else:
    page_header("Tentang penelitian")
    st.markdown("""
**Pertanyaan penelitian.** Bagaimana perbandingan model machine learning (Linear Regression, SVR, Random Forest, XGBoost)
dan LSTM untuk memprediksi harga Close BBCA satu hari ke depan, ditinjau dari akurasi, biaya komputasi (training, inferensi,
ukuran model), dan reproducibility di Docker?

**Temuan utama.**
- Tidak ada model yang jelas mengalahkan baseline "besok = hari ini". SVR hanya sedikit lebih baik, dan tidak lagi signifikan setelah koreksi uji berganda.
- Random Forest dan XGBoost gagal ketika harga melewati rentang data latih (keterbatasan ekstrapolasi model pohon).
- LSTM lebih buruk dari baseline dengan biaya komputasi tertinggi.
- Akurasi arah semua model sekitar 50%, setara tebakan acak.

**Batasan.** Satu saham, satu periode uji, target berupa harga (bukan perubahan harga).

**Peringatan.** Aplikasi ini untuk penelitian dan pembelajaran. Hasilnya bukan saran investasi.
""")

st.markdown('<div class="disc">Aplikasi ini untuk penelitian dan pembelajaran. Hasil prediksi bukan saran investasi.</div>', unsafe_allow_html=True)