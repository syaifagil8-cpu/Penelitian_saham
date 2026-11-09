"""
Jalankan seluruh pipeline di dalam kontainer: Tahap 2 sampai 8, lalu catat environment.
Pemakaian:  python src/run_all.py [tag]
Kalau 'tag' diberikan (mis. run1), folder results/ disalin ke runs/<tag>/.
"""
import csv, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError

STAGES = ["preprocessing", "features", "prepare_data", "train_ml", "train_lstm", "evaluate"]
sys.path.insert(0, os.path.dirname(__file__))
import config as C


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def container_peak_mb():
    v = read("/sys/fs/cgroup/memory.peak") or read("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
    return round(int(v) / 1e6, 1) if v and v.isdigit() else None


def cpu_limit():
    v = read("/sys/fs/cgroup/cpu.max")                     # cgroup v2: "kuota periode" atau "max periode"
    if v and not v.startswith("max"):
        q, p = v.split(); return round(int(q) / int(p), 2)
    return None


def mem_limit_gb():
    v = read("/sys/fs/cgroup/memory.max") or read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    return round(int(v) / 1e9, 2) if v and v.isdigit() and int(v) < 2 ** 60 else None


def cpu_model():
    for line in (read("/proc/cpuinfo") or "").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def libs():
    out = {}
    for p in ["numpy", "pandas", "scikit-learn", "joblib", "psutil", "xgboost", "tensorflow", "matplotlib"]:
        try: out[p] = version(p)
        except PackageNotFoundError: out[p] = None
    return out


os.makedirs("results", exist_ok=True)
env = {"waktu_mulai": datetime.now().isoformat(timespec="seconds"),
       "di_dalam_docker": os.path.exists("/.dockerenv"),
       "python": platform.python_version(), "os": platform.platform(),
       "cpu": cpu_model(), "cpu_logis_terlihat": os.cpu_count(),
       "batas_cpu_kontainer": cpu_limit(), "batas_ram_kontainer_GB": mem_limit_gb(),
       "seed": C.SEED, "n_thread_model": C.N_THREADS, "n_pengulangan": C.N_REPEATS,
       "variabel_env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "TF_ENABLE_ONEDNN_OPTS", "PYTHONHASHSEED"]},
       "library": libs()}
json.dump(env, open("results/environment.json", "w"), indent=2, ensure_ascii=False)

rows = []
t_all = time.perf_counter()
for st in STAGES:
    print(f"\n########## {st} ##########", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run([sys.executable, f"src/{st}.py"]).returncode
    rows.append({"tahap": st, "detik": round(time.perf_counter() - t0, 1),
                 "puncak_RAM_kontainer_MB_kumulatif": container_peak_mb(), "kode_keluar": rc})
    if rc != 0:
        print(f"GAGAL di tahap {st} (kode {rc})", file=sys.stderr)
        break
with open("results/stage_times.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

total = round(time.perf_counter() - t_all, 1)
print(f"\nSelesai dalam {total} detik. Puncak RAM kontainer: {container_peak_mb()} MB (kumulatif)")
if len(sys.argv) > 1 and all(r["kode_keluar"] == 0 for r in rows):
    dst = os.path.join("runs", sys.argv[1])
    shutil.copytree("results", dst, dirs_exist_ok=True)
    print(f"Hasil disalin ke {dst}/")
sys.exit(0 if all(r["kode_keluar"] == 0 for r in rows) else 1)