"""Buat requirements.txt dari versi library yang sedang terpasang di venv-mu
(hanya library utama, supaya cocok untuk Linux di dalam Docker)."""
import sys
from importlib.metadata import version, PackageNotFoundError
PKGS = ["numpy", "pandas", "scikit-learn", "joblib", "psutil", "xgboost", "tensorflow", "matplotlib"]
lines = []
for p in PKGS:
    try:
        lines.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print(f"PERINGATAN: {p} tidak terpasang di venv ini, dilewati.")
open("requirements.txt", "w").write("\n".join(lines) + "\n")
print("requirements.txt ditulis:\n" + "\n".join(lines))
print(f"\nPython lokal: {sys.version_info.major}.{sys.version_info.minor}"
      "  -> isi PYTHON_VERSION di docker-compose.yml dengan angka ini")