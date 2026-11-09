# Environment eksperimen terstandarisasi. Versi library dikunci lewat requirements.txt.
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Satu thread untuk semua library numerik + seed hash tetap -> hasil dan waktu konsisten
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

# libgomp1 dibutuhkan XGBoost dan scikit-learn di image slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
CMD ["python", "src/run_all.py"]