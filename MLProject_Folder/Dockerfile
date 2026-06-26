FROM python:3.10-slim

WORKDIR /app

# Install sistem dependensi mendasar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin kebutuhan dependensi dan install
RUN pip install --no-cache-dir mlflow==2.14.1 pandas==2.2.2 scikit-learn==1.5.0 dagshub>=0.30.0

# Salin seluruh kode program dan dataset
COPY . /app

# Port standar untuk serving MLflow model jika diperlukan nanti
EXPOSE 5001

# Saat kontainer dijalankan, ia akan langsung memicu re-training model
CMD ["python", "modelling.py"]