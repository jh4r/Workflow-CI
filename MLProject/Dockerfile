# Gunakan Python versi 3.12
FROM python:3.12.7-slim

# Set folder kerja di dalam Docker
WORKDIR /app

# Copy semua file dari folder MLProject ke dalam Docker
COPY . /app

# Install library
RUN pip install --no-cache-dir mlflow scikit-learn pandas numpy matplotlib seaborn

# Perintah yang dijalankan saat Docker nyala
CMD ["python", "modelling.py"]
