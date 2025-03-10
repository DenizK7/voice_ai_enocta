FROM python:3.10-slim

WORKDIR /app

# Sistem paketlerini güncelle ve gcc gibi derleyicileri yükle (gerekiyorsa)
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Requirements dosyasını kopyala ve bağımlılıkları yükle
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Gunicorn ile uygulamayı başlat
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.main:app"]
