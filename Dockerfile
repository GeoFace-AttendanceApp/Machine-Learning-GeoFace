# Gunakan image Python sebagai base image
FROM python:3.11.2

# Set direktori kerja dalam container
WORKDIR /app

# Copy semua file ke dalam container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Ekspos port 5000
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "app.py"]