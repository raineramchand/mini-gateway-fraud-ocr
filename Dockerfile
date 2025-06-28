FROM python:3.10-slim

# Prevent .pyc files and unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV/EasyOCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy application code
COPY . .

EXPOSE 8000

# Launch the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
