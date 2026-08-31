FROM python:3.11-slim

# tesseract-ocr is required by pytesseract for the OCR pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Shell form (not exec form) so $PORT is actually substituted by the shell.
# Render sets PORT at runtime and expects the app to bind to it; local
# docker-compose doesn't set PORT, so it falls back to 8000 unchanged.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}