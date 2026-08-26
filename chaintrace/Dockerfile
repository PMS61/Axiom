# ChainTrace POC - one image, two entry points (dashboard and API).
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bundled sample data only - no API key, no network needed.
ENV CHAINTRACE_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8501 8000

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
