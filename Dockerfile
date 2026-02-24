# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ──────────────────────────────────────────────────
COPY . .

# ── Expose Streamlit port ────────────────────────────────────────────────────
EXPOSE 8501

# ── Run Streamlit ────────────────────────────────────────────────────────────
CMD ["python", "-m", "streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]
