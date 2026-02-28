FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY contracts/ contracts/
COPY services/ services/
COPY eval/ eval/
COPY pyproject.toml .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/admin/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "services.api.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
