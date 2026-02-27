FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY contracts/ contracts/
COPY services/ services/
COPY pyproject.toml .

EXPOSE 8000

CMD ["uvicorn", "services.api.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
