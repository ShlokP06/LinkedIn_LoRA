FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi[standard]" \
    httpx \
    groq \
    python-dotenv \
    modal

COPY deploy/ deploy/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "deploy.api:fastapi_app", "--host", "0.0.0.0", "--port", "8000"]
