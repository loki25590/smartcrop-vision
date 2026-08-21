FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.predict:app --host 0.0.0.0 --port ${PORT:-10000}"]