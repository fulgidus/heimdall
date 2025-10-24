FROM python:3.11-slim as builder
WORKDIR /build
RUN apt-get update && apt-get install -y gcc postgresql-client ca-certificates
COPY rf-acquisition/requirements.txt .
RUN pip install --upgrade pip && pip install --user --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser rf-acquisition/src/ ./src/
COPY --chown=appuser:appuser rf-acquisition/entrypoint.py ./entrypoint.py
ENV PATH=/home/appuser/.local/bin:$PATH PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 CMD curl -f http://localhost:8001/health || exit 1
CMD ["python", "entrypoint.py"]
