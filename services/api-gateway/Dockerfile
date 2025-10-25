FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y gcc postgresql-client ca-certificates
COPY api-gateway/requirements.txt .
RUN pip install --upgrade pip && pip install --user --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y postgresql-client curl && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser api-gateway/src/ ./src/
# Copy common auth module
COPY --chown=appuser:appuser common/auth ./auth/
ENV PATH=/home/appuser/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
