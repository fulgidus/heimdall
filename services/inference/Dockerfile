FROM python:3.11-slim AS builder
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y gcc postgresql-client ca-certificates

# Copy centralized requirements and service-specific requirements
COPY services/requirements/ ./requirements/
COPY services/inference/requirements.txt .

# Using locked versions for reproducibility
# Install from requirements that reference centralized dependencies
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y postgresql-client curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser services/inference/src/ ./src/

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"]
