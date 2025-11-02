FROM python:3.11-slim AS deps-train
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y gcc postgresql-client ca-certificates

# Copy centralized requirements and service-specific requirements
COPY services/requirements/ ./requirements/
COPY services/training/requirements.txt .

# Using locked versions for reproducibility
# Resolve all requirements by combining base + ml + data + service-specific
RUN pip install --upgrade pip && \
    { cat requirements/base.txt requirements/ml.txt requirements/data.txt; \
    grep -v '^#' requirements.txt | grep -v '^-r' | grep -v '^$'; } | \
    grep -v '^-r' | grep -v '^#' | grep -v '^$' > /tmp/all_reqs.txt && \
    pip install --user --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r /tmp/all_reqs.txt

FROM python:3.11-slim
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y postgresql-client curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=deps-train /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser services/training/src/ ./src/

# Copy common module
COPY --chown=appuser:appuser services/common/ ./common/

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/app

USER appuser
EXPOSE 8002

# Health check - check if celery worker is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD celery -A src.main:celery_app inspect ping -d celery@$HOSTNAME || exit 1

# Start Celery worker for training tasks
CMD ["celery", "-A", "src.main:celery_app", "worker", "--loglevel=INFO", "--concurrency=2", "--time-limit=21600", "--soft-time-limit=19800"]
