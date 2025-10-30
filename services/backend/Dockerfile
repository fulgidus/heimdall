FROM python:3.11-slim AS builder
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y gcc postgresql-client ca-certificates

# Copy centralized requirements and service-specific requirements
COPY services/requirements/ ./requirements/
COPY services/backend/requirements.txt .

# Using locked versions for reproducibility
# Install from centralized lock files first, then service-specific
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

# Copy shared common module
COPY --chown=appuser:appuser services/common/ ./common/

# Copy application code
COPY --chown=appuser:appuser services/backend/src/ ./src/
COPY --chown=appuser:appuser services/backend/entrypoint.py ./entrypoint.py

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/ready || exit 1

CMD ["python", "entrypoint.py"]
