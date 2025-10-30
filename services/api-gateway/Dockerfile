FROM python:3.11-slim AS deps-api
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y gcc postgresql-client ca-certificates

# Copy centralized requirements and service-specific requirements
COPY services/requirements/ ./requirements/
COPY services/api-gateway/requirements.txt .

# Using locked versions for reproducibility
# Resolve all requirements by combining base + api + service-specific
RUN pip install --upgrade pip && \
    { cat requirements/base.txt requirements/api.txt; \
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
COPY --from=deps-api /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser services/api-gateway/src/ ./src/

# Copy common auth module
COPY --chown=appuser:appuser services/common/auth ./auth/

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/app

USER appuser
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
