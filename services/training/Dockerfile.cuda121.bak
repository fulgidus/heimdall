# Multi-stage build for training service with GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS deps-train
WORKDIR /build

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    gcc \
    postgresql-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

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

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
WORKDIR /app

# Install Python 3.11 and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=deps-train /root/.local /home/appuser/.local

# Create CuPy cache directory with proper permissions
RUN mkdir -p /home/appuser/.cupy && chown -R appuser:appuser /home/appuser/.cupy

# Copy application code
COPY --chown=appuser:appuser services/training/src/ ./src/
COPY --chown=appuser:appuser services/training/entrypoint.py ./entrypoint.py

# Copy common module
COPY --chown=appuser:appuser services/common/ ./common/

# Copy backend modules (needed for storage access)
COPY --chown=appuser:appuser services/backend/src/ ./backend/src/

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER appuser
EXPOSE 8002

# Health check - check if API server is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Start both FastAPI API and Celery worker via entrypoint
CMD ["python", "entrypoint.py"]
