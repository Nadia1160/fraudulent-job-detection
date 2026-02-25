# === Builder Stage ===
FROM python:3.9-slim AS builder

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy ONLY requirements.txt (dev-requirements.txt removed)
COPY requirements.txt ./

# Create a virtual environment and install all dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:"

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch separately with index URL (avoid timeout)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# === Final Stage ===
FROM python:3.9-slim

# Create non-root user
RUN adduser --disabled-password --gecos '' --uid 1001 appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Set working directory and ownership
WORKDIR /app
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PATH="/opt/venv/bin:" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH="/app/models/best_model.pkl"

# Switch to non-root user
USER appuser

EXPOSE 8000
CMD ["python", "-c", "print('Fraudulent Job Detection container is ready. Run your scripts manually.')"]
