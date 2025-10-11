# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WANDB_SILENT=false
ENV WANDB_CONSOLE=wrap
ENV UV_CACHE_DIR=/opt/uv-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install uv

# Create virtual env
RUN uv venv

# Install all python packages
RUN uv sync

# Copy the application code
COPY . .

# Create output directory for model artifacts
RUN mkdir -p /app/food101-vit-model

# Set the default command to run train.py with activated virtual environment
CMD ["/bin/bash", "-c", "source /app/.venv/bin/activate && python train.py"]
