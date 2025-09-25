# Dockerfile for OpenPI UR5e training/inference on vast.ai
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \
    wget \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-distutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3.10 -> python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/python3.10 /usr/bin/python3

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Clone repository with submodules
RUN git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git .

# Set environment variables as per README setup
ENV GIT_LFS_SKIP_SMUDGE=1
ENV PYTHONPATH=/workspace/src

# Install Python dependencies using uv (matches README instructions)
RUN uv sync --extra-index-url https://download.pytorch.org/whl/cu118 && \
    uv pip install -e .

# Default command
CMD ["bash"]
