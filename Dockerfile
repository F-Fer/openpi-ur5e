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
    openssh-server \
    sudo \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

apt-get install -y \
    ffmpeg \
    libavcodec-extra \

# Symlink python3.10 -> python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Clone repository with submodules
RUN git clone --recurse-submodules https://github.com/F-Fer/openpi-ur5e.git .

# Set environment variables as per README setup
ENV GIT_LFS_SKIP_SMUDGE=1
ENV PYTHONPATH=/workspace/src

# Install Python dependencies using uv (matches README instructions)
RUN uv sync --extra-index-url https://download.pytorch.org/whl/cu118 && \
    uv pip install -e .

# Install Jupyter and additional packages
RUN uv pip install jupyter jupyterlab ipywidgets

# Setup SSH server
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Create startup script
RUN echo '#!/bin/bash\n\
# Start SSH server\n\
service ssh start\n\
\n\
# Change to workspace directory and start Jupyter Lab using uv\n\
cd /workspace\n\
uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password="" &\n\
\n\
# Keep container running\n\
sleep infinity' > /start.sh && chmod +x /start.sh

# Expose ports
EXPOSE 22 8888

# Default command
CMD ["/start.sh"]
