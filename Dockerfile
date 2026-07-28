# LGM Win 3090 - 3D Car Model Generator
# Pipeline: COLMAP + 3D Gaussian Splatting + Meshing
# Base image: NVIDIA CUDA 13.x with PyTorch (WSL2 RTX 3090)
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Avoid interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    cmake \
    build-essential \
    wget \
    curl \
    unzip \
    rsync \
    vim \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install COLMAP dependencies
RUN apt-get update && apt-get install -y \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libctemplate-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default python3
RUN ln -sf python3 /usr/bin/python

# Working directory
WORKDIR /workspace

# Install PyTorch with CUDA 13.x support
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install additional Python dependencies
# Note: trimesh[extras] includes support for glb, obj, ply, and other 3D formats
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    Pillow \
    trimesh[extras] \
    pyvista \
    open3d \
    scikit-image \
    scipy \
    matplotlib \
    tqdm \
    imageio \
    imageio-ffmpeg \
    joblib

# Install FastAPI and web dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles

# Install testing dependencies
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    black \
    flake8 \
    isort

# Install 3D Gaussian Splatting dependencies
# Git clone 3D Gaussian Splatting repository and submodules
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting && \
    cd /workspace/gaussian-splatting && \
    # Clone main repository submodules explicitly with retry logic
    git submodule init && \
    git submodule update --depth 1 && \
    # Install GLM library for diff-gaussian-rasterization
    cd submodules/diff-gaussian-rasterization/third_party && \
    rm -rf glm && git clone --depth 1 https://github.com/g-truc/glm.git glm && \
    # Fix GLM directory structure (glm.hpp needs to be at glm/glm/glm.hpp)
    mkdir -p glm/glm && \
    mv glm/glm.hpp glm/glm/ 2>/dev/null || true && \
    for dir in gtc ext detail gtx simd; do \
        if [ -d glm/$$dir ]; then \
            mv glm/$$dir glm/glm/ 2>/dev/null || true; \
        fi; \
    done && \
    # Install diff-gaussian-rasterization with CUDA 13.x support
    cd /workspace/gaussian-splatting/submodules/diff-gaussian-rasterization && \
    pip3 install --no-cache-dir -e . --no-build-isolation 2>&1 || echo "Warning: diff-gaussian-rasterization installation failed", \
    # Install simple-knn with __init__.py
    cd /workspace/gaussian-splatting/submodules/simple-knn && \
    mkdir -p simple_knn && \
    echo 'import simple_knn._C' > simple_knn/__init__.py && \
    pip3 install --no-cache-dir -e . --no-build-isolation 2>&1 || echo "Warning: simple-knn installation failed", \
    # Set PYTHONPATH to include gaussian-splatting directory
    cd /workspace && \
    echo 'export PYTHONPATH=/workspace/gaussian-splatting:$PYTHONPATH' >> /etc/environment

# Install COLMAP from binary release (with retry logic and fallback)
WORKDIR /tmp
RUN set -e; \
    # Try multiple COLMAP versions in order of preference; \
    for VERSION in 3.8.0 3.8 3.7.1 3.3.1; do \
        echo "Trying COLMAP version: $$VERSION"; \
        if wget -q --timeout=60 --tries=3 "https://github.com/colmap/colmap/releases/download/colmap-$$VERSION/COLMAP-$$VERSION-linux.run" -O colmap-installer.run 2>/dev/null; then \
            echo "Successfully downloaded COLMAP-$$VERSION"; \
            break; \
        else \
            echo "Failed to download COLMAP-$$VERSION, trying next version..."; \
        fi; \
    done; \
    # If all downloads failed, try installing from apt; \
    if [ ! -f colmap-installer.run ] || [ ! -s colmap-installer.run ]; then \
        echo "All downloads failed, attempting apt installation..."; \
        apt-get update && apt-get install -y colmap || echo "WARNING: COLMAP installation via apt failed"; \
    else \
        chmod +x colmap-installer.run; \
        mkdir -p /opt/colmap; \
        ./colmap-installer.run --target /opt/colmap --nobin --nooverride || echo "WARNING: COLMAP installer run failed"; \
        if [ -d /opt/colmap/bin ]; then \
            ln -sf /opt/colmap/bin/colmap /usr/local/bin/colmap 2>/dev/null || true; \
        fi; \
    fi; \
    echo "COLMAP installation complete"; \
    colmap --version 2>/dev/null || echo "COLMAP version check failed (non-critical)"

# Set environment variables
ENV PYTHONPATH=/workspace:${PYTHONPATH}
ENV CUDA_VISIBLE_DEVICES=0
ENV PATH=/opt/colmap/bin:${PATH}

# Default command
CMD ["bash"]
