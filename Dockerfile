# LGM Win 3090 - 3D Car Model Generator
# Pipeline: COLMAP + 3D Gaussian Splatting + Meshing
# Base image: NVIDIA CUDA 12.x with PyTorch
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

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

# Install PyTorch with CUDA 12.x support
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu121

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

# Install 3D Gaussian Splatting dependencies
# Git clone 3D Gaussian Splatting repository
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting 2>/dev/null || true

# Install COLMAP from source
WORKDIR /tmp
RUN wget -q https://github.com/colmap/colmap/releases/download/colmap-3.8.0/COLMAP-3.8.0-linux.run -O colmap-installer.run && \
    chmod +x colmap-installer.run && \
    ./colmap-installer.run --target /opt/colmap --nobin --nooverride && \
    ln -s /opt/colmap/bin/colmap /usr/local/bin/colmap

# Set environment variables
ENV PYTHONPATH=/workspace:${PYTHONPATH}
ENV CUDA_VISIBLE_DEVICES=0
ENV PATH=/opt/colmap/bin:${PATH}

# Default command
CMD ["bash"]
