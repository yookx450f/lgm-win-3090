#!/bin/bash

# LGM Win 3090 - Build Script
# Pipeline: COLMAP + 3D Gaussian Splatting + Meshing
# For car 3D modeling with RTX 3090

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed: $(docker --version)"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    if command -v docker compose &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    print_success "Docker Compose is available"
}

# Check NVIDIA Container Toolkit
check_nvidia() {
    if ! docker info 2>&1 | grep -q "NVIDIA"; then
        print_warning "NVIDIA Container Toolkit may not be configured properly."
        print_warning "Please ensure nvidia-container-toolkit is installed and configured."
    else
        print_success "NVIDIA Container Toolkit is configured"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_success "GPU detected: $gpu_name"
    fi
}

# Create required directories
setup_directories() {
    print_info "Setting up directories..."
    mkdir -p input output cache workspace scripts
    print_success "Directories created: input/, output/, cache/, workspace/, scripts/"
}

# Build the Docker image
build_image() {
    print_info "Building Docker image..."
    print_info "This may take a while as it downloads PyTorch, COLMAP, and dependencies..."
    
    $COMPOSE_CMD build --progress=plain
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully: lgm-car-model:latest"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Show usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  build       Build the Docker image"
    echo "  run         Run the 3D car modeling pipeline"
    echo "  shell       Start an interactive shell in the container"
    echo "  stop        Stop running containers"
    echo "  clean       Remove containers and images"
    echo "  check       Check environment requirements"
    echo "  help        Show this help message"
    echo ""
    echo "Pipeline:"
    echo "  [Multiple Images] -> [COLMAP] -> [Gaussian Splatting] -> [Meshing] -> [GLB Output]"
    echo ""
    echo "Examples:"
    echo "  $0 build    Build the Docker image"
    echo "  $0 run      Run 3D car modeling pipeline"
    echo "  $0 shell    Start interactive shell"
}

# Main function
main() {
    local command="${1:-help}"
    
    echo "============================================"
    echo "  LGM Win 3090 - Build System"
    echo "============================================"
    echo ""
    echo "  Pipeline: COLMAP + Gaussian Splatting + Meshing"
    echo "  GPU: RTX 3090 (24GB VRAM)"
    echo ""
    
    case "$command" in
        build)
            check_docker
            check_docker_compose
            check_nvidia
            setup_directories
            build_image
            ;;
        run)
            check_docker
            check_docker_compose
            setup_directories
            print_info "Starting 3D car modeling pipeline..."
            $COMPOSE_CMD up
            ;;
        shell)
            check_docker
            check_docker_compose
            setup_directories
            print_info "Starting interactive shell..."
            $COMPOSE_CMD run --rm car-model-shell
            ;;
        stop)
            check_docker
            check_docker_compose
            print_info "Stopping containers..."
            $COMPOSE_CMD down
            ;;
        clean)
            check_docker
            check_docker_compose
            print_info "Removing containers and images..."
            $COMPOSE_CMD down --rmi all --volumes --remove-orphans
            ;;
        check)
            check_docker
            check_docker_compose
            check_nvidia
            ;;
        help|*)
            usage
            ;;
    esac
}

# Run main function
main "$@"
