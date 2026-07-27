#!/bin/bash

# LGM Win 3090 - Run Script
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

# Default values
INPUT_DIR="${1:-./input}"
OUTPUT_DIR="${2:-./output}"

# Show usage information
usage() {
    echo "Usage: $0 [INPUT_DIR] [OUTPUT_DIR]"
    echo ""
    echo "Pipeline:"
    echo "  [Multiple Images] -> [COLMAP] -> [Gaussian Splatting] -> [Meshing] -> [GLB Output]"
    echo ""
    echo "Arguments:"
    echo "  INPUT_DIR    Directory containing car images (default: ./input)"
    echo "  OUTPUT_DIR   Directory for output 3D models (default: ./output)"
    echo ""
    echo "Examples:"
    echo "  $0                                    Use default directories"
    echo "  $0 /path/to/images /path/to/output    Use custom directories"
    echo ""
    echo "Image format requirements:"
    echo "  - Place multiple images of the car in INPUT_DIR"
    echo "  - Supported formats: .jpg, .jpeg, .png"
    echo "  - Images should show the car from different angles (front, back, sides, etc.)"
}

# Check if input directory exists
check_input() {
    if [ ! -d "$INPUT_DIR" ]; then
        print_warning "Input directory '$INPUT_DIR' does not exist. Creating it..."
        mkdir -p "$INPUT_DIR"
    fi
    
    # Count images in input directory
    local img_count
    img_count=$(find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
    
    if [ "$img_count" -eq 0 ]; then
        print_warning "No images found in '$INPUT_DIR'"
        print_info "Please add car images to the input directory"
        print_info "Supported formats: .jpg, .jpeg, .png"
    else
        print_success "Found $img_count image(s) in '$INPUT_DIR'"
    fi
}

# Main function
main() {
    echo "============================================"
    echo "  LGM Win 3090 - 3D Car Model Generator"
    echo "============================================"
    echo ""
    echo "  Pipeline: COLMAP + Gaussian Splatting + Meshing"
    echo "  GPU: RTX 3090 (24GB VRAM)"
    echo ""
    
    check_input
    
    print_info "Input directory: $INPUT_DIR"
    print_info "Output directory: $OUTPUT_DIR"
    echo ""
    
    # List images
    if [ "$(find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)" -gt 0 ]; then
        print_info "Images to process:"
        find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | while read -r img; do
            echo "  - $(basename "$img")"
        done
        echo ""
    fi
    
    # Run docker compose
    print_info "Starting 3D car modeling pipeline..."
    echo ""
    echo "  Step 1: Preprocessing (Image normalization, masking)"
    echo "  Step 2: COLMAP (Camera estimation, Sparse point cloud)"
    echo "  Step 3: Dense Reconstruction (Multi-view Stereo)"
    echo "  Step 4: Gaussian Splatting (High-quality 3D reconstruction)"
    echo "  Step 5: Meshing (Poisson / Instant Meshes)"
    echo "  Step 6: Texture Baking (UV, textures, materials)"
    echo "  Step 7: Export (GLB, OBJ, PLY)"
    echo ""
    
    docker compose run --rm -v "$INPUT_DIR:/workspace/input:ro" -v "$OUTPUT_DIR:/workspace/output" car-model
}

# Run main function
main "$@"
