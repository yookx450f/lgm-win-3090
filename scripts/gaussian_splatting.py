#!/usr/bin/env python3
"""
Gaussian Splatting Script for Car 3D Modeling
- 3D Gaussian distribution optimization
- High-quality rendering
- High-precision texture representation
- Reflection and gloss reproduction
"""

import argparse
import os
import sys
import subprocess
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Gaussian Splatting for car 3D modeling')
    parser.add_argument('--source', type=str, required=True,
                        help='Input directory from COLMAP (colmap output)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for Gaussian Splatting results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--iterations', type=int, default=30000,
                        help='Number of training iterations (default: 30000)')
    parser.add_argument('--resolution', type=int, default=4,
                        help='Image resolution scaling factor (default: 4)')
    parser.add_argument('--sh_degree', type=int, default=3,
                        help='Spherical harmonics degree (default: 3)')
    parser.add_argument('--white_background', type=bool, default=False,
                        help='Use white background (default: False)')
    return parser.parse_args()


def prepare_colmap_data(source_path, output_path):
    """Prepare COLMAP data for Gaussian Splatting"""
    print("[Prepare COLMAP Data] Starting...")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy COLMAP data
    models_path = os.path.join(source_path, 'sparse', '0')
    if os.path.exists(models_path):
        # Create expected directory structure
        expected_path = os.path.join(output_path, 'sparse', '0')
        os.makedirs(expected_path, exist_ok=True)
        
        # Copy point cloud file
        ply_file = os.path.join(models_path, 'points3D.ply')
        if os.path.exists(ply_file):
            subprocess.run(['cp', ply_file, os.path.join(expected_path, 'points3D.ply')],
                          check=False, capture_output=True)
        
        # Copy camera parameters
        cameras_file = os.path.join(models_path, 'cameras.bin')
        if os.path.exists(cameras_file):
            subprocess.run(['cp', cameras_file, os.path.join(expected_path, 'cameras.bin')],
                          check=False, capture_output=True)
        
        # Copy image list
        images_file = os.path.join(models_path, 'images.bin')
        if os.path.exists(images_file):
            subprocess.run(['cp', images_file, os.path.join(expected_path, 'images.bin')],
                          check=False, capture_output=True)
        
        print("[Prepare COLMAP Data] Complete!")
        return True
    
    print("[Error] COLMAP data not found")
    return False


def train_gaussian_splatting(source_path, output_path, device='cuda',
                              iterations=30000, resolution=4, sh_degree=3,
                              white_background=False):
    """Train 3D Gaussian Splatting"""
    print("[Gaussian Splatting Training] Starting...")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")
    print(f"  Device: {device}")
    print(f"  Iterations: {iterations}")
    
    # Check if gaussian-splatting repository exists
    gs_path = '/workspace/gaussian-splatting'
    training_script = os.path.join(gs_path, 'train.py')
    
    if not os.path.exists(training_script):
        print("[Error] Gaussian Splatting repository not found")
        print("[Info] Please clone: https://github.com/graphdeco-inria/gaussian-splatting")
        return False
    
    # Build training command
    cmd = [
        'python3', training_script,
        '--source', source_path,
        '--output_path', output_path,
        '--device', device,
        '--iterations', str(iterations),
        '--resolution', str(resolution),
        '--sh_degree', str(sh_degree),
    ]
    
    if white_background:
        cmd.extend(['--white_background', '1'])
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Gaussian Splatting Training] Complete!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Gaussian Splatting training failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def render_views(source_path, output_path, render_output_path):
    """Render multiple views from Gaussian Splatting"""
    print("[Render Views] Starting...")
    print(f"  Source: {source_path}")
    print(f"  Output: {render_output_path}")
    
    # Check if render script exists
    gs_path = '/workspace/gaussian-splatting'
    render_script = os.path.join(gs_path, 'render.py')
    
    if not os.path.exists(render_script):
        print("[Warning] Render script not found, skipping")
        return False
    
    # Create render output directory
    os.makedirs(render_output_path, exist_ok=True)
    
    # Build render command
    cmd = [
        'python3', render_script,
        '--source', source_path,
        '--output_path', render_output_path,
        '--skip_train', '1',
        '--skip_test', '1'
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Render Views] Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Warning] Rendering failed: {e}")
        return False


def export_gaussian_splatting(source_path, output_path):
    """Export Gaussian Splatting results"""
    print("[Export Gaussian Splatting] Starting...")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Find trained model
    model_path = os.path.join(source_path, 'train', 'final', 'point_cloud.ply')
    
    if os.path.exists(model_path):
        # Copy the final point cloud
        output_ply = os.path.join(output_path, 'gaussian_splat.ply')
        subprocess.run(['cp', model_path, output_ply],
                      check=False, capture_output=True)
        print(f"  Exported: {output_ply}")
    
    # Find rendered images
    rendered_dir = os.path.join(source_path, 'train', 'final', 'render')
    if os.path.exists(rendered_dir):
        output_images = os.path.join(output_path, 'renders')
        os.makedirs(output_images, exist_ok=True)
        subprocess.run(['cp', '-r', f'{rendered_dir}/*', output_images],
                      check=False, capture_output=True)
        print(f"  Exported renders to: {output_images}")
    
    print("[Export Gaussian Splatting] Complete!")
    return True


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  3D Gaussian Splatting for Car 3D Modeling")
    print("=" * 60)
    print("")
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output_path}")
    print("")
    
    # Step 1: Prepare COLMAP data
    print("Step 1: Prepare COLMAP Data")
    print("-" * 40)
    if not prepare_colmap_data(args.source, args.output_path):
        print("[Error] Failed to prepare COLMAP data")
        sys.exit(1)
    print("")
    
    # Step 2: Train Gaussian Splatting
    print("Step 2: Train Gaussian Splatting")
    print("-" * 40)
    trained_path = os.path.join(args.output_path, 'train')
    if not train_gaussian_splatting(
            args.output_path, trained_path,
            args.device, args.iterations, args.resolution,
            args.sh_degree, args.white_background):
        print("[Error] Gaussian Splatting training failed")
        sys.exit(1)
    print("")
    
    # Step 3: Render views
    print("Step 3: Render Views")
    print("-" * 40)
    render_output_path = os.path.join(args.output_path, 'renders')
    render_views(args.output_path, args.output_path, render_output_path)
    print("")
    
    # Step 4: Export results
    print("Step 4: Export Results")
    print("-" * 40)
    export_gaussian_splatting(trained_path, args.output_path)
    print("")
    
    print("=" * 60)
    print("  Gaussian Splatting Processing Complete!")
    print("=" * 60)
    print("")
    print(f"  Output: {args.output_path}")


if __name__ == '__main__':
    main()
