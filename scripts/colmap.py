#!/usr/bin/env python3
"""
COLMAP Script for Car 3D Modeling
- Structure-from-Motion (SfM)
- Camera parameter estimation
- Sparse point cloud generation
- Feature detection and matching
"""

import argparse
import os
import sys
import subprocess
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='COLMAP for car 3D modeling')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Input directory containing preprocessed images')
    parser.add_argument('--database_path', type=str, required=True,
                        help='Output path for COLMAP database')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for COLMAP results (sparse/reconX)')
    parser.add_argument('--feature_extractor', type=str, default='exhaustive',
                        choices=['exhaustive', 'sequential', 'vocab_tree'],
                        help='Feature extraction method (default: exhaustive)')
    parser.add_argument('--preprocess_scale', type=float, default=0,
                        help='Scale of preprocessing (0 = original)')
    parser.add_argument('--preprocess_focal', type=str, default='default',
                        choices=['default', 'none', 'average'],
                        help='Focal length preprocessing (default: default)')
    return parser.parse_args()


def run_feature_extractor(image_path, database_path, feature_extractor='exhaustive',
                          preprocess_scale=0, preprocess_focal='default'):
    """Run COLMAP feature extractor"""
    print("[COLMAP Feature Extractor] Starting...")
    print(f"  Image path: {image_path}")
    print(f"  Database path: {database_path}")
    print(f"  Feature extraction: {feature_extractor}")
    
    # Create output directories
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    
    # Build COLMAP command
    cmd = [
        'colmap', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_path,
        '--ImageReader.single_camera', '1'
    ]
    
    # Feature extraction settings for cars
    if feature_extractor == 'exhaustive':
        cmd.extend([
            '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
            '--ImageReader.camera_params', '0.0,0.0,0.0',
            '--FeatureExtractor.max_num_features', '16384',
            '--FeatureExtractor.rescale_factor_size', str(preprocess_scale),
            '--FeatureExtractor.guided_matching', '1'
        ])
    elif feature_extractor == 'sequential':
        cmd.extend([
            '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
            '--ImageReader.camera_params', '0.0,0.0,0.0',
            '--FeatureExtractor.max_num_features', '16384',
            '--FeatureExtractor.sequence_matcher', '1'
        ])
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Feature Extractor] Complete!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Feature extraction failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def run_feature_matcher(database_path, output_path):
    """Run COLMAP feature matcher"""
    print("[COLMAP Feature Matcher] Starting...")
    print(f"  Database path: {database_path}")
    print(f"  Output path: {output_path}")
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    
    # Build COLMAP command
    cmd = [
        'colmap', 'feature_matcher',
        '--database_path', database_path,
        '--SiftMatching.guided_matching', '1'
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Feature Matcher] Complete!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Feature matching failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def run_sparse_reconstructor(database_path, image_path, output_path):
    """Run COLMAP sparse reconstructor (mapper)"""
    print("[COLMAP Sparse Reconstructor] Starting...")
    print(f"  Database path: {database_path}")
    print(f"  Image path: {image_path}")
    print(f"  Output path: {output_path}")
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    
    # Build COLMAP command
    cmd = [
        'colmap', 'mapper',
        '--database_path', database_path,
        '--image_path', image_path,
        '--output_path', output_path,
        '--MinErrorRatio', '3',
        '--Refine_intrinsics', '0'
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Sparse Reconstructor] Complete!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Sparse reconstruction failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def export_sparse_point_cloud(output_path, export_path):
    """Export sparse point cloud to PLY file"""
    print("[Export Sparse Point Cloud] Starting...")
    print(f"  Input: {output_path}")
    print(f"  Export path: {export_path}")
    
    # Find the reconstruction file
    recon_files = glob.glob(os.path.join(output_path, '**', 'reconX', 'models', 'benchmarks_*.bin'), recursive=True)
    if not recon_files:
        # Try alternative path structure
        recon_files = glob.glob(os.path.join(output_path, '**', 'models', '*.bin'), recursive=True)
    
    if not recon_files:
        print("[Error] No reconstruction file found")
        return False
    
    recon_file = recon_files[0]
    print(f"  Found reconstruction: {recon_file}")
    
    # Build COLMAP command
    cmd = [
        'colmap', 'model_converter',
        '--input_path', recon_file,
        '--output_path', export_path,
        '--output_type', 'PLY'
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Export] Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Export failed: {e}")
        return False


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  COLMAP Processing for Car 3D Modeling")
    print("=" * 60)
    print("")
    print(f"  Image path: {args.image_path}")
    print(f"  Database path: {args.database_path}")
    print(f"  Output path: {args.output_path}")
    print("")
    
    # Count images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_path, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_path, ext.upper())))
    print(f"  Found {len(image_files)} images")
    print("")
    
    if len(image_files) == 0:
        print("[Error] No images found in the specified directory")
        sys.exit(1)
    
    # Step 1: Feature Extraction
    print("Step 1: Feature Extraction")
    print("-" * 40)
    if not run_feature_extractor(
            args.image_path, args.database_path,
            args.feature_extractor, args.preprocess_scale, args.preprocess_focal):
        print("[Error] Feature extraction failed")
        sys.exit(1)
    print("")
    
    # Step 2: Feature Matching
    print("Step 2: Feature Matching")
    print("-" * 40)
    if not run_feature_matcher(args.database_path, args.output_path):
        print("[Error] Feature matching failed")
        sys.exit(1)
    print("")
    
    # Step 3: Sparse Reconstruction
    print("Step 3: Sparse Reconstruction")
    print("-" * 40)
    if not run_sparse_reconstructor(
            args.database_path, args.image_path, args.output_path):
        print("[Error] Sparse reconstruction failed")
        sys.exit(1)
    print("")
    
    # Step 4: Export Sparse Point Cloud
    print("Step 4: Export Sparse Point Cloud")
    print("-" * 40)
    sparse_ply_path = os.path.join(args.output_path, 'sparse_point_cloud.ply')
    export_sparse_point_cloud(args.output_path, sparse_ply_path)
    print("")
    
    print("=" * 60)
    print("  COLMAP Processing Complete!")
    print("=" * 60)
    print("")
    print(f"  Database: {args.database_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Sparse Point Cloud: {sparse_ply_path}")


if __name__ == '__main__':
    main()
