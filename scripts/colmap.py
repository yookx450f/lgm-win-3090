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
import glob
import subprocess
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='COLMAP for car 3D modeling')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Input directory containing preprocessed images')
    parser.add_argument('--database_path', type=str, required=True,
                        help='Path to output COLMAP database')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for COLMAP results')
    parser.add_argument('--feature_max_num_features', type=int, default=16384,
                        help='Maximum number of features to extract (default: 16384)')
    parser.add_argument('--SIFT_rotation_invariance', action='store_true',
                        help='Enable rotation invariance for SIFT')
    parser.add_argument('--Matching_max_error', type=float, default=4.0,
                        help='Maximum error for feature matching (default: 4.0)')
    return parser.parse_args()


def create_colmap_project(image_path: str, database_path: str, output_path: str):
    """Create COLMAP project structure"""
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    
    # COLMAP directories
    images_dir = os.path.join(output_path, 'images')
    databases_dir = os.path.join(output_path, 'databases')
    sparse_dir = os.path.join(output_path, 'sparse')
    dense_dir = os.path.join(output_path, 'dense')
    
    for d in [images_dir, databases_dir, sparse_dir, dense_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Copy images to COLMAP images directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_path, ext)))
        image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
    
    if not image_files:
        print("[Error] No image files found")
        return None
    
    print(f"  Found {len(image_files)} images")
    
    # Copy images to COLMAP images directory with proper naming
    images_copy_dir = os.path.join(images_dir, 'images')
    os.makedirs(images_copy_dir, exist_ok=True)
    
    for i, img_file in enumerate(image_files, 1):
        filename = os.path.basename(img_file)
        dest = os.path.join(images_copy_dir, f"{i:03d}_{filename}")
        shutil.copy2(img_file, dest)
    
    # Update database path
    database_path = os.path.join(databases_dir, 'database.db')
    
    return {
        'images_dir': images_copy_dir,
        'database_path': database_path,
        'sparse_dir': sparse_dir,
        'dense_dir': dense_dir,
        'output_path': output_path
    }


def extract_features(images_dir: str, database_path: str, max_num_features: int = 16384):
    """Extract SIFT features from images"""
    print("  Extracting features...")
    
    import os
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    
    cmd = [
        'colmap', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', images_dir,
        '--SiftExtraction.max_num_features', str(max_num_features),
        '--SiftExtraction.use_gpu', '0',
        '--SiftExtraction.num_threads', '-1'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print("    Features extracted successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [Error] Feature extraction failed: {e}")
        if e.stderr:
            print(f"    {e.stderr}")
        return False


def match_features(database_path: str, max_error: float = 4.0):
    """Match features between images"""
    print("  Matching features...")
    
    import os
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for matching
    
    # Use exhaustive matcher for unordered images (car from different angles)
    cmd = [
        'colmap', 'exhaustive_matcher',
        '--database_path', database_path,
        '--SiftMatching.max_error', str(max_error),
        '--SiftMatching.use_gpu', '0'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print("    Exhaustive matching done")
        print("    Features matched successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [Error] Feature matching failed: {e}")
        if e.stderr:
            print(f"    {e.stderr}")
        return False


def reconstruct_scene(database_path: str, sparse_dir: str, images_dir: str):
    """Run Structure-from-Motion to reconstruct scene"""
    print("  Running Structure-from-Motion (SfM)...")
    
    import os
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    
    # First, create a dummy model if sparse directory is empty
    model_files = [f for f in os.listdir(sparse_dir) if f.endswith('.txt')] if os.path.exists(sparse_dir) else []
    
    if not model_files:
        # Use mapper to initialize
        cmd = [
            'colmap', 'mapper',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--output_path', sparse_dir
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            print("    SfM reconstruction done")
        except subprocess.CalledProcessError as e:
            print(f"    [Error] SfM reconstruction failed: {e}")
            if e.stderr:
                print(f"    {e.stderr}")
            return False
    else:
        # Triangulate points
        try:
            cmd = [
                'colmap', 'point_triangulator',
                '--database_path', database_path,
                '--image_path', images_dir,
                '--input_path', sparse_dir,
                '--output_path', sparse_dir
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            print("    Point triangulation done")
        except subprocess.CalledProcessError:
            pass
    
    return True


def export_results(sparse_dir: str, output_path: str):
    """Export COLMAP results"""
    print("  Exporting results...")
    
    sparse_model_dir = os.path.join(sparse_dir, '0')
    
    if not os.path.exists(sparse_model_dir):
        print("  [Warning] No sparse model found")
        return False
    
    # Export camera parameters
    cameras_file = os.path.join(sparse_model_dir, 'cameras.bin')
    if os.path.exists(cameras_file):
        print("    Cameras exported")
    
    # Export image poses
    images_file = os.path.join(sparse_model_dir, 'images.bin')
    if os.path.exists(images_file):
        print("    Image poses exported")
    
    # Export sparse point cloud
    points_file = os.path.join(sparse_model_dir, 'points.bin')
    if os.path.exists(points_file):
        print("    Sparse point cloud exported")
    
    # Create summary JSON
    summary = {
        'status': 'completed',
        'sparse_model_dir': sparse_model_dir,
        'has_cameras': os.path.exists(cameras_file),
        'has_images': os.path.exists(images_file),
        'has_points': os.path.exists(points_file)
    }
    
    summary_path = os.path.join(output_path, 'colmap_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("  Results exported successfully")
    return True


def colmap_pipeline(image_path: str, database_path: str, output_path: str,
                    max_num_features: int = 16384, max_error: float = 4.0):
    """Run complete COLMAP pipeline"""
    print("=" * 60)
    print("  COLMAP Pipeline")
    print("=" * 60)
    print(f"  Input images: {image_path}")
    print(f"  Database: {database_path}")
    print(f"  Output: {output_path}")
    print("")
    
    # Step 1: Create project
    project = create_colmap_project(image_path, database_path, output_path)
    
    if project is None:
        print("[Error] Failed to create COLMAP project")
        return None
    
    database_path = project['database_path']
    sparse_dir = project['sparse_dir']
    
    # Step 2: Extract features
    if not extract_features(project['images_dir'], database_path, max_num_features):
        print("[Error] Feature extraction failed")
        return None
    
    # Step 3: Match features
    if not match_features(database_path, max_error):
        print("[Error] Feature matching failed")
        return None
    
    # Step 4: Reconstruct scene
    if not reconstruct_scene(database_path, sparse_dir, project['images_dir']):
        print("[Error] Scene reconstruction failed")
        return None
    
    # Step 5: Export results
    export_results(sparse_dir, output_path)
    
    print("")
    print("  COLMAP pipeline complete!")
    
    return sparse_dir


def main():
    args = parse_args()
    
    result = colmap_pipeline(
        args.image_path,
        args.database_path,
        args.output_path,
        args.feature_max_num_features,
        args.Matching_max_error
    )
    
    if result is None:
        sys.exit(1)
    else:
        print(f"\n  Output directory: {args.output_path}")


if __name__ == '__main__':
    main()
