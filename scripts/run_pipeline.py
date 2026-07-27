#!/usr/bin/env python3
"""
Main Pipeline Script for Car 3D Modeling
- Orchestrates the entire pipeline
- COLMAP + Gaussian Splatting + Meshing + Texture Baking
- Blender video generation
"""

import argparse
import os
import sys
import glob
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Main pipeline for car 3D modeling')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing car images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for 3D models and video')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'preprocess', 'colmap', 'dense', 
                                'gaussian_splatting', 'meshing', 'texture_baking', 'video'],
                        help='Pipeline step to run (default: all)')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Target image size for preprocessing (default: 1024)')
    parser.add_argument('--bg_color', type=str, default='white',
                        help='Background color: white, black, green (default: white)')
    parser.add_argument('--mesh_method', type=str, default='poisson',
                        choices=['poisson', 'instant_meshes', 'dmver2'],
                        help='Meshing method (default: poisson)')
    parser.add_argument('--mesh_depth', type=int, default=10,
                        help='Poisson reconstruction depth (default: 10)')
    parser.add_argument('--mesh_resolution', type=int, default=256,
                        help='Mesh resolution (default: 256)')
    parser.add_argument('--mesh_smooth', type=bool, default=True,
                        help='Apply mesh smoothing (default: True)')
    parser.add_argument('--animation_type', type=str, default='orbit',
                        choices=['orbit', 'pan', 'fly_through', 'comparison'],
                        help='Animation type for video (default: orbit)')
    parser.add_argument('--video_duration', type=float, default=10.0,
                        help='Video duration in seconds (default: 10.0)')
    parser.add_argument('--texture_size', type=int, default=2048,
                        help='Texture resolution (default: 2048)')
    parser.add_argument('--specular_strength', type=float, default=0.5,
                        help='Specular strength (default: 0.5)')
    parser.add_argument('--roughness', type=float, default=0.3,
                        help='Roughness value (default: 0.3)')
    parser.add_argument('--metallic', type=float, default=0.1,
                        help='Metallic value (default: 0.1)')
    parser.add_argument('--clearcoat', type=float, default=0.5,
                        help='Clearcoat value for car paint (default: 0.5)')
    parser.add_argument('--source_images', type=str, default=None,
                        help='Directory containing source images for texturing')
    return parser.parse_args()


def check_dependencies():
    """Check required dependencies"""
    print("[Dependency Check] Checking required tools...")
    
    required_tools = {
        'python3': 'Python 3',
        'colmap': 'COLMAP',
        'ffmpeg': 'FFmpeg'
    }
    
    missing_tools = []
    for tool, name in required_tools.items():
        try:
            result = subprocess.run([tool, '--version'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  [OK] {name}: {result.stdout.strip().split(chr(10))[0]}")
            else:
                missing_tools.append(name)
        except FileNotFoundError:
            missing_tools.append(name)
    
    if missing_tools:
        print(f"[Error] Missing tools: {', '.join(missing_tools)}")
        print("[Info] Please install the missing tools")
        return False
    
    print("[Dependency Check] All dependencies found!")
    return True


def run_preprocessing(input_dir, output_dir, image_size, bg_color):
    """Run preprocessing step"""
    print("=" * 60)
    print("  Step 1: Preprocessing")
    print("=" * 60)
    
    script = '/workspace/scripts/preprocess.py'
    cmd = [
        'python3', script,
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--image_size', str(image_size),
        '--bg_color', bg_color
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Preprocessing] Complete!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Preprocessing failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def run_colmap(preprocessed_dir, output_dir):
    """Run COLMAP step"""
    print("=" * 60)
    print("  Step 2: COLMAP")
    print("=" * 60)
    
    database_path = os.path.join(output_dir, 'database.db')
    colmap_output = os.path.join(output_dir, 'colmap_output')
    
    script = '/workspace/scripts/colmap.py'
    cmd = [
        'python3', script,
        '--image_path', preprocessed_dir,
        '--database_path', database_path,
        '--output_path', colmap_output
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[COLMAP] Complete!")
        if result.stdout:
            print(result.stdout)
        return colmap_output
    except subprocess.CalledProcessError as e:
        print(f"[Error] COLMAP failed: {e}")
        if e.stderr:
            print(e.stderr)
        return None


def run_dense_reconstruction(colmap_output, output_dir):
    """Run dense reconstruction (Multi-View Stereo)"""
    print("=" * 60)
    print("  Step 3: Dense Reconstruction")
    print("=" * 60)
    
    stereo_path = os.path.join(output_dir, 'stereo')
    
    # Use COLMAP's patch_match_stereo
    database_path = os.path.join(output_dir, 'database.db')
    
    cmd = [
        'colmap', 'patch_match_stereo',
        '--workspace_path', stereo_path,
        '--database_path', database_path
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Dense Reconstruction] Complete!")
        if result.stdout:
            print(result.stdout)
        return stereo_path
    except subprocess.CalledProcessError as e:
        print(f"[Error] Dense reconstruction failed: {e}")
        return None


def run_gaussian_splatting(colmap_output, output_dir):
    """Run Gaussian Splatting step"""
    print("=" * 60)
    print("  Step 4: Gaussian Splatting")
    print("=" * 60)
    
    gs_output = os.path.join(output_dir, 'gaussian_splatting_output')
    
    script = '/workspace/scripts/gaussian_splatting.py'
    cmd = [
        'python3', script,
        '--source', colmap_output,
        '--output_path', gs_output
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Gaussian Splatting] Complete!")
        if result.stdout:
            print(result.stdout)
        return gs_output
    except subprocess.CalledProcessError as e:
        print(f"[Error] Gaussian Splatting failed: {e}")
        return None


def run_meshing(gs_output, output_path, method='poisson',
                depth=10, resolution=256, smooth=True):
    """Run meshing step"""
    print("=" * 60)
    print("  Step 5: Meshing")
    print("=" * 60)
    
    script = '/workspace/scripts/meshing.py'
    cmd = [
        'python3', script,
        '--input', gs_output,
        '--output', output_path,
        '--method', method,
        '--depth', str(depth),
        '--resolution', str(resolution),
        '--smooth', str(smooth).lower()
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Meshing] Complete!")
        if result.stdout:
            print(result.stdout)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[Error] Meshing failed: {e}")
        if e.stderr:
            print(e.stderr)
        return None


def run_texture_baking(mesh_path, output_path, texture_size=2048,
                       specular_strength=0.5, roughness=0.3,
                       metallic=0.1, clearcoat=0.5, source_images=None):
    """Run texture baking step"""
    print("=" * 60)
    print("  Step 6: Texture Baking")
    print("=" * 60)
    
    script = '/workspace/scripts/texture_baking.py'
    cmd = [
        'python3', script,
        '--input', mesh_path,
        '--output', output_path,
        '--texture_size', str(texture_size),
        '--specular_strength', str(specular_strength),
        '--roughness', str(roughness),
        '--metallic', str(metallic),
        '--clearcoat', str(clearcoat)
    ]
    
    if source_images:
        cmd.extend(['--source_images', source_images])
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Texture Baking] Complete!")
        if result.stdout:
            print(result.stdout)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[Error] Texture baking failed: {e}")
        if e.stderr:
            print(e.stderr)
        return None


def run_video_generation(models_dir, output_video, animation_type, duration):
    """Run Blender video generation"""
    print("=" * 60)
    print("  Step 7: Video Generation")
    print("=" * 60)
    
    script = '/workspace/scripts/blender_video.py'
    cmd = [
        'python3', script,
        '--models_dir', models_dir,
        '--output_video', output_video,
        '--animation_type', animation_type,
        '--duration', str(duration)
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Video Generation] Complete!")
        if result.stdout:
            print(result.stdout)
        return output_video
    except subprocess.CalledProcessError as e:
        print(f"[Error] Video generation failed: {e}")
        return None


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Car 3D Modeling Pipeline")
    print("=" * 60)
    print("")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Step: {args.step}")
    print("")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths
    preprocessed_dir = os.path.join(args.output_dir, 'preprocessed')
    colmap_output_dir = os.path.join(args.output_dir, 'colmap_output')
    gs_output_dir = os.path.join(args.output_dir, 'gaussian_splatting_output')
    mesh_output_path = os.path.join(args.output_dir, 'model.glb')
    textured_output_path = os.path.join(args.output_dir, 'model_textured.glb')
    video_output_path = os.path.join(args.output_dir, 'car_comparison.mp4')
    
    # Run pipeline steps
    current_step = args.step
    
    if current_step in ['all', 'preprocess']:
        print("")
        if not run_preprocessing(args.input_dir, preprocessed_dir, 
                                 args.image_size, args.bg_color):
            print("[Error] Preprocessing failed")
            sys.exit(1)
    
    if current_step in ['all', 'colmap']:
        print("")
        colmap_result = run_colmap(preprocessed_dir, colmap_output_dir)
        if not colmap_result:
            print("[Error] COLMAP failed")
            sys.exit(1)
    
    if current_step in ['all', 'dense']:
        print("")
        dense_result = run_dense_reconstruction(colmap_output_dir, args.output_dir)
        if not dense_result:
            print("[Warning] Dense reconstruction failed, continuing...")
    
    if current_step in ['all', 'gaussian_splatting']:
        print("")
        gs_result = run_gaussian_splatting(colmap_output_dir, gs_output_dir)
        if not gs_result:
            print("[Error] Gaussian Splatting failed")
            sys.exit(1)
    
    if current_step in ['all', 'meshing']:
        print("")
        mesh_result = run_meshing(
            gs_output_dir, mesh_output_path,
            args.mesh_method, args.mesh_depth,
            args.mesh_resolution, args.mesh_smooth)
        if not mesh_result:
            print("[Warning] Meshing failed, continuing...")
    
    if current_step in ['all', 'texture_baking']:
        print("")
        if os.path.exists(mesh_output_path):
            texture_result = run_texture_baking(
                mesh_output_path, textured_output_path,
                args.texture_size, args.specular_strength,
                args.roughness, args.metallic, args.clearcoat,
                args.source_images)
            if not texture_result:
                print("[Warning] Texture baking failed, continuing...")
        else:
            print("[Warning] No mesh file found, skipping texture baking")
    
    if current_step in ['all', 'video']:
        print("")
        # Find all GLB/OBJ files for video
        models_dir = args.output_dir
        video_result = run_video_generation(
            models_dir, video_output_path,
            args.animation_type, args.video_duration)
        if not video_result:
            print("[Warning] Video generation failed")
    
    # Summary
    print("")
    print("=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("")
    print("  Output files:")
    
    # List output files
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = '  ' * (level + 1)
        for file in files[:10]:  # Show only first 10 files per directory
            print(f"{sub_indent}{file}")
        if len(files) > 10:
            print(f"{sub_indent}... and {len(files) - 10} more files")
    
    print("")
    print(f"  Full output: {args.output_dir}")


if __name__ == '__main__':
    main()
