#!/usr/bin/env python3
"""
Blender Video Generation Script for Car Comparison
- Multiple 3D model comparison
- YouTube-ready video creation
- Animation addition
"""

import argparse
import os
import sys
import glob
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Blender video generation for car comparison')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing GLB/OBJ models to compare')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Output video file path (.mp4)')
    parser.add_argument('--resolution', type=int, default=1920,
                        help='Video resolution width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Video resolution height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Video duration in seconds (default: 10.0)')
    parser.add_argument('--animation_type', type=str, default='orbit',
                        choices=['orbit', 'pan', 'fly_through', 'comparison'],
                        help='Animation type (default: orbit)')
    return parser.parse_args()


def create_blender_scene(models, output_video, resolution, height, fps, duration, animation_type):
    """Create Blender Python script for car comparison video"""
    
    # Generate Blender Python script
    blender_script = """
import bpy
import os
import math

# Setup scene
bpy.context.scene.render.resolution_x = {resolution}
bpy.context.scene.render.resolution_y = {height}
bpy.context.scene.render.fps = {fps}
bpy.context.scene.render.image_format.image_format = 'FFmpeg'
bpy.context.scene.render.ffmpeg.format = 'MPEG-4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.context.scene.render.ffmpeg.constant_rate = 'fixed'
bpy.context.scene.render.ffmpeg.video_codec = 'h264'
bpy.context.scene.render.ffmpeg.encoding_rate = 50000000

# Output path
bpy.context.scene.render.filepath = "{output_video}"

# Create camera
bpy.ops.object.camera_add(location=(0, -5, 3))
camera = bpy.data.objects['Camera']
camera.data.lens = 50

# Create light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.data.objects['Sun']
sun.data.energy = 1.5

# Create background
bpy.data.worlds['World'].use_nodes = True
bg = bpy.data.worlds['World'].node_tree.nodes['Background']
bg.inputs[0].default_value = (0.2, 0.2, 0.2, 1)

# Import models and create animation
models = {models}
animation_type = "{animation_type}"
duration = {duration}

for i, model_path in enumerate(models):
    if not os.path.exists(model_path):
        continue
    
    # Determine file type and import
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.glb' or ext == '.gltf':
        bpy.ops.import_scene.gltf(filepath=model_path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=model_path)
    elif ext == '.ply':
        bpy.ops.import_mesh.ply(filepath=model_path)
    else:
        continue
    
    # Get the last imported object
    selected_objects = bpy.context.selected_objects
    if selected_objects:
        obj = selected_objects[-1]
        
        # Center and scale model
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
        
        # Scale to fit view
        bounds = obj.dimensions
        max_dim = max(bounds.x, bounds.y, bounds.z)
        scale_factor = 2.0 / max_dim
        obj.scale = (scale_factor, scale_factor, scale_factor)
        
        # Position models side by side for comparison
        num_models = len([m for m in models if os.path.exists(m)])
        if num_models > 1:
            offset = (i - (num_models - 1) / 2) * 4
            obj.location.x = offset
        else:
            obj.location.x = 0
        
        # Set rotation keyframes based on animation type
        if animation_type == 'orbit':
            # Orbit animation
            obj.rotation_euler = (0, 0, 0)
            obj.keyframe_insert(data_path='rotation_euler', frame=0)
            
            end_frame = int(duration * fps)
            obj.rotation_euler = (0, 0, math.radians(360))
            obj.keyframe_insert(data_path='rotation_euler', frame=end_frame)
            
            # Camera orbit
            camera.keyframe_insert(data_path='location', frame=0)
            camera.location = (5 * math.cos(0), 5 * math.sin(0), 3)
            camera.keyframe_insert(data_path='location', frame=0)
            
            camera.keyframe_insert(data_path='location', frame=end_frame)
            camera.location = (5 * math.cos(math.radians(360)), 5 * math.sin(math.radians(360)), 3)
            camera.keyframe_insert(data_path='location', frame=end_frame)
        
        elif animation_type == 'pan':
            # Pan animation
            start_x = -3
            end_x = 3
            obj.keyframe_insert(data_path='location', frame=0)
            obj.location.x = start_x
            obj.keyframe_insert(data_path='location', frame=0)
            
            end_frame = int(duration * fps)
            obj.location.x = end_x
            obj.keyframe_insert(data_path='location', frame=end_frame)
        
        elif animation_type == 'fly_through':
            # Fly through animation - move camera
            end_frame = int(duration * fps)
            
            # Camera path
            bpy.ops.curve.bezier_add()
            curve = bpy.data.curves['BezierCurve']
            curve.dimensions = '3D'
            
            # Create path points
            for t in range(20):
                angle = math.radians(t * 18)
                x = 4 * math.cos(angle)
                z = 2 + 1.5 * math.sin(angle * 2)
                point = curve.splines[0].bezier_points.add(1)
                point.co = (x, -4 * math.sin(angle), z)
                point.handle_left = (x - 0.5, -4 * math.sin(angle - 0.1), z)
                point.handle_right = (x + 0.5, -4 * math.sin(angle + 0.1), z)
            
            # Add camera constraint
            bpy.ops.object.constraint_add(type='FOLLOW_PATH')
            camera.constraints['Follow Path'].target = bpy.data.objects[curve.name]
            camera.constraints['Follow Path'].use_frame_path = True

# Set frame range
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(duration * fps)

# Render
bpy.ops.render.render(write_still=False, animation=True)

print("Blender video generation complete!")
""".format(
        resolution=resolution,
        height=height,
        fps=fps,
        output_video=output_video,
        models=models,
        animation_type=animation_type,
        duration=duration
    )
    
    # Write Blender script to file
    script_path = '/workspace/blender_script.py'
    with open(script_path, 'w') as f:
        f.write(blender_script)
    
    print(f"Blender script created: {script_path}")
    return script_path


def run_blender(script_path, output_video):
    """Run Blender headlessly"""
    print("[Blender Rendering] Starting...")
    print(f"  Script: {script_path}")
    print(f"  Output: {output_video}")
    
    # Check Blender installation
    try:
        result = subprocess.run(['blender', '--version'],
                              capture_output=True, text=True)
        print(f"  Blender version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("[Error] Blender not found. Please install Blender 3.x+")
        print("[Info] Download from: https://www.blender.org/download/")
        return False
    
    # Run Blender headlessly
    cmd = [
        'blender', '--background', '--python', script_path,
        '--enable-console'
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[Blender Rendering] Complete!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Blender rendering failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def encode_video(input_dir, output_video, resolution, fps):
    """Encode video using FFmpeg (fallback if Blender not available)"""
    print("[FFmpeg Encoding] Starting...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_video}")
    
    # Find rendered frames
    frame_pattern = os.path.join(input_dir, '**', 'ffmpegy*', '*.png')
    frames = glob.glob(frame_pattern, recursive=True)
    
    if not frames:
        print("[Warning] No rendered frames found")
        return False
    
    # Sort frames
    frames.sort()
    print(f"  Found {len(frames)} frames")
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', os.path.join(input_dir, '**', '*.png'),
        '-vf', f'scale={resolution}:-1',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[FFmpeg Encoding] Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] FFmpeg encoding failed: {e}")
        return False


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Blender Video Generation for Car Comparison")
    print("=" * 60)
    print("")
    print(f"  Models directory: {args.models_dir}")
    print(f"  Output video: {args.output_video}")
    print(f"  Resolution: {args.resolution}x{args.height}")
    print(f"  FPS: {args.fps}")
    print(f"  Duration: {args.duration}s")
    print(f"  Animation: {args.animation_type}")
    print("")
    
    # Find models
    model_extensions = ['.glb', '.gltf', '.obj', '.ply']
    models = []
    for ext in model_extensions:
        models.extend(glob.glob(os.path.join(args.models_dir, f'*{ext}')))
        models.extend(glob.glob(os.path.join(args.models_dir, f'*{ext.upper()}')))
    
    print(f"  Found {len(models)} model(s)")
    for model in models:
        print(f"    - {os.path.basename(model)}")
    print("")
    
    if len(models) == 0:
        print("[Error] No models found in the specified directory")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_video) if os.path.dirname(args.output_video) else '.', exist_ok=True)
    
    # Step 1: Create Blender scene
    print("Step 1: Create Blender Scene")
    print("-" * 40)
    script_path = create_blender_scene(
        models, args.output_video,
        args.resolution, args.height, args.fps,
        args.duration, args.animation_type)
    print("")
    
    # Step 2: Run Blender
    print("Step 2: Render Video with Blender")
    print("-" * 40)
    blender_success = run_blender(script_path, args.output_video)
    print("")
    
    if not blender_success:
        print("[Warning] Blender rendering failed, trying FFmpeg encoding...")
        encode_video(args.models_dir, args.output_video, args.resolution, args.fps)
    
    print("=" * 60)
    print("  Video Generation Complete!")
    print("=" * 60)
    print("")
    print(f"  Output video: {args.output_video}")


if __name__ == '__main__':
    main()
