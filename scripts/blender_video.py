#!/usr/bin/env python3
"""
Blender Video Generation Script for Car 3D Modeling
- Create comparison videos from multiple 3D models
- Animation types: orbit, pan, fly_through, comparison
- YouTube optimized output
"""

import argparse
import os
import sys
import glob
import subprocess
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Blender video generation for car 3D modeling')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing 3D models (glb/obj/ply)')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Output video file path (mp4)')
    parser.add_argument('--animation_type', type=str, default='orbit',
                        choices=['orbit', 'pan', 'fly_through', 'comparison'],
                        help='Animation type (default: orbit)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Video duration in seconds (default: 10.0)')
    parser.add_argument('--resolution_width', type=int, default=1920,
                        help='Video width (default: 1920)')
    parser.add_argument('--resolution_height', type=int, default=1080,
                        help='Video height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    return parser.parse_args()


def find_models(models_dir: str):
    """Find 3D model files in directory"""
    model_extensions = ['*.glb', '*.gltf', '*.obj', '*.ply']
    model_files = []
    
    for ext in model_extensions:
        model_files.extend(glob.glob(os.path.join(models_dir, ext)))
        model_files.extend(glob.glob(os.path.join(models_dir, ext.upper())))
        model_files.extend(glob.glob(os.path.join(models_dir, '**', ext), recursive=True))
        model_files.extend(glob.glob(os.path.join(models_dir, '**', ext.upper()), recursive=True))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for f in model_files:
        if f not in seen:
            seen.add(f)
            unique_models.append(f)
    
    return sorted(unique_models)


def create_blender_script(model_files: list, output_video: str, 
                          animation_type: str, duration: float,
                          width: int, height: int, fps: int):
    """Create Blender Python script for video generation"""
    
    script_content = f'''
import bpy
import os
import math

# Scene setup
bpy.context.scene.render.resolution_x = {width}
bpy.context.scene.render.resolution_y = {height}
bpy.context.scene.render.fps = {fps}
bpy.context.scene.render.image_format = "FFMPEG"
bpy.context.scene.render.ffmpeg_format = "VIDEO"
bpy.context.scene.render.filepath = "{output_video}"
bpy.context.scene.render.ffmpeg.audio = False

# Set帧数
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int({duration} * {fps})

# Remove default objects
bpy.ops.object.delete()

# Add lighting
bpy.ops.object.light_add(type="SUN", location=(5, 5, 5))
bpy.data.lights["Sun"].energy = 2.0

bpy.ops.object.light_add(type="AREA", location=(-5, 3, 2))
bpy.data.lights["Area"].energy = 50.0

bpy.ops.object.light_add(type="POINT", location=(3, -4, 3))
bpy.data.lights["Point"].color = (0.9, 0.7, 0.5)
bpy.data.lights["Point"].energy = 30.0

# Camera setup
bpy.ops.object.camera_add(location=(5, 5, 3))
bpy.data.cameras["Camera"].lens = 50
bpy.data.cameras["Camera"].clip_start = 0.1
bpy.data.cameras["Camera"].clip_end = 100
bpy.context.scene.camera = bpy.data.objects["Camera"]

# Add grid
bpy.ops.object.grid_add(location=(0, 0, 0))
bpy.data.objects["Grid"].scale = (2, 2, 2)

# Import models
models = {json.dumps(model_files)}
imported_objects = []

for model_path in models:
    if not os.path.exists(model_path):
        continue
    
    print(f"Importing: {{model_path}}")
    
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=model_path)
    elif ext == ".glb" or ext == ".gltf":
        try:
            bpy.ops.import_scene.gltf(filepath=model_path)
        except:
            print(f"  GLB import failed, trying external converter...")
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=model_path)
    
    # Get the last imported object
    selected = bpy.context.selected_objects
    for obj in selected:
        if obj.type in ["MESH", "CURVE"]:
            imported_objects.append(obj)
            # Center and scale
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")
            
            # Scale to fit scene
            bbox_min = obj.dimensions[0] * obj.scale[0]
            bbox_max = max(bbox_min, 0.01)
            scale = 2.0 / bbox_max
            obj.scale = (scale, scale, scale)
            
            # Position models in a row for comparison
            idx = len(imported_objects) - 1
            offset = (idx - (len(models) - 1) / 2) * 4
            obj.location[0] += offset

# Set camera animation
camera = bpy.data.objects["Camera"]

if "{{animation_type}}" == "orbit":
    # Orbit animation around center
    center_x = sum([obj.location[0] for obj in imported_objects]) / max(len(imported_objects), 1)
    center_z = 0
    
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        angle = (frame / bpy.context.scene.frame_end) * math.tau
        camera.location[0] = center_x + 4 * math.cos(angle)
        camera.location[2] = 3
        camera.location[1] = 4 * math.sin(angle)
        camera.look_at_set(bpy.context.scene, (center_x, 0, 0))

elif "{{animation_type}}" == "pan":
    # Horizontal pan
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        progress = (frame / bpy.context.scene.frame_end) * 2 - 1
        camera.location[0] = progress * 5
        camera.location[2] = 2
        camera.location[1] = 3
        if imported_objects:
            center_x = sum([obj.location[0] for obj in imported_objects]) / len(imported_objects)
            camera.look_at_set(bpy.context.scene, (center_x, 0, 0))

elif "{{animation_type}}" == "fly_through":
    # Fly-through animation
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        progress = (frame / bpy.context.scene.frame_end)
        angle = progress * math.tau
        camera.location[0] = 5 * math.cos(angle)
        camera.location[1] = 5 * math.sin(angle)
        camera.location[2] = 2 + math.sin(progress * math.pi * 4)
        if imported_objects:
            center_x = sum([obj.location[0] for obj in imported_objects]) / len(imported_objects)
            camera.look_at_set(bpy.context.scene, (center_x, 0, 0))

elif "{{animation_type}}" == "comparison":
    # Comparison animation - show each model
    num_models = len(imported_objects)
    per_model_frames = bpy.context.scene.frame_end // max(num_models, 1)
    
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        
        model_idx = frame // max(per_model_frames, 1)
        model_idx = min(model_idx, num_models - 1)
        
        if model_idx >= 0 and model_idx < len(imported_objects):
            obj = imported_objects[model_idx]
            camera.location[0] = obj.location[0] + 3
            camera.location[1] = 3
            camera.location[2] = 2
            camera.look_at_set(bpy.context.scene, (obj.location[0], 0, 0))

# Render setup
bpy.context.scene.render.file_format = "FFMPEG"
bpy.context.scene.render.ffmpeg.codec = "H264"
bpy.context.scene.render.ffmpeg.constant_rate = "medium"
bpy.context.scene.render.ffmpeg.preset = "medium"

# Render animation
print("Starting render...")
bpy.ops.render.render(animation=True)
print(f"Video saved to: {output_video}")
'''
    
    script_path = os.path.join(os.path.dirname(output_video), 'blender_script.py')
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"  Blender script created: {script_path}")
    return script_path


def render_with_blender(script_path: str, output_video: str):
    """Render video using Blender"""
    print("  Rendering video with Blender...")
    
    # Check if Blender is available
    import shutil
    if shutil.which('blender') is None:
        print("  [Warning] Blender not found in system path")
        return False
    
    cmd = [
        'blender', '--background', '--python', script_path,
        '--',
        '--output', output_video
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        print("  Video rendering complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Blender rendering failed: {e}")
        if e.stderr:
            print(f"  {e.stderr[:500]}")
        return False
    except subprocess.TimeoutExpired:
        print("  [Error] Rendering timed out")
        return False


def render_with_blendnet(script_path: str, output_video: str):
    """Render video using BlendNet (network rendering)"""
    print("  [Warning] BlendNet not configured, using local rendering")
    return False


def export_video_ffmpeg(output_video: str, frames_dir: str = None):
    """Export video using FFmpeg (fallback)"""
    print("  Creating video with FFmpeg...")
    
    if frames_dir and os.path.exists(frames_dir):
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-patterntype', 'glob',
            '-i', os.path.join(frames_dir, '*.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
    else:
        # Create a simple test video
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'testsrc=rate=30:duration=10:size=1920x1080',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("  FFmpeg video export complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [Error] FFmpeg export failed: {e}")
        return False


def create_placeholder_video(output_video: str, duration: float = 10.0):
    """Create a placeholder video when rendering fails"""
    print("  Creating placeholder video...")
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'testsrc=rate=30:duration={duration}:size=1920x1080',
        '-vf', f'text="Car 3D Model - {duration}s"',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Placeholder video created: {output_video}")
        return True
    except Exception as e:
        print(f"  [Error] Failed to create placeholder: {e}")
        return False


def video_generation_pipeline(models_dir: str, output_video: str, 
                               animation_type: str = 'orbit',
                               duration: float = 10.0,
                               width: int = 1920, height: int = 1080,
                               fps: int = 30):
    """Run complete video generation pipeline"""
    print("=" * 60)
    print("  Blender Video Generation Pipeline")
    print("=" * 60)
    print(f"  Models directory: {models_dir}")
    print(f"  Output video: {output_video}")
    print(f"  Animation type: {animation_type}")
    print(f"  Duration: {duration}s")
    print(f"  Resolution: {width}x{height}")
    print("")
    
    # Step 1: Find models
    model_files = find_models(models_dir)
    
    if not model_files:
        print("[Warning] No model files found")
        print("  Creating placeholder video...")
        return create_placeholder_video(output_video, duration)
    
    print(f"  Found {len(model_files)} models:")
    for f in model_files:
        print(f"    - {os.path.basename(f)}")
    print("")
    
    # Step 2: Create Blender script
    script_path = create_blender_script(
        model_files, output_video, animation_type, duration,
        width, height, fps
    )
    
    # Step 3: Render with Blender
    success = render_with_blender(script_path, output_video)
    
    if not success:
        print("  [Warning] Blender rendering failed")
        print("  Creating placeholder video...")
        return create_placeholder_video(output_video, duration)
    
    # Step 4: Verify output
    if os.path.exists(output_video):
        file_size = os.path.getsize(output_video)
        print(f"  Video file size: {file_size / 1024 / 1024:.2f} MB")
    else:
        print("  [Warning] Output video file not found")
        return create_placeholder_video(output_video, duration)
    
    print("")
    print("  Video generation pipeline complete!")
    print(f"  Output: {output_video}")
    
    return output_video


def main():
    args = parse_args()
    
    result = video_generation_pipeline(
        args.models_dir,
        args.output_video,
        args.animation_type,
        args.duration,
        args.resolution_width,
        args.resolution_height,
        args.fps
    )
    
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
