#!/usr/bin/env python3
"""
Texture Baking Script for Car 3D Modeling
- UV unwrapping
- Texture mapping
- Color correction
- Specular (gloss)
- Reflection
"""

import argparse
import os
import sys
import glob
import subprocess
import json
import shutil
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Texture baking for car 3D modeling')
    parser.add_argument('--input', type=str, required=True,
                        help='Input 3D model file (glb/obj/ply)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output textured 3D model file')
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


def find_model_file(input_dir: str):
    """Find 3D model file in input directory"""
    # Look for GLB files
    glb_files = glob.glob(os.path.join(input_dir, '*.glb')) + \
                glob.glob(os.path.join(input_dir, '*.gltf'))
    
    # Look for OBJ files
    obj_files = glob.glob(os.path.join(input_dir, '*.obj'))
    
    # Look for PLY files
    ply_files = glob.glob(os.path.join(input_dir, '*.ply'))
    
    if glb_files:
        return glb_files[0]
    elif obj_files:
        return obj_files[0]
    elif ply_files:
        return ply_files[0]
    
    return None


def load_model(file_path: str):
    """Load 3D model file"""
    print(f"  Loading model: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.obj':
        return load_obj(file_path)
    elif ext == '.ply':
        return load_ply_model(file_path)
    elif ext == '.glb' or ext == '.gltf':
        return load_glb(file_path)
    else:
        print(f"  [Error] Unsupported format: {ext}")
        return None


def load_obj(file_path: str):
    """Load OBJ model file"""
    vertices = []
    faces = []
    uv_coords = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vt':
                uv_coords.append([float(x) for x in parts[1:3]])
            elif parts[0] == 'f':
                face = []
                for x in parts[1:4]:
                    parts_idx = x.split('/')
                    face.append(int(parts_idx[0]) - 1)
                faces.append(face)
    
    if not vertices:
        return None
    
    result = {
        'vertices': np.array(vertices),
        'faces': np.array(faces),
        'uv_coords': np.array(uv_coords) if uv_coords else None,
        'colors': None,
        'file_path': file_path
    }
    
    print(f"    Loaded: {len(vertices)} vertices, {len(faces)} faces")
    return result


def load_ply_model(file_path: str):
    """Load PLY model file"""
    vertices = []
    faces = []
    colors = []
    
    header_end = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
                continue
            
            if line == 'end_header':
                header_end = True
                continue
            
            if header_end:
                parts = line.split()
                if len(parts) >= 3:
                    vertices.append([float(x) for x in parts[:3]])
                    
                    if len(parts) >= 6:
                        colors.append([float(x) for x in parts[3:6]])
                    elif len(parts) >= 6:
                        # Check for RGB colors
                        if 'red' in line.lower() or 'green' in line.lower() or 'blue' in line.lower():
                            colors.append([float(x) for x in parts[3:6]])
    
    if not vertices:
        return None
    
    result = {
        'vertices': np.array(vertices),
        'faces': np.array(faces) if faces else None,
        'colors': np.array(colors) if colors else None,
        'uv_coords': None,
        'file_path': file_path
    }
    
    print(f"    Loaded: {len(vertices)} vertices")
    return result


def load_glb(file_path: str):
    """Load GLB model file (simplified)"""
    print("  [Warning] GLB loading is simplified. Use external tools for full support.")
    return None


def generate_uv_coords(vertices, faces):
    """Generate UV coordinates using simple projection"""
    print("  Generating UV coordinates...")
    
    num_vertices = len(vertices)
    uv_coords = np.zeros((num_vertices, 2), dtype=np.float32)
    
    # Calculate bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    range_z = max_coords[2] - min_coords[2]
    
    # Use largest axis for UV projection
    if range_x >= range_y and range_x >= range_z:
        # Project onto YZ plane
        uv_coords[:, 0] = (vertices[:, 1] - min_coords[1]) / max(range_y, 0.001)
        uv_coords[:, 1] = (vertices[:, 2] - min_coords[2]) / max(range_z, 0.001)
    elif range_y >= range_x and range_y >= range_z:
        # Project onto XZ plane
        uv_coords[:, 0] = (vertices[:, 0] - min_coords[0]) / max(range_x, 0.001)
        uv_coords[:, 1] = (vertices[:, 2] - min_coords[2]) / max(range_z, 0.001)
    else:
        # Project onto XY plane
        uv_coords[:, 0] = (vertices[:, 0] - min_coords[0]) / max(range_x, 0.001)
        uv_coords[:, 1] = (vertices[:, 1] - min_coords[1]) / max(range_y, 0.001)
    
    # Normalize to [0, 1]
    uv_min = np.min(uv_coords, axis=0)
    uv_max = np.max(uv_coords, axis=0)
    uv_range = uv_max - uv_min
    
    if uv_range[0] > 0:
        uv_coords[:, 0] = (uv_coords[:, 0] - uv_min[0]) / uv_range[0]
    if uv_range[1] > 0:
        uv_coords[:, 1] = (uv_coords[:, 1] - uv_min[1]) / uv_range[1]
    
    print(f"    Generated UV coordinates for {num_vertices} vertices")
    return uv_coords


def create_texture_from_images(source_images: str, texture_size: int):
    """Create texture from source images using average color"""
    if not source_images or not os.path.exists(source_images):
        return None
    
    print("  Creating texture from source images...")
    
    image_files = glob.glob(os.path.join(source_images, '*.jpg')) + \
                  glob.glob(os.path.join(source_images, '*.png'))
    
    if not image_files:
        print("  [Warning] No source images found")
        return None
    
    # Load and average images
    images = []
    for img_file in image_files[:10]:  # Use up to 10 images
        try:
            from PIL import Image
            img = Image.open(img_file).resize((texture_size, texture_size), Image.Resampling.LANCZOS)
            images.append(np.array(img).astype(np.float32))
        except Exception as e:
            print(f"    Warning: Failed to load {img_file}: {e}")
    
    if not images:
        return None
    
    # Create average texture
    texture = np.mean(images, axis=0).astype(np.uint8)
    print(f"    Created texture: {texture.shape}")
    
    return texture


def apply_material_properties(mesh: dict, specular_strength: float, 
                              roughness: float, metallic: float, clearcoat: float):
    """Apply material properties to mesh"""
    print("  Applying material properties...")
    print(f"    Specular: {specular_strength}")
    print(f"    Roughness: {roughness}")
    print(f"    Metallic: {metallic}")
    print(f"    Clearcoat: {clearcoat}")
    
    # Store material properties in mesh
    mesh['material'] = {
        'specular_strength': specular_strength,
        'roughness': roughness,
        'metallic': metallic,
        'clearcoat': clearcoat
    }
    
    return mesh


def export_textured_model(mesh: dict, output_path: str, texture_size: int = 2048):
    """Export textured model"""
    print(f"  Exporting textured model: {output_path}")
    
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.obj':
        export_textured_obj(mesh, output_path, texture_size)
    elif ext == '.glb' or ext == '.gltf':
        export_textured_glb(mesh, output_path, texture_size)
    elif ext == '.ply':
        export_textured_ply(mesh, output_path)
    else:
        export_textured_obj(mesh, output_path.replace('.glb', '.obj'), texture_size)
    
    print(f"  Textured model exported: {output_path}")


def export_textured_obj(mesh: dict, output_path: str, texture_size: int = 2048):
    """Export textured OBJ model"""
    vertices = mesh['vertices']
    faces = mesh['faces']
    colors = mesh.get('colors')
    uv_coords = mesh.get('uv_coords')
    
    # Generate UV coords if not present
    if uv_coords is None:
        uv_coords = generate_uv_coords(vertices, faces)
    
    # Create MTL file
    mtl_path = output_path.replace('.obj', '.mtl')
    material = mesh.get('material', {})
    
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl car_paint\n")
        f.write(f"Ka 0.0 0.0 0.0\n")  # Ambient
        f.write(f"Kd 0.8 0.8 0.8\n")  # Diffuse (white base)
        f.write(f"Ks {material.get('specular_strength', 0.5):.4f} {material.get('specular_strength', 0.5):.4f} {material.get('specular_strength', 0.5):.4f}\n")  # Specular
        f.write(f"Ns {(1.0 - material.get('roughness', 0.3)) * 1000:.0f}\n")  # Shininess
        f.write(f"Ni 1.5\n")  # Index of refraction
        f.write(f"d 1.0\n")  # Dissolve
        f.write(f"illum 2\n")  # Model 2 (specular + alpha)
    
    # Write OBJ file
    with open(output_path, 'w') as f:
        f.write(f"# Textured model with {len(vertices)} vertices\n")
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write UV coordinates
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        # Write normals
        for i in range(len(vertices)):
            f.write(f"vn 0.0 1.0 0.0\n")
        
        # Write faces with UV and normals
        f.write(f"usemtl car_paint\n")
        for face in faces:
            f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")
    
    print(f"    OBJ file created: {len(vertices)} vertices, {len(faces)} faces")


def export_textured_glb(mesh: dict, output_path: str, texture_size: int = 2048):
    """Export textured GLB model using trimesh"""
    try:
        import trimesh
        import json
        
        vertices = mesh['vertices']
        faces = mesh['faces']
        colors = mesh.get('colors')
        uv_coords = mesh.get('uv_coords')
        material = mesh.get('material', {})
        
        print("  Creating GLB file using trimesh...")
        
        # Create vertex colors (normalized to 0-1)
        vertex_colors = None
        if colors is not None and len(colors) > 0:
            vertex_colors = colors.astype(float) / 255.0
        
        # Create trimesh Trimesh object with optional UV coordinates (note: class is Trimesh, not Mesh)
        if uv_coords is not None and len(uv_coords) > 0:
            mesh_obj = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                face_attributes={
                    'material': json.dumps({
                        'specular_strength': material.get('specular_strength', 0.5),
                        'roughness': material.get('roughness', 0.3),
                        'metallic': material.get('metallic', 0.1),
                        'clearcoat': material.get('clearcoat', 0.5)
                    })
                }
            )
            # Add UV coordinates if available
            if len(uv_coords) == len(vertices):
                mesh_obj.visual.uv = uv_coords
        else:
            mesh_obj = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors
            )
        
        # Export as GLB
        mesh_obj.export(output_path, file_type='glb')
        file_size = os.path.getsize(output_path)
        print(f"    GLB file created: {len(vertices)} vertices, {len(faces)} faces ({file_size} bytes)")
        
    except ImportError:
        print("  [Warning] trimesh not available, creating minimal GLB")
        _fallback_textured_glb(mesh, output_path)
    except Exception as e:
        print(f"  [Error] GLB export failed: {e}")
        # Fallback to OBJ
        obj_path = output_path.replace('.glb', '.obj').replace('.gltf', '.obj')
        export_textured_obj(mesh, obj_path, texture_size)


def _fallback_textured_glb(mesh: dict, output_path: str):
    """Fallback GLB export without trimesh"""
    try:
        import struct
        
        vertices = mesh['vertices']
        faces = mesh['faces']
        colors = mesh.get('colors')
        
        num_vertices = len(vertices)
        num_faces = len(faces)
        num_indices = num_faces * 3
        
        # Create vertex buffer: position (3 floats) + color (3 floats)
        vertex_data = b''
        for i in range(num_vertices):
            v = vertices[i]
            vertex_data += struct.pack('<fff', float(v[0]), float(v[1]), float(v[2]))
            if colors is not None and len(colors) > i:
                c = colors[i].astype(float) / 255.0
                vertex_data += struct.pack('<fff', float(c[0]), float(c[1]), float(c[2]))
            else:
                vertex_data += struct.pack('<fff', 0.8, 0.8, 0.8)
        
        # Create index buffer
        index_data = struct.pack(f'<{num_indices}I', *faces.flatten())
        
        # Calculate chunk sizes with padding
        vertex_chunk_size = len(vertex_data)
        index_chunk_size = len(index_data)
        vertex_padding = (8 - vertex_chunk_size % 8) % 8
        index_padding = (8 - index_chunk_size % 8) % 8
        
        # GLB header (12 bytes)
        header = struct.pack('<I3sI', 2, b'glTF', 12)
        
        # JSON chunk (minimal)
        json_content = json.dumps({
            "asset": {"version": "2.0", "generator": "texture_baking fallback"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "count": num_vertices}]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": num_vertices, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": num_indices, "type": "SCALAR"}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": vertex_chunk_size + vertex_padding, "target": 34962},
                {"buffer": 0, "byteOffset": vertex_chunk_size + vertex_padding, "byteLength": index_chunk_size + index_padding, "target": 34963}
            ],
            "buffers": [{"byteLength": vertex_chunk_size + vertex_padding + index_chunk_size + index_padding}]
        })
        json_bytes = json_content.encode('utf-8')
        json_padding = (8 - len(json_bytes) % 8) % 8
        json_bytes += b' ' * json_padding
        
        # Binary data
        binary_data = vertex_data + (b'\x00' * vertex_padding) + index_data + (b'\x00' * index_padding)
        
        # Calculate chunk headers
        json_chunk = struct.pack('<II', len(json_bytes) + 5, 0x4E4F534A) + json_bytes
        index_chunk = struct.pack('<II', len(binary_data), 0x004C4942) + binary_data
        
        # Write GLB file
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(json_chunk)
            f.write(index_chunk)
        
        file_size = os.path.getsize(output_path)
        print(f"    Fallback GLB file created: {len(vertices)} vertices, {len(faces)} faces ({file_size} bytes)")
        
    except Exception as e:
        print(f"  [Error] Fallback GLB export failed: {e}")


def export_textured_ply(mesh: dict, output_path: str):
    """Export textured PLY model"""
    vertices = mesh['vertices']
    faces = mesh.get('faces')
    colors = mesh.get('colors')
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        
        for i, v in enumerate(vertices):
            line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            
            if colors is not None and len(colors) > i:
                c = colors[i]
                line += f" {int(c[0]):d} {int(c[1]):d} {int(c[2]):d}"
            
            line += " 0.0 1.0 0.0\n"
            f.write(line)
        
        if faces is not None:
            for face in faces:
                f.write(f"face {face[0]} {face[1]} {face[2]}\n")
    
    print(f"    PLY file created: {len(vertices)} vertices")


def texture_baking_pipeline(input_dir: str, output_path: str, 
                             texture_size: int = 2048,
                             specular_strength: float = 0.5,
                             roughness: float = 0.3,
                             metallic: float = 0.1,
                             clearcoat: float = 0.5,
                             source_images: str = None):
    """Run complete texture baking pipeline"""
    print("=" * 60)
    print("  Texture Baking Pipeline")
    print("=" * 60)
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_path}")
    print(f"  Texture Size: {texture_size}")
    print("")
    
    # Step 1: Find model file
    model_file = find_model_file(input_dir)
    
    if model_file is None:
        # Try input_dir as the file path directly
        if os.path.exists(input_dir):
            model_file = input_dir
        else:
            print("[Error] No model file found")
            return None
    
    # Step 2: Load model
    mesh = load_model(model_file)
    
    if mesh is None:
        print("[Error] Failed to load model")
        return None
    
    # Step 3: Generate UV coordinates if not present
    if mesh.get('uv_coords') is None and mesh.get('faces') is not None:
        mesh['uv_coords'] = generate_uv_coords(mesh['vertices'], mesh['faces'])
    
    # Step 4: Apply material properties
    mesh = apply_material_properties(mesh, specular_strength, roughness, metallic, clearcoat)
    
    # Step 5: Create texture from source images (optional)
    if source_images:
        texture = create_texture_from_images(source_images, texture_size)
        if texture is not None:
            mesh['texture'] = texture
            print("  Texture created from source images")
    
    # Step 6: Export textured model
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    export_textured_model(mesh, output_path, texture_size)
    
    print("")
    print("  Texture baking pipeline complete!")
    print(f"  Output: {output_path}")
    
    return output_path


def main():
    args = parse_args()
    
    result = texture_baking_pipeline(
        args.input,
        args.output,
        args.texture_size,
        args.specular_strength,
        args.roughness,
        args.metallic,
        args.clearcoat,
        args.source_images
    )
    
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
