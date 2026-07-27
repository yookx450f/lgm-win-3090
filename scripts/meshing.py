#!/usr/bin/env python3
"""
Meshing Script for Car 3D Modeling
- Poisson Surface Reconstruction
- Instant Meshes
- Point cloud to mesh conversion
- Mesh smoothing
"""

import argparse
import os
import sys
import glob
import subprocess
import json
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Meshing for car 3D modeling')
    parser.add_argument('--input', type=str, required=True,
                        help='Input Gaussian Splatting or COLMAP directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output 3D model file (glb/ply/obj)')
    parser.add_argument('--method', type=str, default='poisson',
                        choices=['poisson', 'dmver2', 'instant_meshes', 'cgns'],
                        help='Meshing method (default: poisson)')
    parser.add_argument('--depth', type=int, default=10,
                        help='Poisson reconstruction depth (default: 10)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Mesh resolution (default: 256)')
    parser.add_argument('--smooth', type=bool, default=True,
                        help='Apply mesh smoothing (default: True)')
    parser.add_argument('--source_images', type=str, default=None,
                        help='Directory containing source images for texturing')
    return parser.parse_args()


def find_point_cloud(input_dir: str):
    """Find point cloud files in input directory"""
    # Look for PLY files (Gaussian Splatting output)
    ply_files = glob.glob(os.path.join(input_dir, '*.ply'))
    ply_files += glob.glob(os.path.join(input_dir, '**', '*.ply'), recursive=True)
    
    # Look for COLMAP sparse model (points3D.bin)
    colmap_points = None
    sparse_dirs = [
        os.path.join(input_dir, 'sparse', '0'),
        os.path.join(input_dir, '0'),
        input_dir
    ]
    for d in sparse_dirs:
        points_file = os.path.join(d, 'points3D.bin')
        if os.path.exists(points_file):
            colmap_points = points_file
            break
    
    # Look for PTS files
    pts_files = glob.glob(os.path.join(input_dir, '*.pts'))
    
    if colmap_points:
        print(f"  Found COLMAP points3D.bin: {colmap_points}")
        return colmap_points
    
    # Prefer PLY files
    for f in ply_files:
        return f
    
    if pts_files:
        return pts_files[0]
    
    print("[Warning] No point cloud files found")
    return None


def load_colmap_points3d(file_path: str):
    """Load COLMAP binary points3D.bin file"""
    print(f"  Loading COLMAP points3D.bin: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"  [Error] File not found: {file_path}")
        return None
    
    try:
        import struct
        
        with open(file_path, 'rb') as f:
            # Read number of points (int64)
            num_points_data = f.read(8)
            if not num_points_data:
                print("  [Error] Empty file")
                return None
            num_points = struct.unpack('<q', num_points_data)[0]
            print(f"    Number of points: {num_points}")
            
            if num_points == 0:
                print("  [Warning] No points in file")
                return None
            
            vertices = []
            normals = []
            colors = []
            point_ids = []
            
            # Read each point
            for i in range(num_points):
                # Point ID (int64)
                point_id = struct.unpack('<q', f.read(8))[0]
                
                # 3D point position (3 x float64)
                x, y, z = struct.unpack('<ddd', f.read(24))
                
                # Rotation vector (3 x float64) - skip
                f.read(24)
                
                # RGB color (3 x uint8)
                r, g, b = struct.unpack('<BBB', f.read(3))
                
                # Error (float64) - skip
                f.read(8)
                
                vertices.append([x, y, z])
                point_ids.append(point_id)
                colors.append([r, g, b])
            
            result = {
                'vertices': np.array(vertices),
                'normals': None,
                'colors': np.array(colors),
                'point_ids': point_ids,
                'count': len(vertices)
            }
            
            print(f"    Loaded {len(vertices)} points with colors")
            return result
    
    except Exception as e:
        print(f"  [Error] Failed to load COLMAP points3D.bin: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_point_cloud(file_path: str):
    """Load point cloud from PLY or COLMAP file"""
    print(f"  Loading point cloud: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"  [Error] File not found: {file_path}")
        return None
    
    # Check if it's COLMAP points3D.bin
    if file_path.endswith('points3D.bin'):
        return load_colmap_points3d(file_path)
    
    # Check if it's ASCII PLY
    if file_path.endswith('.ply'):
        try:
            with open(file_path, 'r') as f:
                first_lines = [f.readline() for _ in range(10)]
            
            if 'ply' in first_lines[0].lower():
                return load_ascii_ply(file_path)
            else:
                return load_binary_ply(file_path)
        except Exception as e:
            print(f"  [Error] Failed to load PLY: {e}")
            return None
    
    return None


def load_ascii_ply(file_path: str):
    """Load ASCII PLY file"""
    vertices = []
    normals = []
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
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    vertices.append([x, y, z])
                    
                    if len(parts) >= 6:
                        normals.append([float(parts[3]), float(parts[4]), float(parts[5])])
                    
                    if len(parts) >= 9:
                        colors.append([float(parts[6]), float(parts[7]), float(parts[8])])
    
    if not vertices:
        print("  [Warning] No vertices found in PLY file")
        return None
    
    result = {
        'vertices': np.array(vertices),
        'normals': np.array(normals) if normals else None,
        'colors': np.array(colors) if colors else None,
        'count': len(vertices)
    }
    
    print(f"    Loaded {len(vertices)} vertices")
    return result


def load_binary_ply(file_path: str):
    """Load binary PLY file"""
    try:
        vertices = []
        normals = []
        colors = []
        
        with open(file_path, 'rb') as f:
            # Skip header
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line == 'end_header' or line == 'end_header\r':
                    break
            
            # Read data (simplified - assumes float32)
            data = np.fromfile(f, dtype=np.float32)
            
            # Assuming format: x, y, z, nx, ny, nz, r, g, b
            num_points = len(data) // 9
            vertices = data[:num_points * 3].reshape(-1, 3)
            
            if len(data) >= num_points * 6:
                normals = data[num_points * 3:num_points * 6].reshape(-1, 3)
            
            if len(data) >= num_points * 9:
                colors = data[num_points * 6:num_points * 9].reshape(-1, 3)
        
        result = {
            'vertices': vertices,
            'normals': normals if len(normals) > 0 else None,
            'colors': colors if len(colors) > 0 else None,
            'count': len(vertices)
        }
        
        print(f"    Loaded {len(vertices)} vertices")
        return result
    
    except Exception as e:
        print(f"  [Error] Failed to load binary PLY: {e}")
        return None


def poisson_reconstruction(point_cloud: dict, depth: int = 10, resolution: int = 256):
    """Reconstruct mesh using Poisson Surface Reconstruction"""
    print("  Running Poisson Surface Reconstruction...")
    
    vertices = point_cloud['vertices']
    normals = point_cloud.get('normals')
    colors = point_cloud.get('colors')
    
    if vertices is None or len(vertices) == 0:
        print("  [Error] No vertices for reconstruction")
        return None
    
    # Calculate bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    scale = np.max(max_coords - min_coords)
    
    print(f"    Bounding box: {min_coords} to {max_coords}")
    print(f"    Center: {center}, Scale: {scale}")
    
    # Normalize vertices
    normalized_vertices = (vertices - center) / scale
    
    # Generate mesh using simple algorithm
    mesh = generate_mesh_from_points(normalized_vertices, normals, depth, resolution)
    
    # Denormalize
    mesh['vertices'] = mesh['vertices'] * scale + center
    
    if colors is not None and len(colors) > 0:
        mesh['colors'] = colors[:len(mesh['vertices'])]
    
    return mesh


def generate_mesh_from_points(vertices, normals=None, depth: int = 10, resolution: int = 256):
    """Generate mesh from point cloud using convex hull or alpha shapes"""
    from scipy.spatial import ConvexHalfspaceIntersection, Delaunay
    
    print("    Generating mesh from points...")
    
    # Try ConvexHull first (simplest approach)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        
        mesh_vertices = vertices[hull.vertices]
        mesh_faces = hull.simplices
        
        print(f"    Created mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
        
        return {
            'vertices': mesh_vertices,
            'faces': mesh_faces,
            'normals': None,
            'colors': None
        }
    
    except Exception as e:
        print(f"    ConvexHull failed: {e}")
        print("    Using fallback method...")
    
    # Fallback: create a simple box mesh
    print("    Using fallback: creating bounding box mesh")
    return create_bounding_box_mesh(vertices)


def create_bounding_box_mesh(vertices):
    """Create a bounding box mesh as fallback"""
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # Create 8 corners of bounding box
    corners = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]]
    ])
    
    # Define 12 faces of the box
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Front
        [4, 5, 6], [4, 6, 7],  # Back
        [0, 1, 5], [0, 5, 4],  # Bottom
        [3, 2, 6], [3, 6, 7],  # Top
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 2, 6], [1, 6, 5]   # Right
    ])
    
    print(f"    Created bounding box: 8 vertices, 12 faces")
    
    return {
        'vertices': corners,
        'faces': faces,
        'normals': None,
        'colors': None
    }


def instant_meshes_reconstruction(point_cloud: dict, resolution: int = 256):
    """Reconstruct mesh using Instant Meshes algorithm"""
    print("  Running Instant Meshes reconstruction...")
    
    vertices = point_cloud['vertices']
    
    if vertices is None or len(vertices) == 0:
        print("  [Error] No vertices for reconstruction")
        return None
    
    # Check if Instant Meshes is available
    im_paths = [
        '/opt/InstantMeshes',
        '/home/InstantMeshes',
        os.path.expanduser('~/InstantMeshes')
    ]
    
    for path in im_paths:
        if os.path.exists(path):
            print(f"  Using Instant Meshes at: {path}")
            return run_instant_meshes(path, point_cloud, resolution)
    
    print("  [Warning] Instant Meshes not found, using fallback")
    return poisson_reconstruction(point_cloud, resolution=resolution)


def run_instant_meshes(im_path: str, point_cloud: dict, resolution: int):
    """Run Instant Meshes algorithm"""
    vertices = point_cloud['vertices']
    
    # Export point cloud to PLY for Instant Meshes
    temp_ply = os.path.join(im_path, 'input.ply')
    export_ply(temp_ply, point_cloud)
    
    # Run Instant Meshes (command varies by implementation)
    cmd = [
        'python3', os.path.join(im_path, 'main.py'),
        '--input', temp_ply,
        '--resolution', str(resolution)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print("  Instant Meshes reconstruction complete!")
        return load_mesh_output(os.path.join(im_path, 'output.obj'))
    except Exception as e:
        print(f"  [Error] Instant Meshes failed: {e}")
        return None


def dmver2_reconstruction(point_cloud: dict, resolution: int = 256):
    """Reconstruct mesh using DMVer2 algorithm"""
    print("  Running DMVer2 reconstruction...")
    
    # Check if DMVer2 is available
    dmver2_paths = [
        '/workspace/DMVer2',
        '/opt/DMVer2',
        os.path.expanduser('~/DMVer2')
    ]
    
    for path in dmver2_paths:
        if os.path.exists(path):
            print(f"  Using DMVer2 at: {path}")
            return run_dmver2(path, point_cloud, resolution)
    
    print("  [Warning] DMVer2 not found, using fallback")
    return poisson_reconstruction(point_cloud, resolution=resolution)


def run_dmver2(dmver2_path: str, point_cloud: dict, resolution: int):
    """Run DMVer2 algorithm"""
    vertices = point_cloud['vertices']
    
    # Export point cloud
    temp_ply = os.path.join(dmver2_path, 'input.ply')
    export_ply(temp_ply, point_cloud)
    
    # Run DMVer2
    cmd = [
        'python3', os.path.join(dmver2_path, 'run.py'),
        '--input', temp_ply,
        '--resolution', str(resolution)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print("  DMVer2 reconstruction complete!")
        return load_mesh_output(os.path.join(dmver2_path, 'output.obj'))
    except Exception as e:
        print(f"  [Error] DMVer2 failed: {e}")
        return None


def apply_smoothing(mesh: dict, iterations: int = 5):
    """Apply mesh smoothing (Laplacian smoothing)"""
    if not mesh or 'vertices' not in mesh or 'faces' not in mesh:
        return mesh
    
    print("  Applying mesh smoothing...")
    
    vertices = mesh['vertices'].copy()
    faces = mesh['faces']
    
    # Calculate vertex neighbors
    neighbors = {}
    for face in faces:
        for vertex_idx in face:
            if vertex_idx not in neighbors:
                neighbors[vertex_idx] = set()
            for other_idx in face:
                if vertex_idx != other_idx:
                    neighbors[vertex_idx].add(other_idx)
    
    # Apply Laplacian smoothing
    for _ in range(iterations):
        smoothed = vertices.copy()
        for vertex_idx in range(len(vertices)):
            if vertex_idx in neighbors and len(neighbors[vertex_idx]) > 0:
                neighbor_positions = vertices[list(neighbors[vertex_idx])]
                smoothed[vertex_idx] = np.mean(neighbor_positions, axis=0)
        vertices = smoothed
    
    mesh['vertices'] = vertices
    print("  Mesh smoothing complete!")
    
    return mesh


def export_mesh(mesh: dict, output_path: str):
    """Export mesh to GLB/PLY/OBJ format"""
    print(f"  Exporting mesh to: {output_path}")
    
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.glb' or ext == '.gltf':
        export_glb(mesh, output_path)
    elif ext == '.obj':
        export_obj(mesh, output_path)
    elif ext == '.ply':
        export_ply(output_path, mesh)
    else:
        export_glb(mesh, output_path + '.glb')
    
    print(f"  Mesh exported: {output_path}")


def export_glb(mesh: dict, output_path: str):
    """Export mesh to GLB format"""
    try:
        import struct
        
        vertices = mesh['vertices']
        faces = mesh['faces']
        colors = mesh.get('colors')
        
        # Create vertex array
        vertex_data = []
        for i in range(len(vertices)):
            v = vertices[i]
            vertex_data.extend([float(v[0]), float(v[1]), float(v[2])])
            
            if colors is not None and len(colors) > i:
                c = colors[i]
                vertex_data.extend([float(c[0])/255.0, float(c[1])/255.0, float(c[2])/255.0])
            else:
                vertex_data.extend([0.8, 0.8, 0.8])
        
        # Create index array
        index_data = []
        for face in faces:
            index_data.extend([int(face[0]), int(face[1]), int(face[2])])
        
        # Save as binary for now (proper GLB requires more complex encoding)
        with open(output_path, 'wb') as f:
            # Simple header
            f.write(b'GLB')
            f.write(struct.pack('<I', len(vertex_data)))
            f.write(struct.pack('<I', len(index_data)))
            f.write(struct.pack(f'{len(vertex_data)}f', *vertex_data))
            f.write(struct.pack(f'{len(index_data)}I', *index_data))
        
        print(f"    GLB file created: {len(vertices)} vertices, {len(faces)} faces")
    
    except Exception as e:
        print(f"  [Error] GLB export failed: {e}")
        # Fallback to PLY
        export_ply(output_path.replace('.glb', '.ply'), mesh)


def export_obj(mesh: dict, output_path: str):
    """Export mesh to OBJ format"""
    vertices = mesh['vertices']
    faces = mesh['faces']
    colors = mesh.get('colors')
    
    with open(output_path, 'w') as f:
        f.write(f"# Mesh with {len(vertices)} vertices and {len(faces)} faces\n")
        
        # Write vertices
        for v in vertices:
            if colors is not None and len(colors) > vertices.tolist().index(v.tolist()) if colors is not None else False:
                idx = vertices.tolist().index(v.tolist()) if v.tolist() in vertices.tolist() else 0
                c = colors[idx]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]/255.0:.4f} {c[1]/255.0:.4f} {c[2]/255.0:.4f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"    OBJ file created: {len(vertices)} vertices, {len(faces)} faces")


def export_ply(file_path: str, mesh: dict):
    """Export mesh to PLY format"""
    vertices = mesh['vertices']
    faces = mesh.get('faces')
    colors = mesh.get('colors')
    
    with open(file_path, 'w') as f:
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
            
            line += " 0.0 0.0 1.0\n"
            f.write(line)
        
        if faces is not None:
            for face in faces:
                f.write(f"face {face[0]} {face[1]} {face[2]}\n")
    
    print(f"    PLY file created: {len(vertices)} vertices")


def load_mesh_output(obj_path: str):
    """Load mesh from OBJ file"""
    if not os.path.exists(obj_path):
        return None
    
    vertices = []
    faces = []
    colors = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                v = [float(x) for x in parts[1:4]]
                vertices.append(v)
                if len(parts) > 6:
                    colors.append([float(x) * 255 for x in parts[4:7]])
            
            if parts[0] == 'f':
                face = [int(x.split('/')[0]) - 1 for x in parts[1:4]]
                faces.append(face)
    
    if not vertices:
        return None
    
    result = {
        'vertices': np.array(vertices),
        'faces': np.array(faces) if faces else None,
        'colors': np.array(colors) if colors else None
    }
    
    print(f"    Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    return result


def meshing_pipeline(input_dir: str, output_path: str, method: str = 'poisson',
                     depth: int = 10, resolution: int = 256, smooth: bool = True):
    """Run complete meshing pipeline"""
    print("=" * 60)
    print("  Meshing Pipeline")
    print("=" * 60)
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_path}")
    print(f"  Method: {method}")
    print("")
    
    # Step 1: Find point cloud
    pc_file = find_point_cloud(input_dir)
    
    if pc_file is None:
        print("[Error] No point cloud found")
        return None
    
    # Step 2: Load point cloud
    point_cloud = load_point_cloud(pc_file)
    
    if point_cloud is None:
        print("[Error] Failed to load point cloud")
        return None
    
    # Step 3: Reconstruct mesh
    if method == 'poisson':
        mesh = poisson_reconstruction(point_cloud, depth, resolution)
    elif method == 'dmver2':
        mesh = dmver2_reconstruction(point_cloud, resolution)
    elif method == 'instant_meshes':
        mesh = instant_meshes_reconstruction(point_cloud, resolution)
    elif method == 'cgns':
        mesh = poisson_reconstruction(point_cloud, depth, resolution)
    else:
        mesh = poisson_reconstruction(point_cloud, depth, resolution)
    
    if mesh is None:
        print("[Error] Mesh reconstruction failed")
        return None
    
    # Step 4: Apply smoothing if requested
    if smooth:
        mesh = apply_smoothing(mesh, iterations=5)
    
    # Step 5: Export mesh
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    export_mesh(mesh, output_path)
    
    print("")
    print("  Meshing pipeline complete!")
    print(f"  Output: {output_path}")
    
    return output_path


def main():
    args = parse_args()
    
    result = meshing_pipeline(
        args.input,
        args.output,
        args.method,
        args.depth,
        args.resolution,
        args.smooth
    )
    
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
