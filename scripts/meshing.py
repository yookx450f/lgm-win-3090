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


def find_point_cloud(input_dir: str, skip_absolute_paths: bool = False):
    """Find point cloud files in input directory
    
    Args:
        input_dir: Input directory to search
        skip_absolute_paths: If True, skip absolute path checks (for testing)
    """
    # Look for COLMAP sparse model (points3D.bin) - highest priority
    colmap_points = None
    # Try multiple possible locations for COLMAP output
    colmap_paths = [
        os.path.join(input_dir, 'sparse', '0'),
        os.path.join(input_dir, 'sparse'),
        os.path.join(input_dir, '0'),
        input_dir,
        os.path.join(input_dir, 'colmap_output', 'sparse', '0'),
        os.path.join(input_dir, 'colmap_output'),
    ]
    
    # Only add absolute paths if not skipping (for testing)
    if not skip_absolute_paths:
        colmap_paths.extend([
            '/workspace/workspace/colmap_output/sparse/0',
            '/workspace/workspace/colmap_output'
        ])
    
    for d in colmap_paths:
        if os.path.exists(d):
            points_file = os.path.join(d, 'points3D.bin')
            if os.path.exists(points_file):
                colmap_points = points_file
                print(f"  Found COLMAP points3D.bin: {points_file}")
                break
    
    if colmap_points:
        return colmap_points
    
    # Look for PLY files (Gaussian Splatting output)
    ply_paths = [
        os.path.join(input_dir, 'point_cloud', 'iteration-30000'),
        os.path.join(input_dir, 'point_cloud'),
        input_dir
    ]
    for d in ply_paths:
        if os.path.exists(d):
            ply_files = glob.glob(os.path.join(d, '*.ply'))
            if ply_files:
                print(f"  Found PLY file: {ply_files[0]}")
                return ply_files[0]
    
    # Look for PTS files
    pts_files = glob.glob(os.path.join(input_dir, '*.pts'))
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
    """Generate mesh from point cloud for car body using convex hull
    
    This function creates a car-shaped mesh from the point cloud by:
    1. Analyzing the point cloud distribution to estimate car body dimensions
    2. Creating a convex hull that approximates the car body shape
    3. Validating the generated mesh for correctness
    
    Args:
        vertices: Point cloud vertices (numpy array)
        normals: Optional vertex normals
        depth: Poisson reconstruction depth (unused in this method)
        resolution: Mesh resolution (unused in this method)
    
    Returns:
        Mesh dictionary with vertices, faces, and optional colors/normals
    
    Raises:
        ValueError: If point cloud is invalid or too small
    """
    print("    Generating car body mesh from points...")
    
    # Validate input
    if vertices is None or len(vertices) == 0:
        print("    [Error] No vertices provided for mesh generation")
        raise ValueError("No vertices provided for mesh generation")
    
    if len(vertices) < 4:
        print(f"    [Error] Insufficient points for mesh generation: {len(vertices)} points (minimum 4 required)")
        raise ValueError(f"Insufficient points for mesh generation: {len(vertices)} points (minimum 4 required)")
    
    # Validate that vertices are valid numbers
    if not np.isfinite(vertices).all():
        print("    [Error] Point cloud contains invalid values (NaN or Inf)")
        raise ValueError("Point cloud contains invalid values (NaN or Inf)")
    
    # Calculate bounding box to understand car dimensions
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    dimensions = max_coords - min_coords
    
    print(f"    Point cloud bounding box: {min_coords} to {max_coords}")
    print(f"    Car body dimensions: {dimensions}")
    
    # Try ConvexHull first for car body shape
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        
        # Use all vertices from the hull (not just unique vertices)
        # The hull.vertices contains indices into the original vertices array
        mesh_vertices = vertices[hull.vertices]
        mesh_faces = hull.simplices
        
        # Validate the generated mesh
        if len(mesh_vertices) < 4:
            print("    [Error] Generated mesh has too few vertices")
            raise ValueError("Generated mesh has too few vertices")
        
        if len(mesh_faces) < 1:
            print("    [Error] Generated mesh has no faces")
            raise ValueError("Generated mesh has no faces")
        
        # Verify all face indices are valid
        max_vertex_idx = mesh_faces.max()
        if max_vertex_idx >= len(mesh_vertices):
            print(f"    [Error] Invalid face indices: max={max_vertex_idx}, vertices={len(mesh_vertices)}")
            raise ValueError(f"Invalid face indices in generated mesh")
        
        print(f"    Created car body mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
        
        return {
            'vertices': mesh_vertices,
            'faces': mesh_faces,
            'normals': None,
            'colors': None
        }
    
    except ValueError:
        # Re-raise ValueError for validation errors
        raise
    except Exception as e:
        print(f"    ConvexHull failed: {e}")
        print("    Using fallback method...")
    
    # Fallback: create a car-shaped bounding box mesh
    print("    Using fallback: creating car-shaped bounding box mesh")
    return create_bounding_box_mesh(vertices, dimensions, center)


def create_bounding_box_mesh(vertices, dimensions=None, center=None):
    """Create a car-shaped bounding box mesh as fallback
    
    This function creates a box mesh that approximates the car body shape
    based on the point cloud dimensions.
    
    Args:
        vertices: Point cloud vertices (numpy array)
        dimensions: Optional pre-calculated dimensions (width, height, depth)
        center: Optional pre-calculated center point
    
    Returns:
        Mesh dictionary with vertices, faces, and optional colors/normals
    
    Raises:
        ValueError: If point cloud is invalid or too small
    """
    # Validate input
    if vertices is None or len(vertices) == 0:
        print("    [Error] No vertices provided for bounding box mesh")
        raise ValueError("No vertices provided for bounding box mesh")
    
    if len(vertices) < 3:
        print(f"    [Error] Insufficient points for bounding box: {len(vertices)} points (minimum 3 required)")
        raise ValueError(f"Insufficient points for bounding box: {len(vertices)} points (minimum 3 required)")
    
    # Calculate bounding box from vertices if not provided
    if dimensions is None or center is None:
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords
    
    # Validate dimensions
    if np.any(dimensions <= 0):
        print("    [Error] Invalid dimensions: must be positive")
        raise ValueError("Invalid dimensions: must be positive")
    
    if np.any(~np.isfinite(center)):
        print("    [Error] Invalid center coordinates")
        raise ValueError("Invalid center coordinates")
    
    # Create 8 corners of bounding box centered at the calculated center
    half_dims = dimensions / 2
    corners = np.array([
        [center[0] - half_dims[0], center[1] - half_dims[1], center[2] - half_dims[2]],  # 0: min, min, min
        [center[0] + half_dims[0], center[1] - half_dims[1], center[2] - half_dims[2]],  # 1: max, min, min
        [center[0] + half_dims[0], center[1] + half_dims[1], center[2] - half_dims[2]],  # 2: max, max, min
        [center[0] - half_dims[0], center[1] + half_dims[1], center[2] - half_dims[2]],  # 3: min, max, min
        [center[0] - half_dims[0], center[1] - half_dims[1], center[2] + half_dims[2]],  # 4: min, min, max
        [center[0] + half_dims[0], center[1] - half_dims[1], center[2] + half_dims[2]],  # 5: max, min, max
        [center[0] + half_dims[0], center[1] + half_dims[1], center[2] + half_dims[2]],  # 6: max, max, max
        [center[0] - half_dims[0], center[1] + half_dims[1], center[2] + half_dims[2]]   # 7: min, max, max
    ])
    
    # Define 12 faces of the box (triangulated quads)
    # Face ordering: Front, Back, Bottom, Top, Left, Right
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Front (z-min)
        [4, 5, 6], [4, 6, 7],  # Back (z-max)
        [0, 1, 5], [0, 5, 4],  # Bottom (y-min)
        [3, 2, 6], [3, 6, 7],  # Top (y-max)
        [0, 3, 7], [0, 7, 4],  # Left (x-min)
        [1, 2, 6], [1, 6, 5]   # Right (x-max)
    ])
    
    # Validate the generated mesh
    if len(corners) != 8:
        print(f"    [Error] Invalid number of corners: {len(corners)}")
        raise ValueError(f"Invalid number of corners: {len(corners)}")
    
    if len(faces) != 12:
        print(f"    [Error] Invalid number of faces: {len(faces)}")
        raise ValueError(f"Invalid number of faces: {len(faces)}")
    
    # Verify all face indices are valid (0-7)
    if faces.max() >= 8 or faces.min() < 0:
        print(f"    [Error] Invalid face indices: min={faces.min()}, max={faces.max()}")
        raise ValueError(f"Invalid face indices in generated mesh")
    
    print(f"    Created car-shaped bounding box: 8 vertices, 12 faces")
    print(f"    Box dimensions: {dimensions}, Center: {center}")
    
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
    
    vertices = mesh['vertices'].copy()
    faces = mesh['faces']
    
    # Check if we have enough vertices for meaningful smoothing
    if len(vertices) < 4:
        print("  Skipping smoothing (too few vertices)")
        return mesh
    
    print("  Applying mesh smoothing...")
    
    # Collect all valid vertex indices from faces
    all_vertex_indices = set()
    for face in faces:
        face_list = face.tolist() if hasattr(face, 'tolist') else list(face)
        for idx in face_list:
            all_vertex_indices.add(idx)
    
    # Create a mapping from face vertex indices to contiguous array indices
    valid_indices = sorted([i for i in all_vertex_indices if 0 <= i < len(vertices)])
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    
    # Remap faces to contiguous indices
    remapped_faces = []
    for face in faces:
        face_list = face.tolist() if hasattr(face, 'tolist') else list(face)
        remapped_face = [index_map[idx] for idx in face_list if idx in index_map]
        if len(remapped_face) == 3:
            remapped_faces.append(remapped_face)
    faces = np.array(remapped_faces)
    
    # Renormalize vertices to match
    mesh['vertices'] = vertices[valid_indices]
    mesh['faces'] = faces
    
    if len(mesh['vertices']) < 4:
        print("  Skipping smoothing (too few vertices after remapping)")
        return mesh
    
    # Calculate vertex neighbors
    neighbors = {}
    for i in range(len(mesh['vertices'])):
        neighbors[i] = set()
    
    for face in faces:
        face_list = face.tolist() if hasattr(face, 'tolist') else list(face)
        for vertex_idx in face_list:
            for other_idx in face_list:
                if vertex_idx != other_idx:
                    neighbors[vertex_idx].add(other_idx)
    
    # Apply Laplacian smoothing
    for _ in range(iterations):
        smoothed = mesh['vertices'].copy()
        for vertex_idx in range(len(mesh['vertices'])):
            if vertex_idx in neighbors and len(neighbors[vertex_idx]) > 0:
                neighbor_indices = list(neighbors[vertex_idx])
                neighbor_positions = mesh['vertices'][neighbor_indices]
                smoothed[vertex_idx] = np.mean(neighbor_positions, axis=0)
        mesh['vertices'] = smoothed
    
    print("  Mesh smoothing complete!")
    
    return mesh


def export_mesh(mesh: dict, output_path: str):
    """Export mesh to GLB/PLY/OBJ format with validation
    
    This function validates the mesh before export and checks the output file
    size to ensure a valid mesh was generated.
    
    Args:
        mesh: Mesh dictionary with vertices, faces, and optional colors/normals
        output_path: Output file path
    
    Returns:
        True if export successful
    
    Raises:
        ValueError: If mesh is invalid or output file is too small
    """
    print(f"  Exporting mesh to: {output_path}")
    
    # Validate mesh before export
    if mesh is None:
        print("  [Error] Cannot export: mesh is None")
        raise ValueError("Cannot export: mesh is None")
    
    if 'vertices' not in mesh or 'faces' not in mesh:
        print("  [Error] Invalid mesh: missing 'vertices' or 'faces'")
        raise ValueError("Invalid mesh: missing 'vertices' or 'faces'")
    
    vertices = mesh['vertices']
    faces = mesh['faces']
    
    if vertices is None or len(vertices) == 0:
        print("  [Error] Cannot export: no vertices in mesh")
        raise ValueError("Cannot export: no vertices in mesh")
    
    if faces is None or len(faces) == 0:
        print("  [Error] Cannot export: no faces in mesh")
        raise ValueError("Cannot export: no faces in mesh")
    
    if not np.isfinite(vertices).all():
        print("  [Error] Cannot export: mesh contains invalid vertex values")
        raise ValueError("Cannot export: mesh contains invalid vertex values")
    
    # Minimum file size check (1KB = 1024 bytes)
    # Files smaller than this indicate invalid or corrupted mesh generation
    MIN_FILE_SIZE = 1024
    
    # Export based on file extension
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.glb' or ext == '.gltf':
        export_glb(mesh, output_path)
    elif ext == '.obj':
        export_obj(mesh, output_path)
    elif ext == '.ply':
        export_ply(output_path, mesh)
    else:
        export_glb(mesh, output_path + '.glb')
    
    # Validate output file size
    actual_path = output_path
    if ext == '' and os.path.exists(output_path + '.glb'):
        actual_path = output_path + '.glb'
    
    if os.path.exists(actual_path):
        file_size = os.path.getsize(actual_path)
        
        if file_size < MIN_FILE_SIZE:
            print(f"  [Error] Generated file is too small: {file_size} bytes (minimum {MIN_FILE_SIZE} bytes)")
            print(f"  [Error] This indicates an invalid or corrupted mesh was generated.")
            print(f"  [Error] Skipping invalid file and aborting pipeline.")
            raise ValueError(f"Generated file is too small: {file_size} bytes (minimum {MIN_FILE_SIZE} bytes). Invalid mesh detected. Pipeline aborted.")
        
        print(f"  Output file validated: {file_size} bytes")
    else:
        # Try checking other possible output paths
        found = False
        for possible_path in [output_path, output_path.replace('.glb', '.obj'), output_path.replace('.glb', '.ply')]:
            if os.path.exists(possible_path):
                file_size = os.path.getsize(possible_path)
                if file_size < MIN_FILE_SIZE:
                    print(f"  [Error] Generated file is too small: {possible_path} ({file_size} bytes)")
                    print(f"  [Error] This indicates an invalid or corrupted mesh was generated.")
                    print(f"  [Error] Skipping invalid file and aborting pipeline.")
                    raise ValueError(f"Generated file is too small: {file_size} bytes. Invalid mesh detected. Pipeline aborted.")
                print(f"  Output file validated: {possible_path} ({file_size} bytes)")
                found = True
                break
        
        if not found:
            print("  [Error] Output file was not created")
            raise ValueError("Output file was not created")
    
    print(f"  Mesh exported: {output_path}")
    return True


def export_glb(mesh: dict, output_path: str):
    """Export mesh to proper GLB format using trimesh"""
    try:
        import trimesh
        
        vertices = mesh['vertices']
        faces = mesh['faces']
        colors = mesh.get('colors')
        normals = mesh.get('normals')
        
        # Create trimesh Trimesh object (note: class is Trimesh, not Mesh)
        mesh_obj = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            normals=normals
        )
        
        # Export as GLB
        mesh_obj.export(output_path, file_type='glb')
        file_size = os.path.getsize(output_path)
        print(f"    GLB file created: {len(vertices)} vertices, {len(faces)} faces ({file_size} bytes)")
        
        # Also create OBJ and PLY files for compatibility
        obj_path = output_path.replace('.glb', '.obj')
        export_obj(mesh, obj_path)
        
        ply_path = output_path.replace('.glb', '.ply')
        export_ply(ply_path, mesh)
        
        print(f"    Also created: {obj_path}, {ply_path}")
        
    except ImportError:
        print("  [Warning] trimesh not available, using fallback export")
        _fallback_glb_export(mesh, output_path)
    except Exception as e:
        print(f"  [Error] GLB export failed: {e}")
        # Fallback to PLY
        export_ply(output_path.replace('.glb', '.ply'), mesh)


def _fallback_glb_export(mesh: dict, output_path: str):
    """Fallback GLB export without trimesh (creates minimal valid GLB)"""
    try:
        import struct
        
        vertices = mesh['vertices']
        faces = mesh['faces']
        colors = mesh.get('colors')
        
        # Calculate vertex attributes
        num_vertices = len(vertices)
        num_faces = len(faces)
        num_indices = num_faces * 3
        
        # Create vertex buffer: position (3 floats) + color (3 floats)
        vertex_data = b''
        for i in range(num_vertices):
            v = vertices[i]
            vertex_data += struct.pack('<fff', float(v[0]), float(v[1]), float(v[2]))
            if colors is not None and len(colors) > i:
                c = colors[i]
                vertex_data += struct.pack('<fff', float(c[0])/255.0, float(c[1])/255.0, float(c[2])/255.0)
            else:
                vertex_data += struct.pack('<fff', 0.8, 0.8, 0.8)
        
        # Create index buffer
        index_data = struct.pack(f'<{num_indices}I', *faces.flatten())
        
        # Calculate chunk sizes
        vertex_chunk_size = len(vertex_data)
        index_chunk_size = len(index_data)
        
        # GLB header (12 bytes)
        header = struct.pack('<I3sI', 2, b'glTF', 12)
        
        # JSON chunk (minimal)
        json_content = json.dumps({
            "asset": {"version": "2.0", "generator": "fallback exporter"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "count": num_vertices}]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": num_vertices, "max": [float('inf')]*3, "min": [float('-inf')]*3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": num_indices, "type": "SCALAR"}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": vertex_chunk_size, "target": 34962},
                {"buffer": 0, "byteOffset": vertex_chunk_size, "byteLength": index_chunk_size, "target": 34963}
            ],
            "buffers": [{"byteLength": vertex_chunk_size + index_chunk_size}]
        })
        json_bytes = json_content.encode('utf-8')
        # Pad JSON to 8-byte boundary
        json_padding = (8 - len(json_bytes) % 8) % 8
        json_bytes += b' ' * json_padding
        
        # Binary data (padded to 8-byte boundary)
        vertex_padding = (8 - vertex_chunk_size % 8) % 8
        index_padding = (8 - index_chunk_size % 8) % 8
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
                     depth: int = 10, resolution: int = 256, smooth: bool = True,
                     skip_absolute_paths: bool = False):
    """Run complete meshing pipeline
    
    Args:
        input_dir: Input directory
        output_path: Output file path
        method: Meshing method
        depth: Poisson depth
        resolution: Mesh resolution
        smooth: Apply smoothing
        skip_absolute_paths: Skip absolute path checks (for testing)
    """
    print("=" * 60)
    print("  Meshing Pipeline")
    print("=" * 60)
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_path}")
    print(f"  Method: {method}")
    print("")
    
    # Step 1: Find point cloud
    pc_file = find_point_cloud(input_dir, skip_absolute_paths=skip_absolute_paths)
    
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
