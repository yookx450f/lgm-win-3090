"""
Pytest configuration and shared fixtures for Car 3D Modeling Pipeline tests
"""

import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_dir(temp_dir):
    """Create a directory with sample images for testing"""
    input_dir = temp_dir / "input_images"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy image files (we'll test file discovery, not actual image processing)
    for i in range(5):
        img_path = input_dir / f"car_{i+1:03d}.jpg"
        img_path.write_bytes(b"dummy image content")
    
    return input_dir


@pytest.fixture
def sample_preprocessed_dir(temp_dir):
    """Create a directory with sample preprocessed images"""
    preprocessed_dir = temp_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):
        img_path = preprocessed_dir / f"preprocessed_{i+1:03d}_car_{i+1:03d}.jpg"
        img_path.write_bytes(b"dummy preprocessed image")
    
    return preprocessed_dir


@pytest.fixture
def sample_colmap_dir(temp_dir):
    """Create a directory with sample COLMAP output structure"""
    colmap_dir = temp_dir / "colmap_output"
    sparse_dir = colmap_dir / "sparse" / "0"
    images_dir = colmap_dir / "images" / "images"
    
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy COLMAP files
    (sparse_dir / "cameras.bin").write_bytes(b"dummy cameras data")
    (sparse_dir / "images.bin").write_bytes(b"dummy images data")
    (sparse_dir / "points3D.bin").write_bytes(b"dummy points data")
    
    for i in range(5):
        (images_dir / f"{i+1:03d}_car_{i+1:03d}.jpg").write_bytes(b"dummy image")
    
    return colmap_dir


@pytest.fixture
def sample_gs_output_dir(temp_dir):
    """Create a directory with sample Gaussian Splatting output"""
    gs_dir = temp_dir / "gaussian_splatting_output"
    gs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary file
    import json
    summary = {
        "status": "synthetic",
        "iterations": 30000,
        "output_path": str(gs_dir)
    }
    (gs_dir / "gs_summary.json").write_text(json.dumps(summary, indent=2))
    
    # Create dummy PLY file
    (gs_dir / "point_cloud.ply").write_text("""ply
format ascii 1.0
element vertex 0
property float x
property float y
property float z
end_header
""")
    
    return gs_dir


@pytest.fixture
def sample_mesh_dir(temp_dir):
    """Create a directory with sample mesh files"""
    mesh_dir = temp_dir / "mesh_output"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy OBJ file
    obj_content = """# Test mesh
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3
f 1 3 4
"""
    (mesh_dir / "model.obj").write_text(obj_content)
    
    return mesh_dir


@pytest.fixture
def sample_models_dir(temp_dir):
    """Create a directory with sample 3D models"""
    models_dir = temp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy model files
    (models_dir / "car1.glb").write_bytes(b"dummy glb data")
    (models_dir / "car2.obj").write_bytes(b"dummy obj data")
    (models_dir / "car3.ply").write_bytes(b"dummy ply data")
    
    return models_dir


@pytest.fixture
def colmap_points_data():
    """Create sample COLMAP points3D.bin binary data"""
    import struct
    
    # Create binary data: num_points (int64) + points data
    num_points = 3
    data = struct.pack('<q', num_points)
    
    # Point 1
    data += struct.pack('<q', 1)  # point ID
    data += struct.pack('<ddd', 0.0, 0.0, 0.0)  # position
    data += struct.pack('<ddd', 0.0, 0.0, 0.0)  # rotation (dummy)
    data += struct.pack('<BBB', 255, 255, 255)  # color
    data += struct.pack('<d', 1.0)  # error (dummy)
    
    # Point 2
    data += struct.pack('<q', 2)
    data += struct.pack('<ddd', 1.0, 0.0, 0.0)
    data += struct.pack('<ddd', 0.0, 0.0, 0.0)
    data += struct.pack('<BBB', 128, 128, 128)
    data += struct.pack('<d', 1.0)
    
    # Point 3
    data += struct.pack('<q', 3)
    data += struct.pack('<ddd', 0.5, 1.0, 0.0)
    data += struct.pack('<ddd', 0.0, 0.0, 0.0)
    data += struct.pack('<BBB', 64, 64, 64)
    data += struct.pack('<d', 1.0)
    
    return data


@pytest.fixture
def ascii_ply_content():
    """Create sample ASCII PLY file content"""
    return """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
0.0 0.0 0.0 0.0 0.0 1.0 255 255 255
1.0 0.0 0.0 0.0 0.0 1.0 128 128 128
1.0 1.0 0.0 0.0 0.0 1.0 64 64 64
0.0 1.0 0.0 0.0 0.0 1.0 255 128 64
"""


@pytest.fixture
def mesh_data():
    """Create sample mesh data dictionary with sufficient vertices and faces for validation
    
    This fixture creates a mesh that meets the minimum validation requirements:
    - MIN_VERTEX_COUNT = 100 vertices
    - MIN_FACE_COUNT = 200 faces
    - MIN_FILE_SIZE = 10240 bytes (10KB minimum for valid car model)
    """
    import numpy as np
    
    # Create a larger grid of vertices (20x20 = 400 vertices) to ensure file size > 10KB
    vertices = []
    grid_size = 20
    for i in range(grid_size):
        for j in range(grid_size):
            x = i / (grid_size - 1) * 4.0 - 2.0  # -2.0 to 2.0 (larger car proportions)
            y = j / (grid_size - 1) * 2.0 - 1.0  # -1.0 to 1.0
            # Add car-like shape with hood, cabin, trunk slopes
            z_frac = (i / (grid_size - 1))  # 0 to 1 along length
            
            # Base height
            z = 0.0
            
            # Hood slope (front, 0-25%)
            if z_frac < 0.25:
                hood_frac = z_frac / 0.25
                z = 0.3 * (1 - hood_frac)  # Slopes up towards cabin
            # Cabin area (25-75%)
            elif z_frac < 0.75:
                cabin_frac = (z_frac - 0.25) / 0.50
                # Windshield (front 15% of cabin)
                if cabin_frac < 0.15:
                    ws_frac = cabin_frac / 0.15
                    z = 0.3 + 0.5 * ws_frac  # Slopes up
                # Roof (middle 70%)
                elif cabin_frac < 0.85:
                    z = 0.8  # Roof height
                # Rear window (back 15%)
                else:
                    rw_frac = (cabin_frac - 0.85) / 0.15
                    z = 0.8 - 0.3 * rw_frac  # Slopes down
            # Trunk (75-100%)
            else:
                trunk_frac = (z_frac - 0.75) / 0.25
                z = 0.5 + 0.3 * trunk_frac  # Slopes down to trunk
            
            vertices.append([x, z, y])
    
    vertices = np.array(vertices, dtype=float)
    
    # Create faces (18x18 grid = 324 quads = 648 triangles)
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v0 = i * grid_size + j
            v1 = v0 + 1
            v2 = v0 + grid_size
            v3 = v2 + 1
            
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    faces = np.array(faces, dtype=int)
    
    # Create colors matching vertex count
    colors = np.tile([255, 128, 64], (len(vertices), 1)).astype(int)
    
    return {
        'vertices': vertices,
        'faces': faces,
        'colors': colors,
        'normals': None,
        'material': {
            'specular_strength': 0.5,
            'roughness': 0.3,
            'metallic': 0.1,
            'clearcoat': 0.5
        }
    }
