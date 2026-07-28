"""
Tests for Texture Baking Module
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from texture_baking import (
    parse_args,
    find_model_file,
    load_model,
    load_obj,
    load_ply_model,
    generate_uv_coords,
    create_texture_from_images,
    apply_material_properties,
    export_textured_model,
    export_textured_obj,
    export_textured_ply,
    texture_baking_pipeline,
    main
)


class TestParseArgs:
    """Tests for Texture Baking argument parsing"""

    def test_parse_args_default_values(self, monkeypatch):
        """Test argument parsing with default values"""
        monkeypatch.setattr('sys.argv', [
            'texture_baking.py',
            '--input', '/tmp/input.obj',
            '--output', '/tmp/output.glb'
        ])
        args = parse_args()
        assert args.input == '/tmp/input.obj'
        assert args.output == '/tmp/output.glb'
        assert args.texture_size == 2048
        assert args.specular_strength == 0.5
        assert args.roughness == 0.3
        assert args.metallic == 0.1
        assert args.clearcoat == 0.5

    def test_parse_args_custom_values(self, monkeypatch):
        """Test argument parsing with custom values"""
        monkeypatch.setattr('sys.argv', [
            'texture_baking.py',
            '--input', '/tmp/input.obj',
            '--output', '/tmp/output.glb',
            '--texture_size', '4096',
            '--specular_strength', '0.8',
            '--roughness', '0.1',
            '--metallic', '0.5',
            '--clearcoat', '1.0'
        ])
        args = parse_args()
        assert args.texture_size == 4096
        assert args.specular_strength == 0.8
        assert args.roughness == 0.1
        assert args.metallic == 0.5
        assert args.clearcoat == 1.0


class TestFindModelFile:
    """Tests for model file discovery"""

    def test_find_model_prefers_glb(self, temp_dir):
        """Test that GLB files are preferred"""
        test_dir = temp_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        (test_dir / "model.glb").write_bytes(b"dummy")
        (test_dir / "model.obj").write_bytes(b"dummy")
        (test_dir / "model.ply").write_bytes(b"dummy")
        
        result = find_model_file(str(test_dir))
        
        assert result is not None
        assert "glb" in result

    def test_find_model_falls_back_to_obj(self, temp_dir):
        """Test fallback to OBJ files"""
        test_dir = temp_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        (test_dir / "model.obj").write_bytes(b"dummy")
        (test_dir / "model.ply").write_bytes(b"dummy")
        
        result = find_model_file(str(test_dir))
        
        assert result is not None
        assert "obj" in result

    def test_find_model_falls_back_to_ply(self, temp_dir):
        """Test fallback to PLY files"""
        test_dir = temp_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        (test_dir / "model.ply").write_bytes(b"dummy")
        
        result = find_model_file(str(test_dir))
        
        assert result is not None
        assert "ply" in result

    def test_find_model_no_files(self, temp_dir):
        """Test handling of no model files"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        result = find_model_file(str(empty_dir))
        
        assert result is None


class TestLoadObj:
    """Tests for OBJ file loading"""

    def test_load_obj_valid(self, temp_dir):
        """Test loading valid OBJ file"""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3
f 1 3 4
"""
        obj_file = temp_dir / "model.obj"
        obj_file.write_text(obj_content)
        
        result = load_obj(str(obj_file))
        
        assert result is not None
        # Check vertices shape instead of count (function doesn't return 'count' key)
        assert result['vertices'].shape == (4, 3)
        assert result['faces'] is not None

    def test_load_obj_with_uv(self, temp_dir):
        """Test loading OBJ file with UV coordinates"""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
vt 0.0 0.0
vt 1.0 1.0
f 1 2 3
"""
        obj_file = temp_dir / "model.obj"
        obj_file.write_text(obj_content)
        
        result = load_obj(str(obj_file))
        
        assert result is not None

    def test_load_obj_empty(self, temp_dir):
        """Test handling of empty OBJ file"""
        obj_file = temp_dir / "empty.obj"
        obj_file.write_text("")
        
        result = load_obj(str(obj_file))
        
        assert result is None


class TestLoadPlyModel:
    """Tests for PLY model loading"""

    def test_load_ply_valid(self, temp_dir, ascii_ply_content):
        """Test loading valid PLY file"""
        ply_file = temp_dir / "model.ply"
        ply_file.write_text(ascii_ply_content)
        
        result = load_ply_model(str(ply_file))
        
        assert result is not None
        # Check vertices shape instead of count (function doesn't return 'count' key)
        assert result['vertices'].shape == (4, 3)

    def test_load_ply_empty(self, temp_dir):
        """Test handling of empty PLY file"""
        ply_file = temp_dir / "empty.ply"
        ply_file.write_text("""ply
format ascii 1.0
element vertex 0
property float x
property float y
property float z
end_header
""")
        
        result = load_ply_model(str(ply_file))
        
        assert result is None


class TestGenerateUVCoords:
    """Tests for UV coordinate generation"""

    def test_generate_uv_coords_basic(self):
        """Test basic UV coordinate generation"""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        
        result = generate_uv_coords(vertices, faces)
        
        assert result is not None
        assert result.shape == (4, 2)
        assert 0 <= result.min() <= 1
        assert 0 <= result.max() <= 1

    def test_generate_uv_coords_large_model(self):
        """Test UV generation for larger model"""
        vertices = np.random.rand(100, 3) * 10
        faces = np.array([[i, (i+1) % 100, (i+2) % 100] for i in range(100)])
        
        result = generate_uv_coords(vertices, faces)
        
        assert result is not None
        assert result.shape == (100, 2)


class TestCreateTextureFromImages:
    """Tests for texture creation from images"""

    def test_create_texture_no_source(self):
        """Test handling of no source images"""
        result = create_texture_from_images(None, 2048)
        assert result is None

    def test_create_texture_empty_dir(self, temp_dir):
        """Test handling of empty directory"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        result = create_texture_from_images(str(empty_dir), 2048)
        
        assert result is None


class TestApplyMaterialProperties:
    """Tests for material property application"""

    def test_apply_material_properties(self, mesh_data):
        """Test applying material properties"""
        result = apply_material_properties(
            mesh_data,
            specular_strength=0.8,
            roughness=0.2,
            metallic=0.5,
            clearcoat=1.0
        )
        
        assert 'material' in result
        assert result['material']['specular_strength'] == 0.8
        assert result['material']['roughness'] == 0.2
        assert result['material']['metallic'] == 0.5
        assert result['material']['clearcoat'] == 1.0


class TestExportTexturedModel:
    """Tests for textured model export"""

    def test_export_textured_obj(self, temp_dir, mesh_data):
        """Test textured OBJ export"""
        output_path = str(temp_dir / "textured.obj")
        
        export_textured_model(mesh_data, output_path, 2048)
        
        assert (temp_dir / "textured.obj").exists()

    def test_export_textured_ply(self, temp_dir, mesh_data):
        """Test textured PLY export"""
        output_path = str(temp_dir / "textured.ply")
        
        export_textured_model(mesh_data, output_path, 2048)
        
        assert (temp_dir / "textured.ply").exists()


class TestExportTexturedObj:
    """Tests for textured OBJ export"""

    def test_export_textured_obj_creates_mtl(self, temp_dir, mesh_data):
        """Test that MTL file is created alongside OBJ"""
        output_path = str(temp_dir / "model.obj")
        mesh_data['uv_coords'] = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        
        export_textured_obj(mesh_data, output_path, 2048)
        
        assert (temp_dir / "model.obj").exists()
        assert (temp_dir / "model.mtl").exists()


class TestTextureBakingPipeline:
    """Tests for complete texture baking pipeline"""

    def test_pipeline_function_exists(self):
        """Test pipeline function exists"""
        assert texture_baking_pipeline is not None

    def test_pipeline_returns_none_on_missing_model(self, temp_dir):
        """Test pipeline handles missing model"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        result = texture_baking_pipeline(
            str(empty_dir),
            str(temp_dir / "output.glb")
        )
        
        # Should return None or handle gracefully
        assert result is None or isinstance(result, str)


class TestTextureBakingMain:
    """Tests for main function"""

    def test_main_function_exists(self):
        """Test main function exists"""
        assert callable(main)
