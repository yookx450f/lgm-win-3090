"""
Tests for COLMAP Module
"""

import os
import sys
import pytest
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from colmap import (
    parse_args,
    create_colmap_project,
    extract_features,
    match_features,
    reconstruct_scene,
    export_results,
    colmap_pipeline
)


class TestParseArgs:
    """Tests for COLMAP argument parsing"""

    def test_parse_args_default_values(self, monkeypatch):
        """Test argument parsing with default values"""
        monkeypatch.setattr('sys.argv', [
            'colmap.py',
            '--image_path', '/tmp/images',
            '--database_path', '/tmp/database.db',
            '--output_path', '/tmp/output'
        ])
        args = parse_args()
        assert args.image_path == '/tmp/images'
        assert args.database_path == '/tmp/database.db'
        assert args.output_path == '/tmp/output'
        assert args.feature_max_num_features == 16384
        assert args.Matching_max_error == 4.0

    def test_parse_args_custom_values(self, monkeypatch):
        """Test argument parsing with custom values"""
        monkeypatch.setattr('sys.argv', [
            'colmap.py',
            '--image_path', '/tmp/images',
            '--database_path', '/tmp/database.db',
            '--output_path', '/tmp/output',
            '--feature_max_num_features', '32768',
            '--Matching_max_error', '8.0'
        ])
        args = parse_args()
        assert args.feature_max_num_features == 32768
        assert args.Matching_max_error == 8.0


class TestCreateColmapProject:
    """Tests for COLMAP project creation"""

    def test_create_project_returns_dict(self, sample_preprocessed_dir, temp_dir):
        """Test project creation returns expected dictionary"""
        output_dir = temp_dir / "colmap_output"
        result = create_colmap_project(
            str(sample_preprocessed_dir),
            str(output_dir / "database.db"),
            str(output_dir)
        )
        assert result is not None
        assert 'images_dir' in result
        assert 'database_path' in result
        assert 'sparse_dir' in result
        assert 'dense_dir' in result
        assert 'output_path' in result

    def test_create_project_creates_directories(self, sample_preprocessed_dir, temp_dir):
        """Test that required directories are created"""
        output_dir = temp_dir / "colmap_output"
        result = create_colmap_project(
            str(sample_preprocessed_dir),
            str(output_dir / "database.db"),
            str(output_dir)
        )
        assert output_dir.exists()
        assert (output_dir / "images").exists()
        assert (output_dir / "databases").exists()
        assert (output_dir / "sparse").exists()
        assert (output_dir / "dense").exists()

    def test_create_project_no_images(self, temp_dir):
        """Test handling of missing images"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        result = create_colmap_project(
            str(empty_dir),
            str(temp_dir / "database.db"),
            str(temp_dir)
        )
        assert result is None


class TestExtractFeatures:
    """Tests for feature extraction"""

    def test_extract_features_function_exists(self):
        """Test that feature extraction function exists"""
        assert extract_features is not None

    @patch('colmap.subprocess.run')
    def test_extract_features_calls_colmap(self, mock_subprocess):
        """Test that feature extraction calls COLMAP correctly"""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="success", stderr="")
        
        result = extract_features("/tmp/images", "/tmp/database.db", 16384)
        
        assert result is True
        mock_subprocess.assert_called_once()


class TestMatchFeatures:
    """Tests for feature matching"""

    def test_match_features_function_exists(self):
        """Test that feature matching function exists"""
        assert match_features is not None

    @patch('colmap.subprocess.run')
    def test_match_features_sequential_success(self, mock_subprocess):
        """Test sequential matching success"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = match_features("/tmp/database.db", 4.0)
        
        assert result is True

    @patch('colmap.subprocess.run')
    def test_match_features_falls_back_to_spatial(self, mock_subprocess):
        """Test fallback to spatial matcher"""
        mock_subprocess.side_effect = [
            subprocess.CalledProcessError(1, "cmd"),  # sequential fails
            MagicMock(returncode=0)  # spatial succeeds
        ]
        
        result = match_features("/tmp/database.db", 4.0)
        
        assert result is True


class TestReconstructScene:
    """Tests for scene reconstruction"""

    def test_reconstruct_scene_function_exists(self):
        """Test that reconstruction function exists"""
        assert reconstruct_scene is not None


class TestExportResults:
    """Tests for result export"""

    def test_export_results_creates_summary(self, temp_dir):
        """Test that export creates summary JSON"""
        sparse_dir = temp_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy COLMAP files
        (sparse_dir / "cameras.bin").write_bytes(b"dummy")
        (sparse_dir / "images.bin").write_bytes(b"dummy")
        (sparse_dir / "points3D.bin").write_bytes(b"dummy")
        
        output_dir = temp_dir / "output"
        result = export_results(str(sparse_dir), str(output_dir))
        
        assert result is True
        summary_path = output_dir / "colmap_summary.json"
        assert summary_path.exists()

    def test_export_results_no_model(self, temp_dir):
        """Test handling of missing model"""
        sparse_dir = temp_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_dir / "output"
        
        result = export_results(str(sparse_dir), str(output_dir))
        
        assert result is False


class TestColmapPipeline:
    """Tests for complete COLMAP pipeline"""

    def test_colmap_pipeline_function_exists(self):
        """Test that pipeline function exists"""
        assert colmap_pipeline is not None

    def test_colmap_pipeline_returns_none_on_failure(self, temp_dir):
        """Test pipeline returns None on failure"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        result = colmap_pipeline(
            str(empty_dir),
            str(temp_dir / "database.db"),
            str(temp_dir)
        )
        # Should return None or handle gracefully
        assert result is None or isinstance(result, str)


class TestLoadColmapPoints3D:
    """Tests for COLMAP points3D.bin loading"""

    def test_load_colmap_points3D_with_data(self, temp_dir, colmap_points_data):
        """Test loading binary points3D data"""
        from colmap import load_colmap_points3D
        
        points_file = temp_dir / "points3D.bin"
        points_file.write_bytes(colmap_points_data)
        
        result = load_colmap_points3D(str(points_file))
        
        assert result is not None
        assert 'vertices' in result
        assert 'colors' in result
        assert result['count'] == 3

    def test_load_colmap_points3D_empty_file(self, temp_dir):
        """Test handling of empty file"""
        from colmap import load_colmap_points3D
        
        points_file = temp_dir / "points3D.bin"
        points_file.write_bytes(b"")
        
        result = load_colmap_points3D(str(points_file))
        
        assert result is None

    def test_load_colmap_points3D_nonexistent(self, temp_dir):
        """Test handling of nonexistent file"""
        from colmap import load_colmap_points3D
        
        result = load_colmap_points3D(str(temp_dir / "nonexistent.bin"))
        
        assert result is None


class TestLoadPointCloud:
    """Tests for point cloud loading"""

    def test_load_point_cloud_colmap(self, temp_dir, colmap_points_data):
        """Test loading COLMAP points3D.bin"""
        from colmap import load_point_cloud
        
        points_file = temp_dir / "points3D.bin"
        points_file.write_bytes(colmap_points_data)
        
        result = load_point_cloud(str(points_file))
        
        assert result is not None

    def test_load_point_cloud_ply(self, temp_dir, ascii_ply_content):
        """Test loading ASCII PLY file"""
        from colmap import load_point_cloud
        
        ply_file = temp_dir / "points.ply"
        ply_file.write_text(ascii_ply_content)
        
        result = load_point_cloud(str(ply_file))
        
        assert result is not None
        assert 'vertices' in result


class TestLoadAsciiPLY:
    """Tests for ASCII PLY loading"""

    def test_load_ascii_ply(self, temp_dir, ascii_ply_content):
        """Test loading ASCII PLY file"""
        from colmap import load_ascii_ply
        
        ply_file = temp_dir / "points.ply"
        ply_file.write_text(ascii_ply_content)
        
        result = load_ascii_ply(str(ply_file))
        
        assert result is not None
        assert result['count'] == 4


class TestFindPointCloud:
    """Tests for point cloud file discovery"""

    def test_find_point_cloud_ply(self, temp_dir):
        """Test finding PLY files"""
        from colmap import find_point_cloud
        
        test_dir = temp_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "points.ply").write_bytes(b"dummy")
        
        result = find_point_cloud(str(test_dir))
        
        assert result is not None

    def test_find_point_cloud_colmap(self, temp_dir):
        """Test finding COLMAP points3D.bin"""
        from colmap import find_point_cloud
        
        test_dir = temp_dir / "test"
        sparse_dir = test_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        (sparse_dir / "points3D.bin").write_bytes(b"dummy")
        
        result = find_point_cloud(str(test_dir))
        
        assert result is not None
        assert "points3D.bin" in result
