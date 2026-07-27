"""
Tests for Run Pipeline Module
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from run_pipeline import (
    parse_args,
    check_dependencies,
    run_preprocessing,
    run_colmap,
    run_dense_reconstruction,
    run_gaussian_splatting,
    run_meshing,
    run_texture_baking,
    run_video_generation,
    main
)


class TestParseArgs:
    """Tests for Pipeline argument parsing"""

    def test_parse_args_required_args(self, monkeypatch):
        """Test required arguments are parsed correctly"""
        monkeypatch.setattr('sys.argv', [
            'run_pipeline.py',
            '--input_dir', '/tmp/input',
            '--output_dir', '/tmp/output'
        ])
        args = parse_args()
        assert args.input_dir == '/tmp/input'
        assert args.output_dir == '/tmp/output'

    def test_parse_args_step_all(self, monkeypatch):
        """Test step='all' is parsed correctly"""
        monkeypatch.setattr('sys.argv', [
            'run_pipeline.py',
            '--input_dir', '/tmp/input',
            '--output_dir', '/tmp/output',
            '--step', 'all'
        ])
        args = parse_args()
        assert args.step == 'all'

    @pytest.mark.parametrize("step", ['preprocess', 'colmap', 'dense', 
                                       'gaussian_splatting', 'meshing', 
                                       'texture_baking', 'video'])
    def test_parse_args_all_steps(self, step, monkeypatch):
        """Test all valid step values"""
        monkeypatch.setattr('sys.argv', [
            'run_pipeline.py',
            '--input_dir', '/tmp/input',
            '--output_dir', '/tmp/output',
            '--step', step
        ])
        args = parse_args()
        assert args.step == step

    def test_parse_args_default_values(self, monkeypatch):
        """Test default values for optional arguments"""
        monkeypatch.setattr('sys.argv', [
            'run_pipeline.py',
            '--input_dir', '/tmp/input',
            '--output_dir', '/tmp/output'
        ])
        args = parse_args()
        assert args.image_size == 1024
        assert args.bg_color == 'white'
        assert args.mesh_method == 'poisson'
        assert args.mesh_depth == 10
        assert args.mesh_resolution == 256
        assert args.mesh_smooth is True
        assert args.animation_type == 'orbit'
        assert args.video_duration == 10.0
        assert args.texture_size == 2048


class TestCheckDependencies:
    """Tests for dependency checking"""

    def test_check_dependencies_function_exists(self):
        """Test that dependency check function exists"""
        assert check_dependencies is not None

    @patch('run_pipeline.subprocess.run')
    def test_check_dependencies_all_present(self, mock_subprocess):
        """Test when all dependencies are present"""
        mock_subprocess.return_value = MagicMock(returncode=0, 
                                                  stdout="version 1.0")
        
        result = check_dependencies()
        
        assert result is True

    @patch('run_pipeline.subprocess.run')
    def test_check_dependencies_missing_tools(self, mock_subprocess):
        """Test when some dependencies are missing"""
        mock_subprocess.side_effect = FileNotFoundError()
        
        result = check_dependencies()
        
        assert result is False


class TestRunPreprocessing:
    """Tests for preprocessing step execution"""

    def test_run_preprocessing_function_exists(self):
        """Test that preprocessing function exists"""
        assert run_preprocessing is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_preprocessing_success(self, mock_subprocess):
        """Test successful preprocessing execution"""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="done")
        
        result = run_preprocessing('/tmp/input', '/tmp/output', 1024, 'white')
        
        assert result is True

    @patch('run_pipeline.subprocess.run')
    def test_run_preprocessing_failure(self, mock_subprocess):
        """Test failed preprocessing execution"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        result = run_preprocessing('/tmp/input', '/tmp/output', 1024, 'white')
        
        assert result is False


class TestRunColmap:
    """Tests for COLMAP step execution"""

    def test_run_colmap_function_exists(self):
        """Test that COLMAP function exists"""
        assert run_colmap is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_colmap_success(self, mock_subprocess):
        """Test successful COLMAP execution"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = run_colmap('/tmp/preprocessed', '/tmp/output')
        
        assert result is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_colmap_failure(self, mock_subprocess):
        """Test failed COLMAP execution"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        result = run_colmap('/tmp/preprocessed', '/tmp/output')
        
        assert result is None


class TestRunDenseReconstruction:
    """Tests for dense reconstruction step execution"""

    def test_run_dense_reconstruction_function_exists(self):
        """Test that dense reconstruction function exists"""
        assert run_dense_reconstruction is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_dense_reconstruction_success(self, mock_subprocess):
        """Test successful dense reconstruction"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = run_dense_reconstruction('/tmp/colmap', '/tmp/output')
        
        assert result is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_dense_reconstruction_failure(self, mock_subprocess):
        """Test failed dense reconstruction"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        result = run_dense_reconstruction('/tmp/colmap', '/tmp/output')
        
        assert result is None


class TestRunGaussianSplatting:
    """Tests for Gaussian Splatting step execution"""

    def test_run_gaussian_splatting_function_exists(self):
        """Test that Gaussian Splatting function exists"""
        assert run_gaussian_splatting is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_gaussian_splatting_success(self, mock_subprocess):
        """Test successful Gaussian Splatting"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = run_gaussian_splatting('/tmp/colmap', '/tmp/output')
        
        assert result is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_gaussian_splatting_failure(self, mock_subprocess):
        """Test failed Gaussian Splatting"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        result = run_gaussian_splatting('/tmp/colmap', '/tmp/output')
        
        assert result is None


class TestRunMeshing:
    """Tests for meshing step execution"""

    def test_run_meshing_function_exists(self):
        """Test that meshing function exists"""
        assert run_meshing is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_meshing_success(self, mock_subprocess):
        """Test successful meshing"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = run_meshing('/tmp/gs_output', '/tmp/model.glb')
        
        assert result is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_meshing_failure(self, mock_subprocess):
        """Test failed meshing"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        result = run_meshing('/tmp/gs_output', '/tmp/model.glb')
        
        assert result is None


class TestRunTextureBaking:
    """Tests for texture baking step execution"""

    def test_run_texture_baking_function_exists(self):
        """Test that texture baking function exists"""
        assert run_texture_baking is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_texture_baking_success(self, mock_subprocess):
        """Test successful texture baking"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = run_texture_baking('/tmp/model.glb', '/tmp/textured.glb')
        
        assert result is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_texture_baking_with_source_images(self, mock_subprocess):
        """Test texture baking with source images"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = run_texture_baking(
            '/tmp/model.glb', 
            '/tmp/textured.glb',
            source_images='/tmp/images'
        )
        
        assert result is not None


class TestRunVideoGeneration:
    """Tests for video generation step execution"""

    def test_run_video_generation_function_exists(self):
        """Test that video generation function exists"""
        assert run_video_generation is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_video_generation_success(self, mock_subprocess):
        """Test successful video generation"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = run_video_generation('/tmp/models', '/tmp/output.mp4')
        
        assert result is not None

    @patch('run_pipeline.subprocess.run')
    def test_run_video_generation_failure(self, mock_subprocess):
        """Test failed video generation"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        result = run_video_generation('/tmp/models', '/tmp/output.mp4')
        
        assert result is None


class TestPipelineMain:
    """Tests for pipeline main function"""

    def test_main_function_exists(self):
        """Test main function exists"""
        assert main is not None

    @patch('run_pipeline.check_dependencies')
    def test_main_with_step_all(self, mock_check, monkeypatch, temp_dir):
        """Test main function with step='all'"""
        mock_check.return_value = True
        
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (input_dir / f"car_{i+1}.jpg").write_bytes(b"dummy")
        
        output_dir = temp_dir / "output"
        
        monkeypatch.setattr('sys.argv', [
            'run_pipeline.py',
            '--input_dir', str(input_dir),
            '--output_dir', str(output_dir),
            '--step', 'all'
        ])
        
        # Should not raise exception
        main()
