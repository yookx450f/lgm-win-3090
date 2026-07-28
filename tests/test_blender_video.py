"""
Tests for Blender Video Module
"""

import os
import sys
import subprocess
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from blender_video import (
    parse_args,
    find_models,
    create_blender_script,
    render_with_blender,
    render_with_blendnet,
    export_video_ffmpeg,
    create_placeholder_video,
    video_generation_pipeline,
    main
)


class TestParseArgs:
    """Tests for Blender Video argument parsing"""

    def test_parse_args_default_values(self, monkeypatch):
        """Test argument parsing with default values"""
        monkeypatch.setattr('sys.argv', [
            'blender_video.py',
            '--models_dir', '/tmp/models',
            '--output_video', '/tmp/output.mp4'
        ])
        args = parse_args()
        assert args.models_dir == '/tmp/models'
        assert args.output_video == '/tmp/output.mp4'
        assert args.animation_type == 'orbit'
        assert args.duration == 10.0
        assert args.resolution_width == 1920
        assert args.resolution_height == 1080
        assert args.fps == 30

    def test_parse_args_custom_values(self, monkeypatch):
        """Test argument parsing with custom values"""
        monkeypatch.setattr('sys.argv', [
            'blender_video.py',
            '--models_dir', '/tmp/models',
            '--output_video', '/tmp/output.mp4',
            '--animation_type', 'comparison',
            '--duration', '30.0',
            '--resolution_width', '3840',
            '--resolution_height', '2160',
            '--fps', '60'
        ])
        args = parse_args()
        assert args.animation_type == 'comparison'
        assert args.duration == 30.0
        assert args.resolution_width == 3840
        assert args.resolution_height == 2160
        assert args.fps == 60


class TestFindModels:
    """Tests for model file discovery"""

    def test_find_models_finds_glb(self, sample_models_dir):
        """Test finding GLB files"""
        result = find_models(str(sample_models_dir))
        
        assert len(result) > 0
        assert any(".glb" in f for f in result)

    def test_find_models_finds_obj(self, sample_models_dir):
        """Test finding OBJ files"""
        result = find_models(str(sample_models_dir))
        
        assert len(result) > 0
        assert any(".obj" in f for f in result)

    def test_find_models_finds_ply(self, sample_models_dir):
        """Test finding PLY files"""
        result = find_models(str(sample_models_dir))
        
        assert len(result) > 0
        assert any(".ply" in f for f in result)

    def test_find_models_empty_directory(self, temp_dir):
        """Test handling of empty directory"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        result = find_models(str(empty_dir))
        
        assert result == []

    def test_find_models_removes_duplicates(self, temp_dir):
        """Test that duplicate files are removed"""
        test_dir = temp_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file with same name in subdirectory
        sub_dir = test_dir / "sub"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "model.glb").write_bytes(b"dummy")
        (sub_dir / "model.glb").write_bytes(b"dummy")
        
        result = find_models(str(test_dir))
        
        # Should have unique files only
        assert len(result) == len(set(result))


class TestCreateBlenderScript:
    """Tests for Blender script creation"""

    def test_create_blender_script_creates_file(self, temp_dir):
        """Test that Blender script file is created"""
        model_files = [str(temp_dir / "model1.glb"), str(temp_dir / "model2.obj")]
        output_video = str(temp_dir / "output.mp4")
        
        result = create_blender_script(
            model_files,
            output_video,
            'orbit',
            10.0,
            1920,
            1080,
            30
        )
        
        assert result is not None
        assert result.endswith('.py')
        assert os.path.exists(result)

    def test_create_blender_script_contains_expected_content(self, temp_dir):
        """Test that generated script contains expected Blender commands"""
        model_files = [str(temp_dir / "model.glb")]
        output_video = str(temp_dir / "output.mp4")
        
        script_path = create_blender_script(
            model_files,
            output_video,
            'orbit',
            10.0,
            1920,
            1080,
            30
        )
        
        content = Path(script_path).read_text()
        
        assert "bpy" in content
        assert "resolution_x" in content
        assert "resolution_y" in content
        assert "FFMPEG" in content

    @pytest.mark.parametrize("animation_type", ['orbit', 'pan', 'fly_through', 'comparison'])
    def test_create_blender_script_all_animation_types(self, animation_type, temp_dir):
        """Test all animation types are generated correctly"""
        model_files = [str(temp_dir / "model.glb")]
        output_video = str(temp_dir / "output.mp4")
        
        script_path = create_blender_script(
            model_files,
            output_video,
            animation_type,
            10.0,
            1920,
            1080,
            30
        )
        
        content = Path(script_path).read_text()
        
        assert animation_type in content


class TestRenderWithBlender:
    """Tests for Blender rendering"""

    def test_render_with_blender_function_exists(self):
        """Test that rendering function exists"""
        assert render_with_blender is not None

    @patch('blender_video.subprocess.run')
    def test_render_with_blender_success(self, mock_subprocess):
        """Test successful rendering"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = render_with_blender("/tmp/script.py", "/tmp/output.mp4")
        
        assert result is True

    @patch('blender_video.subprocess.run')
    def test_render_with_blender_failure(self, mock_subprocess):
        """Test failed rendering"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "blender")
        
        result = render_with_blender("/tmp/script.py", "/tmp/output.mp4")
        
        assert result is False


class TestRenderWithBlendNet:
    """Tests for BlendNet rendering"""

    def test_render_with_blendnet_returns_false(self):
        """Test that BlendNet returns false (not configured)"""
        result = render_with_blendnet("/tmp/script.py", "/tmp/output.mp4")
        
        assert result is False


class TestExportVideoFFmpeg:
    """Tests for FFmpeg video export"""

    def test_export_video_ffmpeg_function_exists(self):
        """Test that FFmpeg export function exists"""
        assert export_video_ffmpeg is not None

    @patch('blender_video.subprocess.run')
    def test_export_video_ffmpeg_success(self, mock_subprocess):
        """Test successful FFmpeg export"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = export_video_ffmpeg("/tmp/output.mp4")
        
        assert result is True


class TestCreatePlaceholderVideo:
    """Tests for placeholder video creation"""

    def test_create_placeholder_video_function_exists(self):
        """Test that placeholder video function exists"""
        assert create_placeholder_video is not None

    @patch('blender_video.subprocess.run')
    def test_create_placeholder_video_success(self, mock_subprocess):
        """Test successful placeholder video creation"""
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = create_placeholder_video("/tmp/output.mp4", 10.0)
        
        assert result is True


class TestVideoGenerationPipeline:
    """Tests for complete video generation pipeline"""

    def test_pipeline_function_exists(self):
        """Test pipeline function exists"""
        assert video_generation_pipeline is not None

    @patch('blender_video.create_placeholder_video')
    def test_pipeline_creates_placeholder_when_no_models(self, mock_placeholder, temp_dir):
        """Test pipeline creates placeholder when no models found"""
        mock_placeholder.return_value = True
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        result = video_generation_pipeline(
            str(empty_dir),
            str(temp_dir / "output.mp4"),
            'orbit',
            10.0,
            1920,
            1080,
            30
        )
        
        # Should call create_placeholder_video
        assert mock_placeholder.called

    @patch('blender_video.render_with_blender')
    @patch('blender_video.create_blender_script')
    def test_pipeline_with_models(self, mock_script, mock_render, sample_models_dir, temp_dir):
        """Test pipeline with actual models"""
        mock_script.return_value = "/tmp/script.py"
        mock_render.return_value = True
        
        result = video_generation_pipeline(
            str(sample_models_dir),
            str(temp_dir / "output.mp4"),
            'orbit',
            10.0,
            1920,
            1080,
            30
        )
        
        # Should call render_with_blender
        assert mock_render.called


class TestBlenderVideoMain:
    """Tests for main function"""

    def test_main_function_exists(self):
        """Test main function exists"""
        assert callable(main)
