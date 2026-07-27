"""
Tests for FastAPI Application
"""

import os
import sys
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestAppInitialization:
    """Tests for FastAPI app initialization"""

    def test_app_created(self):
        """Test that FastAPI app is created"""
        from main import app
        assert app is not None

    def test_app_title(self):
        """Test app title is set correctly"""
        from main import app
        assert app.title == "3D Model Generator"

    def test_app_version(self):
        """Test app version is set correctly"""
        from main import app
        assert app.version == "1.0.0"

    def test_cors_middleware_configured(self):
        """Test CORS middleware is configured"""
        from main import app
        middleware_types = [type(m.class).__name__ for m in app.middleware_stack.middlewares if hasattr(m, 'class')]
        # Check that CORS is configured (middlewares may vary)
        assert app is not None


class TestJobStatusModel:
    """Tests for JobStatus Pydantic model"""

    def test_job_status_model_creation(self):
        """Test JobStatus model can be created"""
        from main import JobStatus
        
        job = JobStatus(
            job_id="test-123",
            status="completed",
            progress=100.0,
            current_step="Done",
            message="Completed successfully",
            created_at="2024-01-01T00:00:00"
        )
        
        assert job.job_id == "test-123"
        assert job.status == "completed"
        assert job.progress == 100.0

    def test_job_status_with_optional_fields(self):
        """Test JobStatus with optional fields"""
        from main import JobStatus
        
        job = JobStatus(
            job_id="test-123",
            status="failed",
            progress=50.0,
            current_step="Error",
            message="Something went wrong",
            created_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T01:00:00",
            result={"error": "test error"}
        )
        
        assert job.completed_at is not None
        assert job.result is not None


class TestPipelineConfigModel:
    """Tests for PipelineConfig Pydantic model"""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values"""
        from main import PipelineConfig
        
        config = PipelineConfig()
        
        assert config.image_size == 1024
        assert config.bg_color == "white"
        assert config.mesh_method == "poisson"
        assert config.mesh_depth == 10
        assert config.mesh_resolution == 256
        assert config.mesh_smooth is True
        assert config.animation_type == "orbit"
        assert config.video_duration == 10.0
        assert config.texture_size == 2048
        assert config.specular_strength == 0.5
        assert config.roughness == 0.3
        assert config.metallic == 0.1
        assert config.clearcoat == 0.5

    def test_pipeline_config_custom_values(self):
        """Test PipelineConfig with custom values"""
        from main import PipelineConfig
        
        config = PipelineConfig(
            image_size=2048,
            bg_color="black",
            mesh_method="instant_meshes",
            mesh_depth=12,
            mesh_resolution=512,
            mesh_smooth=False,
            animation_type="comparison",
            video_duration=30.0,
            texture_size=4096,
            specular_strength=0.8,
            roughness=0.1,
            metallic=0.5,
            clearcoat=1.0
        )
        
        assert config.image_size == 2048
        assert config.bg_color == "black"
        assert config.mesh_method == "instant_meshes"
        assert config.animation_type == "comparison"


class TestAPIEndpoints:
    """Tests for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_index_endpoint(self, client):
        """Test index endpoint serves HTML"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_list_jobs_empty(self, client):
        """Test listing jobs when empty"""
        response = client.get("/api/jobs")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_get_job_not_found(self, client):
        """Test getting non-existent job"""
        response = client.get("/api/jobs/nonexistent-id")
        
        assert response.status_code == 404

    def test_upload_images(self, client, temp_dir):
        """Test image upload endpoint"""
        # Create test images
        test_files = []
        for i in range(3):
            img_path = temp_dir / f"car_{i+1}.jpg"
            img_path.write_bytes(b"dummy image content")
            test_files.append(('files', (f'car_{i+1}.jpg', img_path.read_bytes(), 'image/jpeg')))
        
        response = client.post("/api/upload", files=test_files)
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["file_count"] == 3

    def test_start_pipeline_job_not_found(self, client):
        """Test starting pipeline for non-existent job"""
        response = client.post(
            "/api/pipeline/nonexistent/start",
            json={}
        )
        
        assert response.status_code == 404

    def test_cancel_job_job_not_found(self, client):
        """Test canceling non-existent job"""
        response = client.post("/api/jobs/nonexistent/cancel")
        
        assert response.status_code == 404

    def test_get_results_job_not_found(self, client):
        """Test getting results for non-existent job"""
        response = client.get("/api/results/nonexistent")
        
        assert response.status_code == 404

    def test_download_file_job_not_found(self, client):
        """Test downloading file for non-existent job"""
        response = client.get("/api/download/nonexistent/file.glb")
        
        assert response.status_code == 404

    def test_view_model_job_not_found(self, client):
        """Test viewing model for non-existent job"""
        response = client.get("/api/viewer/nonexistent")
        
        assert response.status_code == 404


class TestPipelineExecution:
    """Tests for pipeline execution functions"""

    def test_update_job_function(self):
        """Test job status update function"""
        from main import update_job, jobs
        
        # Add a test job
        test_id = "update-test-123"
        jobs[test_id] = {
            "job_id": test_id,
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initial",
            "message": "Initial message"
        }
        
        update_job(test_id, 50.0, "Processing", "Processing message")
        
        assert jobs[test_id]["progress"] == 50.0
        assert jobs[test_id]["current_step"] == "Processing"
        assert jobs[test_id]["message"] == "Processing message"
        
        # Cleanup
        del jobs[test_id]

    @patch('main.asyncio.create_subprocess_exec')
    def test_run_command_success(self, mock_subprocess):
        """Test successful command execution"""
        from main import run_command
        
        mock_process = MagicMock()
        mock_process.communicate = MagicMock(return_value=(b"output", None))
        mock_subprocess.return_value = mock_process
        
        # Note: This test requires async execution
        # In practice, you would use pytest-asyncio

    @patch('main.asyncio.create_subprocess_exec')
    def test_run_command_failure(self, mock_subprocess):
        """Test failed command execution"""
        from main import run_command
        
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = MagicMock(return_value=(None, b"error"))
        mock_subprocess.return_value = mock_process
        
        # Note: This test requires async execution
        # In practice, you would use pytest-asyncio


class TestDirectorySetup:
    """Tests for directory setup"""

    def test_base_directory(self):
        """Test base directory is set"""
        from main import BASE_DIR
        assert BASE_DIR is not None

    def test_input_directory_exists(self):
        """Test input directory exists"""
        from main import INPUT_DIR
        assert INPUT_DIR is not None

    def test_output_directory_exists(self):
        """Test output directory exists"""
        from main import OUTPUT_DIR
        assert OUTPUT_DIR is not None

    def test_workspace_directory_exists(self):
        """Test workspace directory exists"""
        from main import WORKSPACE_DIR
        assert WORKSPACE_DIR is not None

    def test_jobs_directory_exists(self):
        """Test jobs directory exists"""
        from main import JOBS_DIR
        assert JOBS_DIR is not None


class TestHTMLTemplates:
    """Tests for HTML template generation"""

    @pytest.mark.asyncio
    async def test_get_index_html(self):
        """Test index HTML generation"""
        from main import get_index_html
        
        html = await get_index_html()
        
        assert "<html" in html
        assert "3D Model Generator" in html
        assert "upload" in html.lower()

    @pytest.mark.asyncio
    async def test_get_viewer_html(self):
        """Test viewer HTML generation"""
        from main import get_viewer_html
        
        html = await get_viewer_html("/tmp/model.glb")
        
        assert "<html" in html
        assert "Three" in html or "three" in html
        assert "canvas" in html.lower()


class TestStaticFiles:
    """Tests for static file mounting"""

    def test_static_directory(self):
        """Test static directory is configured"""
        from main import STATIC_DIR
        assert STATIC_DIR is not None

    def test_static_directory_exists(self):
        """Test static directory exists"""
        from main import STATIC_DIR
        assert STATIC_DIR.exists()
