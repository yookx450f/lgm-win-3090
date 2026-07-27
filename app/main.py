"""
3D Model Generator Web API
FastAPI-based server for 3D modeling pipeline management
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add scripts directory to path
sys.path.insert(0, '/workspace/scripts')

# Import pipeline scripts
from scripts.run_pipeline import parse_args as pipeline_parse_args

app = FastAPI(
    title="3D Model Generator",
    description="Web API for generating 3D models from images",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = Path("/workspace")
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
WORKSPACE_DIR = BASE_DIR / "workspace"
JOBS_DIR = BASE_DIR / "jobs"

# Create directories
for d in [INPUT_DIR, OUTPUT_DIR, WORKSPACE_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory job storage
jobs: Dict[str, Dict[str, Any]] = {}


# ==================== Pydantic Models ====================

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 100.0
    current_step: str
    message: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class PipelineConfig(BaseModel):
    image_size: int = 1024
    bg_color: str = "white"
    mesh_method: str = "poisson"
    mesh_depth: int = 10
    mesh_resolution: int = 256
    mesh_smooth: bool = True
    animation_type: str = "orbit"
    video_duration: float = 10.0
    texture_size: int = 2048
    specular_strength: float = 0.5
    roughness: float = 0.3
    metallic: float = 0.1
    clearcoat: float = 0.5


# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main management page"""
    return HTMLResponse(await get_index_html())


@app.get("/api/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all jobs"""
    return [
        JobStatus(
            job_id=job["job_id"],
            status=job["status"],
            progress=job["progress"],
            current_step=job["current_step"],
            message=job["message"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            result=job.get("result")
        )
        for job in jobs.values()
    ]


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        current_step=job["current_step"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result")
    )


@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload car images for 3D modeling"""
    job_id = str(uuid.uuid4())
    job_input_dir = INPUT_DIR / job_id
    
    # Create input directory
    job_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    saved_files = []
    for file in files:
        if file.filename:
            # Sanitize filename
            safe_name = Path(file.filename).name
            file_path = job_input_dir / safe_name
            content = await file.read()
            file_path.write_bytes(content)
            saved_files.append(safe_name)
    
    # Create job record
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "current_step": "Waiting to start",
        "message": f"Uploaded {len(saved_files)} images",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "input_dir": str(job_input_dir),
        "result": None,
        "files": saved_files
    }
    jobs[job_id] = job
    
    return JSONResponse(content={
        "job_id": job_id,
        "status": "uploaded",
        "file_count": len(saved_files),
        "files": saved_files
    })


@app.post("/api/pipeline/{job_id}/start")
async def start_pipeline(job_id: str, config: Optional[PipelineConfig] = None):
    """Start the 3D modeling pipeline"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "pending" and job["status"] != "failed":
        raise HTTPException(status_code=400, detail=f"Job is already {job['status']}")
    
    # Update job status
    job["status"] = "running"
    job["progress"] = 0.0
    job["current_step"] = "Starting pipeline..."
    job["message"] = "Pipeline execution started"
    
    # Start pipeline in background
    asyncio.create_task(run_pipeline_async(job_id, config))
    
    return JSONResponse(content={
        "job_id": job_id,
        "status": "running"
    })


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] == "running":
        job["status"] = "failed"
        job["message"] = "Job cancelled by user"
    
    return JSONResponse(content={
        "job_id": job_id,
        "status": job["status"]
    })


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get pipeline output files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    job_output_dir = OUTPUT_DIR / job_id
    
    if not job_output_dir.exists():
        raise HTTPException(status_code=404, detail="No results available")
    
    # List output files
    files = []
    for root, dirs, filenames in os.walk(job_output_dir):
        for f in filenames:
            file_path = Path(root) / f
            rel_path = file_path.relative_to(job_output_dir)
            files.append(str(rel_path))
    
    return JSONResponse(content={
        "job_id": job_id,
        "files": files
    })


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a result file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    file_path = OUTPUT_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=f"{job_id}_{filename}",
        media_type="application/octet-stream"
    )


@app.get("/api/viewer/{job_id}")
async def view_model(job_id: str):
    """View 3D model in browser"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Find GLB/OBJ file
    job_output_dir = OUTPUT_DIR / job_id
    glb_files = list(job_output_dir.glob("*.glb"))
    obj_files = list(job_output_dir.glob("*.obj"))
    
    model_file = None
    if glb_files:
        model_file = glb_files[0]
    elif obj_files:
        model_file = obj_files[0]
    
    if not model_file:
        raise HTTPException(status_code=404, detail="No 3D model found")
    
    return HTMLResponse(await get_viewer_html(str(model_file)))


# ==================== Pipeline Execution ====================

async def run_pipeline_async(job_id: str, config: Optional[PipelineConfig] = None):
    """Run the 3D modeling pipeline asynchronously"""
    job = jobs[job_id]
    job_output_dir = OUTPUT_DIR / job_id
    workspace_job_dir = WORKSPACE_DIR / job_id
    
    try:
        # Create output directories
        job_output_dir.mkdir(parents=True, exist_ok=True)
        workspace_job_dir.mkdir(parents=True, exist_ok=True)
        
        input_dir = job["input_dir"]
        preprocessed_dir = workspace_job_dir / "preprocessed"
        colmap_output_dir = workspace_job_dir / "colmap_output"
        gs_output_dir = workspace_job_dir / "gaussian_splatting_output"
        model_file = job_output_dir / "model.glb"
        textured_model_file = job_output_dir / "model_textured.glb"
        video_file = job_output_dir / "car_comparison.mp4"
        
        # Use config or defaults
        if config is None:
            config = PipelineConfig()
        
        # Step 1: Preprocessing
        update_job(job_id, 10, "Preprocessing images...", "Running preprocessing...")
        
        import subprocess
        preprocess_cmd = [
            "python3", "/workspace/scripts/preprocess.py",
            "--input_dir", input_dir,
            "--output_dir", str(preprocessed_dir),
            "--image_size", str(config.image_size),
            "--bg_color", config.bg_color
        ]
        
        await run_command(preprocess_cmd, job_id)
        
        # Step 2: COLMAP
        update_job(job_id, 25, "Running COLMAP...", "Extracting features and estimating cameras...")
        
        database_path = workspace_job_dir / "database.db"
        colmap_cmd = [
            "python3", "/workspace/scripts/colmap.py",
            "--image_path", str(preprocessed_dir),
            "--database_path", str(database_path),
            "--output_path", str(colmap_output_dir)
        ]
        
        await run_command(colmap_cmd, job_id)
        
        # Step 3: Dense Reconstruction (optional)
        update_job(job_id, 40, "Running dense reconstruction...", "Creating dense point cloud...")
        
        stereo_path = workspace_job_dir / "stereo"
        try:
            dense_cmd = [
                "colmap", "patch_match_stereo",
                "--workspace_path", str(stereo_path),
                "--database_path", str(database_path)
            ]
            await run_command(dense_cmd, job_id, ignore_errors=True)
        except Exception:
            update_job(job_id, 40, "Dense reconstruction skipped...", "Continuing with sparse point cloud...")
        
        # Step 4: Gaussian Splatting
        update_job(job_id, 50, "Running Gaussian Splatting...", "Optimizing 3D Gaussians...")
        
        gs_cmd = [
            "python3", "/workspace/scripts/gaussian_splatting.py",
            "--source", str(colmap_output_dir),
            "--output_path", str(gs_output_dir)
        ]
        
        await run_command(gs_cmd, job_id)
        
        # Step 5: Meshing
        update_job(job_id, 70, "Running meshing...", "Creating 3D mesh...")
        
        meshing_cmd = [
            "python3", "/workspace/scripts/meshing.py",
            "--input", str(gs_output_dir),
            "--output", str(model_file),
            "--method", config.mesh_method,
            "--depth", str(config.mesh_depth),
            "--resolution", str(config.mesh_resolution),
            "--smooth", str(config.mesh_smooth).lower()
        ]
        
        await run_command(meshing_cmd, job_id, ignore_errors=True)
        
        # Step 6: Texture Baking
        update_job(job_id, 85, "Baking textures...", "Applying textures and materials...")
        
        if model_file.exists():
            texture_cmd = [
                "python3", "/workspace/scripts/texture_baking.py",
                "--input", str(model_file),
                "--output", str(textured_model_file),
                "--texture_size", str(config.texture_size),
                "--specular_strength", str(config.specular_strength),
                "--roughness", str(config.roughness),
                "--metallic", str(config.metallic),
                "--clearcoat", str(config.clearcoat)
            ]
            
            await run_command(texture_cmd, job_id, ignore_errors=True)
        
        # Step 7: Video Generation (optional)
        update_job(job_id, 95, "Generating video...", "Creating comparison video...")
        
        try:
            video_cmd = [
                "python3", "/workspace/scripts/blender_video.py",
                "--models_dir", str(job_output_dir),
                "--output_video", str(video_file),
                "--animation_type", config.animation_type,
                "--duration", str(config.video_duration)
            ]
            
            await run_command(video_cmd, job_id, ignore_errors=True)
        except Exception:
            update_job(job_id, 95, "Video generation skipped...", "You can generate video later...")
        
        # Complete
        update_job(job_id, 100, "Pipeline complete!", "3D model generation finished successfully.")
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["result"] = {
            "model": str(model_file) if model_file.exists() else None,
            "textured_model": str(textured_model_file) if textured_model_file.exists() else None,
            "video": str(video_file) if video_file.exists() else None
        }
        
    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Pipeline failed: {str(e)}"
        job["completed_at"] = datetime.now().isoformat()


def update_job(job_id: str, progress: float, current_step: str, message: str):
    """Update job status"""
    if job_id in jobs:
        jobs[job_id]["progress"] = progress
        jobs[job_id]["current_step"] = current_step
        jobs[job_id]["message"] = message


async def run_command(cmd: List[str], job_id: str, ignore_errors: bool = False):
    """Run a shell command asynchronously"""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0 and not ignore_errors:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise Exception(f"Command failed: {' '.join(cmd)}\n{error_msg}")


# ==================== HTML Templates ====================

async def get_index_html():
    """Generate the main management page HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Model Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 2rem;
            color: #e94560;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #a0a0a0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .upload-section {
            background: #16213e;
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 2px dashed #e94560;
            text-align: center;
            transition: all 0.3s;
        }
        
        .upload-section:hover {
            border-color: #ff6b6b;
            background: #1a2542;
        }
        
        .upload-section.dragover {
            border-color: #4ecca3;
            background: #1a2542;
        }
        
        .upload-btn {
            background: #e94560;
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 5px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .upload-btn:hover {
            background: #ff6b6b;
        }
        
        #fileInput {
            display: none;
        }
        
        .file-list {
            margin-top: 1rem;
            text-align: left;
        }
        
        .file-item {
            background: #0f3460;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .pipeline-section {
            background: #16213e;
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .pipeline-section h2 {
            color: #4ecca3;
            margin-bottom: 1rem;
        }
        
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .config-item label {
            display: block;
            margin-bottom: 0.3rem;
            color: #a0a0a0;
        }
        
        .config-item select,
        .config-item input {
            width: 100%;
            padding: 0.5rem;
            background: #0f3460;
            border: 1px solid #e94560;
            color: #e0e0e0;
            border-radius: 5px;
        }
        
        .run-btn {
            background: #4ecca3;
            color: #1a1a2e;
            border: none;
            padding: 1rem 2rem;
            border-radius: 5px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .run-btn:hover {
            background: #6ff5c7;
            transform: translateY(-2px);
        }
        
        .run-btn:disabled {
            background: #555;
            cursor: not-allowed;
            transform: none;
        }
        
        .jobs-section {
            background: #16213e;
            border-radius: 10px;
            padding: 2rem;
        }
        
        .jobs-section h2 {
            color: #4ecca3;
            margin-bottom: 1rem;
        }
        
        .job-card {
            background: #0f3460;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #e94560;
        }
        
        .job-card.completed {
            border-left-color: #4ecca3;
        }
        
        .job-card.failed {
            border-left-color: #ff6b6b;
        }
        
        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .job-id {
            font-family: monospace;
            color: #a0a0a0;
        }
        
        .job-status {
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.9rem;
        }
        
        .status-pending { background: #f39c12; color: #1a1a2e; }
        .status-running { background: #3498db; color: white; }
        .status-completed { background: #4ecca3; color: #1a1a2e; }
        .status-failed { background: #e74c3c; color: white; }
        
        .progress-bar {
            height: 10px;
            background: #1a1a2e;
            border-radius: 5px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #e94560, #4ecca3);
            transition: width 0.5s;
        }
        
        .job-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .job-btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .view-btn { background: #3498db; color: white; }
        .download-btn { background: #4ecca3; color: #1a1a2e; }
        .cancel-btn { background: #e74c3c; color: white; }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 3D Model Generator</h1>
        <p>複数の画像から3Dモデルを生成</p>
    </div>
    
    <div class="container">
        <div class="upload-section" id="uploadSection">
            <h2>📤 画像アップロード</h2>
            <p>車の複数角度の画像をアップロードしてください</p>
            <br>
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                ファイルを選択
            </button>
            <input type="file" id="fileInput" multiple accept="image/*">
            <div class="file-list" id="fileList"></div>
        </div>
        
        <div class="pipeline-section">
            <h2>⚙️ パイプライン設定</h2>
            <div class="config-grid">
                <div class="config-item">
                    <label>画像サイズ</label>
                    <select id="imageSize">
                        <option value="512">512px</option>
                        <option value="1024" selected>1024px</option>
                        <option value="2048">2048px</option>
                    </select>
                </div>
                <div class="config-item">
                    <label>背景色</label>
                    <select id="bgColor">
                        <option value="white" selected>白</option>
                        <option value="black">黒</option>
                        <option value="green">緑</option>
                    </select>
                </div>
                <div class="config-item">
                    <label>メッシュ手法</label>
                    <select id="meshMethod">
                        <option value="poisson" selected>Poisson</option>
                        <option value="instant_meshes">Instant Meshes</option>
                    </select>
                </div>
                <div class="config-item">
                    <label>アニメーション</label>
                    <select id="animationType">
                        <option value="orbit" selected>回転</option>
                        <option value="pan">パン</option>
                        <option value="comparison">比較</option>
                    </select>
                </div>
            </div>
            <button class="run-btn" id="runBtn" onclick="startPipeline()" disabled>
                🚀 パイプライン実行
            </button>
        </div>
        
        <div class="jobs-section">
            <h2>📋 ジョブ一覧</h2>
            <div id="jobsList"></div>
        </div>
    </div>
    
    <footer>
        <p>3D Model Generator v1.0.0 | GPU: RTX 3090</p>
    </footer>
    
    <script>
        let currentJobId = null;
        let jobs = {};
        
        // File upload handling
        const fileInput = document.getElementById('fileInput');
        const uploadSection = document.getElementById('uploadSection');
        const fileList = document.getElementById('fileList');
        const runBtn = document.getElementById('runBtn');
        
        fileInput.addEventListener('change', handleFiles);
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files);
            uploadFiles(files);
        });
        
        function handleFiles(e) {
            const files = Array.from(e.target.files);
            uploadFiles(files);
        }
        
        async function uploadFiles(files) {
            if (files.length === 0) return;
            
            const formData = new FormData();
            files.forEach(file => formData.append('files', file));
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                currentJobId = result.job_id;
                
                // Display file list
                fileList.innerHTML = result.files.map(f => 
                    '<div class="file-item"><span>' + f + '</span><span>✓</span></div>'
                ).join('');
                
                runBtn.disabled = false;
                runBtn.textContent = '🚀 パイプライン実行';
                
            } catch (error) {
                alert('アップロードに失敗しました: ' + error.message);
            }
        }
        
        async function startPipeline() {
            if (!currentJobId) return;
            
            const config = {
                image_size: parseInt(document.getElementById('imageSize').value),
                bg_color: document.getElementById('bgColor').value,
                mesh_method: document.getElementById('meshMethod').value,
                animation_type: document.getElementById('animationType').value,
                mesh_depth: 10,
                mesh_resolution: 256,
                mesh_smooth: true,
                video_duration: 10.0,
                texture_size: 2048,
                specular_strength: 0.5,
                roughness: 0.3,
                metallic: 0.1,
                clearcoat: 0.5
            };
            
            try {
                const response = await fetch('/api/pipeline/' + currentJobId + '/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                
                const result = await response.json();
                runBtn.disabled = true;
                runBtn.textContent = '⏳ 処理中...';
                
                // Start polling
                pollJobStatus(currentJobId);
                
            } catch (error) {
                alert('パイプラインの開始に失敗しました: ' + error.message);
            }
        }
        
        async function pollJobStatus(jobId) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch('/api/jobs');
                    const jobList = await response.json();
                    const job = jobList.find(j => j.job_id === jobId);
                    
                    if (job) {
                        updateJobUI(job);
                        
                        if (job.status === 'completed' || job.status === 'failed') {
                            clearInterval(interval);
                            runBtn.disabled = false;
                            runBtn.textContent = '🚀 新しいパイプライン';
                            loadJobs();
                        }
                    }
                } catch (error) {
                    console.error('Status poll failed:', error);
                }
            }, 2000);
        }
        
        function updateJobUI(job) {
            const progressBar = document.querySelector('.progress-fill');
            if (progressBar) {
                progressBar.style.width = job.progress + '%';
            }
        }
        
        async function loadJobs() {
            try {
                const response = await fetch('/api/jobs');
                jobs = await response.json();
                
                const jobsList = document.getElementById('jobsList');
                
                if (jobs.length === 0) {
                    jobsList.innerHTML = '<p style="color: #666;">ジョブがありません</p>';
                    return;
                }
                
                jobsList.innerHTML = jobs.map(job => `
                    <div class="job-card ${job.status}">
                        <div class="job-header">
                            <span class="job-id">${job.job_id.substring(0, 8)}...</span>
                            <span class="job-status status-${job.status}">${getStatusLabel(job.status)}</span>
                        </div>
                        <p>${job.message}</p>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${job.progress}%"></div>
                        </div>
                        <p style="font-size: 0.9rem; color: #a0a0a0;">${job.current_step}</p>
                        <div class="job-actions">
                            ${job.status === 'completed' ? `
                                <button class="job-btn view-btn" onclick="viewModel('${job.job_id}')">👁 表示</button>
                                <button class="job-btn download-btn" onclick="downloadModel('${job.job_id}')">📥 ダウンロード</button>
                            ` : ''}
                            ${job.status === 'running' ? `
                                <button class="job-btn cancel-btn" onclick="cancelJob('${job.job_id}')">✕ キャンセル</button>
                            ` : ''}
                        </div>
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Load jobs failed:', error);
            }
        }
        
        function getStatusLabel(status) {
            const labels = {
                'pending': '待機中',
                'running': '処理中',
                'completed': '完了',
                'failed': '失敗'
            };
            return labels[status] || status;
        }
        
        function viewModel(jobId) {
            window.open('/api/viewer/' + jobId, '_blank');
        }
        
        function downloadModel(jobId) {
            window.location.href = '/api/download/' + jobId + '/model_textured.glb';
        }
        
        async function cancelJob(jobId) {
            try {
                await fetch('/api/jobs/' + jobId + '/cancel', { method: 'POST' });
                loadJobs();
            } catch (error) {
                console.error('Cancel failed:', error);
            }
        }
        
        // Initial load
        loadJobs();
        setInterval(loadJobs, 5000);
    </script>
</body>
</html>
    """


async def get_viewer_html(model_path):
    """Generate the 3D model viewer page HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Model Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: #1a1a2e;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
        }
        
        #canvas-container {
            width: 100vw;
            height: 100vh;
        }
        
        .controls {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(22, 33, 62, 0.9);
            padding: 1rem;
            border-radius: 8px;
        }
        
        .controls button {
            background: #e94560;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            margin: 0.3rem;
        }
        
        .info {
            position: absolute;
            bottom: 1rem;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(22, 33, 62, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    
    <div class="controls">
        <button onclick="toggleWireframe()">ワイヤフレーム</button>
        <button onclick="toggleAutoRotate()">自動回転</button>
        <button onclick="resetCamera()">カメラリセット</button>
        <button onclick="window.close()">閉じる</button>
    </div>
    
    <div class="info">マウスで操作 | スクロールでズーム</div>
    
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>
    
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        // Camera
        const camera = new THREE.PerspectiveCamera(
            45, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            1000
        );
        camera.position.set(3, 2, 5);
        
        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        document.getElementById('canvas-container').appendChild(renderer.domElement);
        
        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        
        const pointLight1 = new THREE.PointLight(0xe94560, 0.5);
        pointLight1.position.set(-5, 3, 0);
        scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0x4ecca3, 0.5);
        pointLight2.position.set(5, 3, -5);
        scene.add(pointLight2);
        
        // Grid
        const gridHelper = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
        scene.add(gridHelper);
        
        // Load model
        const loader = new GLTFLoader();
        let mesh;
        
        // Get model path from URL
        const pathParts = window.location.pathname.split('/');
        const jobId = pathParts[pathParts.length - 1];
        const modelUrl = '/api/download/' + jobId + '/model_textured.glb';
        
        loader.load(modelUrl, (gltf) => {
            mesh = gltf.scene;
            
            // Center and scale
            const box = new THREE.Box3().setFromObject(mesh);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 3 / maxDim;
            mesh.scale.multiplyScalar(scale);
            
            mesh.position.sub(center.multiplyScalar(scale));
            
            // Apply materials
            mesh.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            
            scene.add(mesh);
            
        }, undefined, (error) => {
            console.error('Model loading error:', error);
            const errorDiv = document.createElement('div');
            errorDiv.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#e94560;text-align:center;';
            errorDiv.innerHTML = '<h2>モデルの読み込みに失敗しました</h2><p>処理が完了しているか確認してください</p>';
            document.body.appendChild(errorDiv);
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
        
        // Resize handler
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Global functions for buttons
        window.toggleWireframe = function() {
            if (mesh) {
                mesh.traverse((child) => {
                    if (child.isMesh) {
                        child.material.wireframe = !child.material.wireframe;
                    }
                });
            }
        };
        
        let autoRotate = false;
        window.toggleAutoRotate = function() {
            autoRotate = !autoRotate;
            controls.autoRotate = autoRotate;
            controls.autoRotateSpeed = 2.0;
        };
        
        window.resetCamera = function() {
            camera.position.set(3, 2, 5);
            camera.lookAt(0, 0, 0);
            controls.reset();
        };
    </script>
</body>
</html>
    """


# ==================== Mount Static Files ====================

# Create static directories
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files (if any)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
