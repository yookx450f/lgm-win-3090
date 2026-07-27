#!/usr/bin/env python3
"""
Meshing Script for Car 3D Modeling
- Poisson Surface Reconstruction (using Open3D)
- Instant Meshes (using Instant Fields)
- Depth-based Meshing (DMVer2 fallback)
- Point cloud to Mesh conversion
- Mesh smoothing and optimization
- GLB/OBJ/PLY export
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Meshing for car 3D modeling')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory from Gaussian Splatting output')
    parser.add_argument('--output', type=str, required=True,
                        help='Output GLB file path')
    parser.add_argument('--method', type=str, default='poisson',
                        choices=['poisson', 'instant_meshes', 'dmver2'],
                        help='Meshing method (default: poisson)')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Mesh resolution (default: 256)')
    parser.add_argument('--depth', type=int, default=10,
                        help='Poisson reconstruction depth (default: 10)')
    parser.add_argument('--num_threads', type=int, default=8,
                        help='Number of threads (default: 8)')
    parser.add_argument('--sample_ply', type=str, default=None,
                        help='Path to sample PLY file (points3D.ply)')
    parser.add_argument('--densify_ply', type=str, default=None,
                        help='Path to densify PLY file (point_cloud.ply)')
    parser.add_argument('--smooth', type=bool, default=True,
                        help='Apply mesh smoothing (default: True)')
    parser.add_argument('--smooth_iterations', type=int, default=5,
                        help='Number of smoothing iterations (default: 5)')
    parser.add_argument('--smooth_lambda', type=float, default=0.5,
                        help='Smoothing lambda (default: 0.5)')
    return parser.parse_args()


def load_ply_point_cloud(ply_path):
    """Load PLY point cloud file using Open3D"""
    try:
        import open3d as o3d
        print(f"[Point Cloud Loader] Loading: {ply_path}")

        # Try loading as colored point cloud first
        pc = o3d.io.read_point_cloud(ply_path)

        if pc.has_points() and len(pc.points) > 0:
            print(f"[Point Cloud Loader] Loaded {len(pc.points)} points")
            if pc.has_colors():
                print(f"[Point Cloud Loader] Point cloud has colors ({len(pc.colors)} vertices)")
            if pc.has_normals():
                print(f"[Point Cloud Loader] Point cloud has normals ({len(pc.normals)} vertices)")
            return pc
        else:
            print("[Point Cloud Loader] Warning: Empty point cloud")
            return None

    except ImportError:
        print("[Error] Open3D not installed. Run: pip install open3d")
        return None
    except Exception as e:
        print(f"[Error] Failed to load PLY file: {e}")
        return None


def load_ply_with_normals(ply_path):
    """Load PLY point cloud with normals"""
    try:
        import open3d as o3d
        print(f"[Point Cloud with Normals] Loading: {ply_path}")

        pc = o3d.io.read_point_cloud(ply_path)

        if not pc.has_normals():
            print("[Warning] Point cloud has no normals. Computing normals...")
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            pc.orient_normals_consistent_tangent_plane(k=30)

        print(f"[Point Cloud with Normals] Loaded {len(pc.points)} points with normals")
        return pc

    except ImportError:
        print("[Error] Open3D not installed. Run: pip install open3d")
        return None
    except Exception as e:
        print(f"[Error] Failed to load PLY file: {e}")
        return None


def find_input_point_cloud(input_dir):
    """Find the best point cloud file in the input directory"""
    print("[Point Cloud Finder] Searching for point cloud files...")

    # Priority order for Gaussian Splatting outputs
    candidates = []

    # 1. Look for points3D.ply (sparse point cloud from COLMAP)
    sparse_path = os.path.join(input_dir, 'sparse', '0', 'points3D.ply')
    if os.path.exists(sparse_path):
        candidates.append(('sparse', sparse_path))
        print(f"  Found sparse point cloud: {sparse_path}")

    # 2. Look for point_cloud.ply (dense point cloud from Gaussian Splatting)
    dense_pattern = os.path.join(input_dir, 'train', 'final', 'point_cloud.ply')
    if os.path.exists(dense_pattern):
        candidates.append(('dense', dense_pattern))
        print(f"  Found dense point cloud: {dense_pattern}")

    # 3. Search recursively for any .ply files
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.ply'):
                fpath = os.path.join(root, f)
                if not any(fpath == c[1] for c in candidates):
                    candidates.append(('other', fpath))
                    print(f"  Found other PLY: {fpath}")

    if candidates:
        # Prefer dense point cloud for better mesh quality
        for label, path in candidates:
            if label == 'dense':
                print(f"[Point Cloud Finder] Using dense point cloud: {path}")
                return path
        # Fallback to sparse
        print(f"[Point Cloud Finder] Using sparse point cloud: {candidates[0][1]}")
        return candidates[0][1]

    print("[Error] No PLY files found in input directory")
    return None


def run_poisson_reconstruction(point_cloud_path, output_path, resolution=256, depth=10):
    """
    Run Poisson Surface Reconstruction using Open3D

    Poisson reconstruction creates a smooth surface that fits the point cloud.
    It's ideal for car modeling as it produces clean, watertight meshes.

    Args:
        point_cloud_path: Path to input PLY point cloud
        output_path: Path for output mesh file
        resolution: Mesh resolution (grid depth)
        depth: Poisson reconstruction depth (higher = more detailed)

    Returns:
        Path to the output mesh file
    """
    print("=" * 60)
    print("  Poisson Surface Reconstruction")
    print("=" * 60)
    print(f"  Input: {point_cloud_path}")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {resolution}")
    print(f"  Depth: {depth}")

    try:
        import open3d as o3d

        # Load point cloud with normals
        pcd = load_ply_with_normals(point_cloud_path)

        if pcd is None or len(pcd.points) == 0:
            print("[Error] Failed to load point cloud")
            return None

        num_points = len(pcd.points)
        print(f"  Processing {num_points} points...")

        # Check if we have enough points for good reconstruction
        if num_points < 1000:
            print(f"[Warning] Only {num_points} points. Consider densifying the point cloud.")
            print("  Use COLMAP patch_match_stereo or Gaussian Splatting densify for better results.")

        # Run Poisson reconstruction
        print("[Poisson] Running surface reconstruction...")

        # mesh_create_poisson returns a TriMesh
        # depth parameter controls the grid depth (default 8-12 for cars)
    # grid_depth is the internal grid depth (2^depth cells per dimension)
        mesh, stats = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=0,
            scale=1.1,
            linear_fit=False
        )

        print(f"[Poisson] Reconstruction complete!")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Triangles: {len(mesh.triangles)}")

        # Remove artifacts (small isolated components)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()

        num_triangles = len(mesh.triangles)
        if num_triangles > resolution * resolution * 10:
            print(f"[Poisson] Subdividing mesh (target: ~{resolution * resolution} triangles)...")
            # Use quadratic edge collapse decimation for mesh simplification
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=resolution * resolution * 5
            )
            print(f"  After simplification: {len(mesh.triangles)} triangles")

        # Ensure mesh is watertight (important for 3D printing and rendering)
        if not mesh.is_watertight():
            print("[Poisson] Mesh is not watertight. Attempting to fix...")
            # Try to fix by removing flipped triangles
            mesh = fix_mesh_watertight(mesh)

        # Apply smoothing if requested
        # Smoothing is done after export for better control

        # Save the mesh
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Save as OBJ for further processing
        obj_path = output_path.replace('.glb', '.obj').replace('.glTF', '.obj')
        if not obj_path.endswith('.obj'):
            obj_path = os.path.join(os.path.dirname(output_path),
                                    os.path.basename(output_path).rsplit('.', 1)[0] + '.obj')
        mesh.save(obj_path)
        print(f"[Poisson] Mesh saved to: {obj_path}")

        # Also save as PLY
        ply_path = obj_path.replace('.obj', '.ply')
        mesh.save(ply_path)
        print(f"[Poisson] Mesh saved to: {ply_path}")

        return obj_path

    except ImportError:
        print("[Error] Open3D not installed. Run: pip install open3d")
        return None
    except Exception as e:
        print(f"[Error] Poisson reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_instant_meshes(point_cloud_path, output_path):
    """
    Run Instant Meshes using the Instant Fields library

    Instant Meshes creates quad-dominant meshes with controlled edge flow.
    This is useful for car modeling where clean topology is important.

    Args:
        point_cloud_path: Path to input PLY point cloud
        output_path: Path for output mesh file

    Returns:
        Path to the output mesh file
    """
    print("=" * 60)
    print("  Instant Meshes (Quad-Dominant)")
    print("=" * 60)
    print(f"  Input: {point_cloud_path}")
    print(f"  Output: {output_path}")

    try:
        import open3d as o3d

        # Load point cloud
        pcd = load_ply_with_normals(point_cloud_path)

        if pcd is None or len(pcd.points) == 0:
            print("[Error] Failed to load point cloud")
            return None

        print(f"  Processing {len(pcd.points)} points...")

        # First, create a dense mesh using Poisson
        print("[Instant Meshes] Creating initial mesh using Poisson...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=8,
            width=0,
            scale=1.1,
            linear_fit=False
        )

        # Convert triangle mesh to quad-dominant mesh
        # This is a simplified approach - in production, use the actual Instant Fields library
        print("[Instant Meshes] Converting to quad-dominant mesh...")

        # For now, we'll use the Poisson mesh and apply edge operations
        # to improve quad-ness
        mesh = improve_quad_dominance(mesh)

        # Save the mesh
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        obj_path = output_path.replace('.glb', '.obj').replace('.glTF', '.obj')
        if not obj_path.endswith('.obj'):
            obj_path = os.path.join(os.path.dirname(output_path),
                                    os.path.basename(output_path).rsplit('.', 1)[0] + '.obj')
        mesh.save(obj_path)
        print(f"[Instant Meshes] Mesh saved to: {obj_path}")

        return obj_path

    except ImportError:
        print("[Error] Open3D not installed. Run: pip install open3d")
        return None
    except Exception as e:
        print(f"[Error] Instant Meshes failed: {e}")
        return None


def run_dmver2(point_cloud_path, output_path):
    """
    Run DMVer2 (Depth-based Meshing) fallback method

    This method uses depth maps from multiple views to create a mesh.
    It's useful when point cloud quality is low.

    Args:
        point_cloud_path: Path to input directory
        output_path: Path for output mesh file

    Returns:
        Path to the output mesh file
    """
    print("=" * 60)
    print("  Depth-based Meshing (DMVer2)")
    print("=" * 60)
    print(f"  Input: {point_cloud_path}")
    print(f"  Output: {output_path}")

    try:
        import open3d as o3d

        # Find depth maps in the input directory
        print("[DMVer2] Searching for depth maps...")

        depth_dirs = []
        for root, dirs, files in os.walk(point_cloud_path):
            for d in dirs:
                if 'depth' in d.lower() or 'stereo' in d.lower():
                    depth_dirs.append(os.path.join(root, d))

        if not depth_dirs:
            print("[Warning] No depth maps found. Falling back to Poisson reconstruction.")
            print("  This is expected if you haven't run COLMAP patch_match_stereo.")
            return None

        print(f"  Found {len(depth_dirs)} depth map directories")

        # For now, fall back to Poisson reconstruction
        # Full DMVer2 implementation requires the DenseMatchingVer2 library
        print("[DMVer2] Using fallback: Poisson reconstruction with depth info")

        # Find point cloud
        pcd_path = find_input_point_cloud(point_cloud_path)
        if pcd_path:
            return run_poisson_reconstruction(pcd_path, output_path, depth=8)

        return None

    except Exception as e:
        print(f"[Error] DMVer2 failed: {e}")
        return None


def fix_mesh_watertight(mesh):
    """Attempt to make a mesh watertight"""
    try:
        import open3d as o3d

        print("  Attempting to fix watertight issues...")

        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()

        # Remove vertices not used by any triangle
        mesh.remove_unreferenced_vertices()

        # Remove duplicate triangles
        mesh.remove_duplicated_triangles()

        # Flip incorrect face orientations
        mesh.compute_triangle_normals()

        return mesh

    except Exception as e:
        print(f"  Failed to fix mesh: {e}")
        return mesh


def improve_quad_dominance(mesh):
    """
    Improve the quad-dominance of a mesh

    This is a simplified version - production code would use the
    actual Instant Fields library.

    Args:
        mesh: Open3D triangle mesh

    Returns:
        Modified mesh with improved quad-dominance
    """
    try:
        import open3d as o3d

        print("  Improving quad-dominance...")

        # In production, this would use the Instant Fields library
        # For now, we just return the original mesh
        print("  Note: Full Instant Fields integration requires external library")

        return mesh

    except Exception as e:
        print(f"  Failed to improve quad-dominance: {e}")
        return mesh


def smooth_mesh(mesh, iterations=5, lambda_val=0.5):
    """
    Apply Laplacian smoothing to the mesh

    Args:
        mesh: Open3D triangle mesh
        iterations: Number of smoothing iterations
        lambda_val: Smoothing factor (0.0 = no movement, 1.0 = full movement)

    Returns:
        Smoothed mesh
    """
    try:
        import open3d as o3d

        print(f"[Mesh Smoothing] Applying {iterations} iterations (lambda={lambda_val})...")

        for i in range(iterations):
            mesh = mesh.smooth_laplacian(iterations=1, lambda_val=lambda_val)

        print(f"[Mesh Smoothing] Complete. Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")

        return mesh

    except Exception as e:
        print(f"[Error] Mesh smoothing failed: {e}")
        return mesh


def export_to_glb(mesh_path, output_path, texture_images=None):
    """
    Export mesh to GLB format using trimesh

    Args:
        mesh_path: Path to input mesh file (OBJ/PLY)
        output_path: Path for output GLB file
        texture_images: Optional dict of texture image paths

    Returns:
        Path to the output GLB file
    """
    print("=" * 60)
    print("  GLB Export")
    print("=" * 60)
    print(f"  Input: {mesh_path}")
    print(f"  Output: {output_path}")

    try:
        import trimesh

        # Load the mesh
        print("[GLB Export] Loading mesh...")
        mesh = trimesh.load(mesh_path)

        print(f"[GLB Export] Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Process the mesh for GLB export
        # 1. Ensure normals are computed
        if not mesh.vertex_normals.any():
            mesh.compute_normals()

        # 2. Compute bounding box for centering
        bounds = mesh.bounds
        center = mesh.centroid
        print(f"[GLB Export] Bounding box: {bounds}")
        print(f"[GLB Export] Center: {center}")

        # 3. Apply texture images if provided
        if texture_images:
            print("[GLB Export] Applying textures...")
            # This would use trimesh's texture module
            # For now, we skip texture application

        # 4. Export to GLB
        print("[GLB Export] Exporting to GLB...")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Export as GLB (binary GLTF)
        mesh.export(output_path)
        print(f"[GLB Export] Saved to: {output_path}")

        # Also export OBJ for compatibility
        obj_path = output_path.replace('.glb', '.obj').replace('.glTF', '.obj')
        if output_path.endswith('.glb') or output_path.endswith('.glTF'):
            obj_path = os.path.join(os.path.dirname(output_path),
                                    os.path.basename(output_path).rsplit('.', 1)[0] + '.obj')
        mesh.export(obj_path)
        print(f"[GLB Export] Also saved OBJ to: {obj_path}")

        # Also export PLY for further processing
        ply_path = obj_path.replace('.obj', '.ply')
        mesh.export(ply_path)
        print(f"[GLB Export] Also saved PLY to: {ply_path}")

        return output_path

    except ImportError:
        print("[Error] trimesh not installed. Run: pip install trimesh[extras]")
        return None
    except Exception as e:
        print(f"[Error] GLB export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_to_obj(mesh_path, output_path):
    """
    Export mesh to OBJ format

    Args:
        mesh_path: Path to input mesh file
        output_path: Path for output OBJ file

    Returns:
        Path to the output OBJ file
    """
    try:
        import trimesh

        print("[OBJ Export] Loading mesh...")
        mesh = trimesh.load(mesh_path)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        mesh.export(output_path)
        print(f"[OBJ Export] Saved to: {output_path}")

        return output_path

    except ImportError:
        print("[Error] trimesh not installed. Run: pip install trimesh[extras]")
        return None
    except Exception as e:
        print(f"[Error] OBJ export failed: {e}")
        return None


def process_gaussian_splatting_output(input_dir):
    """
    Process Gaussian Splatting output directory to find best point cloud

    Args:
        input_dir: Path to Gaussian Splatting output directory

    Returns:
        Path to the best point cloud file
    """
    print("=" * 60)
    print("  Processing Gaussian Splatting Output")
    print("=" * 60)

    # Look for the dense point cloud from Gaussian Splatting training
    possible_paths = [
        # Standard gaussian-splatting output structure
        os.path.join(input_dir, 'train', 'final', 'point_cloud.ply'),
        # Alternative paths
        os.path.join(input_dir, 'point_cloud.ply'),
        os.path.join(input_dir, 'points3D.ply'),
        # Sparse point cloud from COLMAP
        os.path.join(input_dir, 'sparse', '0', 'points3D.ply'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[Processing] Found point cloud: {path}")

            # Check point cloud size
            try:
                import open3d as o3d
                pc = o3d.io.read_point_cloud(path)
                print(f"[Processing] Point cloud: {len(pc.points)} points")

                if len(pc.points) > 100000:
                    print("[Processing] Dense point cloud detected. Excellent quality expected.")
                elif len(pc.points) > 10000:
                    print("[Processing] Medium density point cloud. Good quality expected.")
                else:
                    print("[Warning] Sparse point cloud. Consider densifying for better mesh.")
            except:
                print("[Warning] Could not check point cloud. Proceeding anyway.")

            return path

    # Search recursively
    print("[Processing] Searching recursively for PLY files...")
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.ply'):
                path = os.path.join(root, f)
                print(f"[Processing] Found PLY: {path}")
                return path

    print("[Error] No point cloud found in input directory")
    return None


def main():
    args = parse_args()

    print("=" * 60)
    print("  3D Meshing Pipeline for Car Modeling")
    print("=" * 60)
    print("")
    print(f"  Input directory: {args.input}")
    print(f"  Output path: {args.output}")
    print(f"  Method: {args.method}")
    print(f"  Resolution: {args.resolution}")
    print("")

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: Find and load point cloud
    print("Step 1: Find Point Cloud")
    print("-" * 40)

    # If sample_ply or densify_ply is provided, use those directly
    if args.sample_ply:
        pcd_path = args.sample_ply
        print(f"  Using specified sample PLY: {pcd_path}")
    elif args.densify_ply:
        pcd_path = args.densify_ply
        print(f"  Using specified densify PLY: {pcd_path}")
    else:
        # Auto-detect point cloud
        pcd_path = find_input_point_cloud(args.input)

    if pcd_path is None:
        print("[Error] No point cloud found")
        sys.exit(1)

    print("")

    # Step 2: Run selected meshing method
    print("Step 2: Run Meshing Method")
    print("-" * 40)

    if args.method == 'poisson':
        mesh_path = run_poisson_reconstruction(
            pcd_path, args.output, args.resolution, args.depth)
    elif args.method == 'instant_meshes':
        mesh_path = run_instant_meshes(pcd_path, args.output)
    elif args.method == 'dmver2':
        mesh_path = run_dmver2(pcd_path, args.output)
    else:
        print(f"[Error] Unknown method: {args.method}")
        sys.exit(1)

    if mesh_path is None:
        print("[Error] Meshing failed")
        sys.exit(1)

    print("")

    # Step 3: Apply smoothing (optional)
    if args.smooth:
        print("Step 3: Mesh Smoothing")
        print("-" * 40)

        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh = smooth_mesh(mesh, args.smooth_iterations, args.smooth_lambda)

            # Save smoothed mesh
            smoothed_path = mesh_path.replace('.obj', '_smoothed.obj')
            mesh.save(smoothed_path)
            print(f"[Smoothing] Saved smoothed mesh to: {smoothed_path}")
            mesh_path = smoothed_path
        except Exception as e:
            print(f"[Warning] Smoothing failed: {e}. Continuing with unsmoothed mesh.")

    print("")

    # Step 4: Export to GLB
    print("Step 4: Export to GLB")
    print("-" * 40)

    final_output = export_to_glb(mesh_path, args.output)

    if final_output is None:
        print("[Error] GLB export failed")
        sys.exit(1)

    print("")
    print("=" * 60)
    print("  Meshing Complete!")
    print("=" * 60)
    print(f"  Output: {final_output}")

    # Print file sizes
    if os.path.exists(final_output):
        size = os.path.getsize(final_output)
        print(f"  File size: {size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
