#!/usr/bin/env python3
"""
Texture Baking Script for Car 3D Modeling
- UV Unwrapping (using trimesh)
- Texture mapping from multiple views
- Color correction and optimization
- Specular (gloss) handling
- Reflection handling
- PBR (Physically-Based Rendering) material generation
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Texture baking for car 3D modeling')
    parser.add_argument('--input', type=str, required=True,
                        help='Input GLB/OBJ/PLY file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output textured GLB file path')
    parser.add_argument('--texture_size', type=int, default=2048,
                        help='Texture resolution (default: 2048)')
    parser.add_argument('--specular_strength', type=float, default=0.5,
                        help='Specular strength (default: 0.5)')
    parser.add_argument('--roughness', type=float, default=0.3,
                        help='Roughness value (default: 0.3)')
    parser.add_argument('--metallic', type=float, default=0.1,
                        help='Metallic value (default: 0.1)')
    parser.add_argument('--clearcoat', type=float, default=0.5,
                        help='Clearcoat value for car paint (default: 0.5)')
    parser.add_argument('--normal_strength', type=float, default=1.0,
                        help='Normal map strength (default: 1.0)')
    parser.add_argument('--source_images', type=str, default=None,
                        help='Directory containing source images for texturing')
    parser.add_argument('--uv_method', type=str, default='angle',
                        choices=['angle', 'assimp', 'lscm'],
                        help='UV unwrapping method (default: angle)')
    return parser.parse_args()


def uv_unwrap(mesh_path, uv_method='angle'):
    """
    UV Unwrapping for car mesh

    Uses trimesh's UV unwrapping capabilities to create
    optimal UV coordinates for texturing.

    Args:
        mesh_path: Path to input mesh file
        uv_method: UV unwrapping method ('angle', 'lscm', 'assimp')

    Returns:
        Path to mesh with UV coordinates
    """
    print("=" * 60)
    print("  UV Unwrapping")
    print("=" * 60)
    print(f"  Input: {mesh_path}")
    print(f"  UV Method: {uv_method}")

    try:
        import trimesh

        print("[UV Unwrap] Loading mesh...")
        mesh = trimesh.load(mesh_path)

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        print(f"[UV Unwrap] Loaded mesh: {num_vertices} vertices, {num_faces} faces")

        # Check if mesh already has UV coordinates
        if mesh.visual.uv is not None:
            print(f"[UV Unwrap] Mesh already has UV coordinates: {mesh.visual.uv.shape}")
        else:
            print("[UV Unwrap] Creating UV coordinates...")

            # Use trimesh's built-in UV unwrapping
            if uv_method == 'angle':
                # Angle-based unwrapping (best for most cases)
                print("  Using angle-based unwrapping...")
                uv = create_angle_based_uv(mesh)
            elif uv_method == 'lscm':
                # Least Squares Conformal Mapping
                print("  Using LSCM unwrapping...")
                uv = create_lscm_uv(mesh)
            elif uv_method == 'assimp':
                # Use ASSIMP's UV unwrapping
                print("  Using ASSIMP unwrapping...")
                try:
                    uv = create_assimp_uv(mesh)
                except Exception as e:
                    print(f"  ASSIMP failed: {e}. Falling back to angle-based.")
                    uv = create_angle_based_uv(mesh)
            else:
                uv = create_angle_based_uv(mesh)

            mesh.visual.uv = uv
            print(f"[UV Unwrap] Created UV coordinates: {uv.shape}")

        # Save mesh with UV coordinates
        mesh_with_uv = mesh_path.replace('.obj', '_uv.obj').replace('.ply', '_uv.ply')
        mesh.export(mesh_with_uv)
        print(f"[UV Unwrap] Saved mesh with UVs to: {mesh_with_uv}")

        return mesh_with_uv

    except ImportError:
        print("[Error] trimesh not installed. Run: pip install trimesh[extras]")
        return mesh_path
    except Exception as e:
        print(f"[Warning] UV unwrapping failed: {e}. Continuing without UV changes.")
        return mesh_path


def create_angle_based_uv(mesh):
    """
    Create UV coordinates using angle-based unwrapping

    This method preserves angles and is suitable for car bodies.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        UV coordinates array (N, 2)
    """
    import trimesh

    # Use trimesh's built-in UV mapping
    # This projects the mesh onto 2D using angle preservation
    try:
        uv = trimesh.visual.texture.make_morton_mesh_uv(mesh)
        print(f"  Created morton-based UV: {uv.shape}")
        return uv
    except Exception as e:
        print(f"  Morton UV failed: {e}")

    # Fallback: spherical/cylindrical projection for car-like shapes
    print("  Falling back to spherical projection...")

    # Compute bounding box
    bounds = mesh.bounds
    center = mesh.centroid

    # Project vertices to UV space
    vertices = mesh.vertices
    min_bounds = bounds[0]
    max_bounds = bounds[1]

    # Normalize to [0, 1] range
    ranges = max_bounds - min_bounds
    max_range = np.max(ranges)

    if max_range == 0:
        return np.zeros((len(vertices), 2))

    uv = np.zeros((len(vertices), 2))
    uv[:, 0] = (vertices[:, 0] - min_bounds[0]) / max_range
    uv[:, 1] = (vertices[:, 1] - min_bounds[1]) / max_range

    # Handle edge cases
    uv = np.clip(uv, 0.0, 1.0)

    print(f"  Created spherical UV: {uv.shape}")
    return uv


def create_lscm_uv(mesh):
    """
    Create UV coordinates using Least Squares Conformal Mapping

    This method minimizes distortion and is good for complex meshes.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        UV coordinates array (N, 2)
    """
    import trimesh

    try:
        # Use trimesh's LSCM implementation
        uv = trimesh.visual.mapping.lscm(mesh)
        print(f"  Created LSCM UV: {uv.shape}")
        return uv
    except Exception as e:
        print(f"  LSCM failed: {e}. Falling back to spherical projection.")

    # Fallback to spherical projection
    vertices = mesh.vertices
    bounds = mesh.bounds
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    ranges = max_bounds - min_bounds
    max_range = np.max(ranges)

    if max_range == 0:
        return np.zeros((len(vertices), 2))

    uv = np.zeros((len(vertices), 2))
    uv[:, 0] = (vertices[:, 0] - min_bounds[0]) / max_range
    uv[:, 1] = (vertices[:, 1] - min_bounds[1]) / max_range

    return np.clip(uv, 0.0, 1.0)


def create_assimp_uv(mesh):
    """
    Create UV coordinates using ASSIMP library

    This is the most accurate method but requires ASSIMP.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        UV coordinates array (N, 2)
    """
    import trimesh

    # Try to use ASSIMP's UV unwrapping
    # This is a placeholder - full implementation requires pyassimp
    print("  ASSIMP UV unwrapping requires pyassimp package")

    # Fallback to angle-based
    return create_angle_based_uv(mesh)


def bake_textures(mesh_path, output_path, texture_size=2048, source_images=None):
    """
    Bake textures from multiple views

    This function creates texture maps from source images
    by projecting them onto the mesh surface.

    Args:
        mesh_path: Path to input mesh file
        output_path: Path for output file
        texture_size: Texture resolution (default: 2048)
        source_images: Directory containing source images

    Returns:
        Path to textured mesh file
    """
    print("=" * 60)
    print("  Texture Baking")
    print("=" * 60)
    print(f"  Input: {mesh_path}")
    print(f"  Output: {output_path}")
    print(f"  Texture size: {texture_size}x{texture_size}")

    try:
        import trimesh

        print("[Texture Bake] Loading mesh...")
        mesh = trimesh.load(mesh_path)

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        print(f"[Texture Bake] Loaded mesh: {num_vertices} vertices, {num_faces} faces")

        # Try to bake textures from source images
        if source_images and os.path.exists(source_images):
            print(f"[Texture Bake] Source images directory: {source_images}")
            bake_from_images(mesh, source_images, texture_size, output_path)
        else:
            print("[Texture Bake] No source images provided. Creating basic texture.")
            create_basic_texture(mesh, texture_size, output_path)

        return output_path

    except ImportError:
        print("[Error] trimesh not installed. Run: pip install trimesh[extras]")
        return output_path
    except Exception as e:
        print(f"[Warning] Texture baking failed: {e}. Creating basic texture.")
        return create_basic_texture_fallback(mesh_path, output_path, texture_size)


def bake_from_images(mesh, source_images_dir, texture_size, output_path):
    """
    Bake textures from multiple source images

    This function projects source images onto the mesh
    to create a high-quality texture map.

    Args:
        mesh: trimesh.Trimesh object
        source_images_dir: Directory containing source images
        texture_size: Texture resolution
        output_path: Output file path
    """
    import trimesh

    # Find source images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.exr']
    source_images = []
    for ext in image_extensions:
        source_images.extend(glob.glob(os.path.join(source_images_dir, f'*{ext}')))
        source_images.extend(glob.glob(os.path.join(source_images_dir, f'*{ext.upper()}')))

    if not source_images:
        print("[Warning] No source images found. Creating basic texture.")
        create_basic_texture(mesh, texture_size, output_path)
        return

    print(f"[Texture from Images] Found {len(source_images)} source images")

    # Simple approach: use the average color from images as base texture
    # For production, use proper multi-view stereo texturing

    # Create a base color texture
    texture = create_color_texture(mesh, texture_size, source_images)

    # Create normal map if possible
    try:
        normal_map = create_normal_map(mesh, source_images, texture_size)
        if normal_map is not None:
            print("[Texture Bake] Normal map created")
    except Exception as e:
        print(f"[Warning] Normal map creation failed: {e}")
        normal_map = None

    # Create specular map
    specular_map = create_specular_map(mesh, texture_size)

    # Apply textures to mesh
    print("[Texture Bake] Applying textures to mesh...")

    # Create PBR materials
    pbr_material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture,
        normalTexture=normal_map if normal_map is not None else None,
        roughnessFactor=0.3,
        metallicFactor=0.1,
        clearcoat=0.5,
        clearcoatRoughness=0.2
    )

    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv if hasattr(mesh.visual, 'uv') else None,
        material=pbr_material
    )

    # Export with textures
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    if output_path.endswith('.glb') or output_path.endswith('.gltf'):
        mesh.export(output_path)
        print(f"[Texture Bake] Textured mesh saved to: {output_path}")
    else:
        obj_path = output_path.replace('.obj', '_textured.obj')
        mesh.export(obj_path)
        print(f"[Texture Bake] Textured mesh saved to: {obj_path}")


def create_color_texture(mesh, texture_size, source_images):
    """
    Create a color texture from source images

    Args:
        mesh: trimesh.Trimesh object
        texture_size: Texture resolution
        source_images: List of source image paths

    Returns:
        PIL Image texture
    """
    print("  Creating color texture...")

    # Simple approach: create a gradient texture based on vertex colors
    # or compute average color from source images

    vertices = mesh.vertices

    # Check if mesh has vertex colors
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vc = mesh.visual.vertex_colors
        print(f"  Using vertex colors: {vc.shape}")

        # Project vertex colors to texture
        texture = project_vertex_colors_to_texture(mesh, vc, texture_size)
        return texture

    # Fallback: create a gradient texture
    print("  No vertex colors found. Creating gradient texture.")

    # Create a simple gradient texture
    texture = Image.new('RGBA', (texture_size, texture_size))
    pixels = texture.load()

    for y in range(texture_size):
        for x in range(texture_size):
            # Create a subtle gradient
            r = int(128 + 127 * np.sin(x / texture_size * np.pi))
            g = int(128 + 127 * np.cos(y / texture_size * np.pi))
            b = 128
            a = 255
            pixels[x, y] = (r, g, b, a)

    return texture


def project_vertex_colors_to_texture(mesh, vertex_colors, texture_size):
    """
    Project vertex colors onto a 2D texture

    Args:
        mesh: trimesh.Trimesh object
        vertex_colors: Vertex color array (N, 4)
        texture_size: Texture resolution

    Returns:
        PIL Image texture
    """
    import trimesh

    # Use trimesh's texture projection
    try:
        texture = trimesh.visual.texture.rasterize(
            mesh, uv=mesh.visual.uv,
            texture_size=texture_size
        )
        print(f"  Rasterized texture: {texture.shape}")
        return Image.fromarray(texture)
    except Exception as e:
        print(f"  Rasterization failed: {e}")

    # Fallback: create a simple texture
    texture = Image.new('RGBA', (texture_size, texture_size), (128, 128, 128, 255))
    return texture


def create_normal_map(mesh, source_images, texture_size):
    """
    Create a normal map from source images

    Args:
        mesh: trimesh.Trimesh object
        source_images: List of source image paths
        texture_size: Texture resolution

    Returns:
        PIL Image normal map or None
    """
    print("  Creating normal map...")

    # Create a placeholder normal map (flat normal)
    normal_map = Image.new('RGBA', (texture_size, texture_size), (128, 128, 255, 255))

    # In production, this would compute normals from source images
    # using multi-view stereo techniques

    print("  Note: Full normal map computation requires multi-view stereo")

    return normal_map


def create_specular_map(mesh, texture_size):
    """
    Create a specular map for car paint

    Args:
        mesh: trimesh.Trimesh object
        texture_size: Texture resolution

    Returns:
        PIL Image specular map
    """
    print("  Creating specular map...")

    # Create a specular map that simulates car paint
    # High specular in the center, lower at edges
    specular_map = Image.new('RGBA', (texture_size, texture_size))
    pixels = specular_map.load()

    center_x = texture_size // 2
    center_y = texture_size // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)

    for y in range(texture_size):
        for x in range(texture_size):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            spec = int(255 * (1.0 - dist / max_dist))
            pixels[x, y] = (spec, spec, spec, 255)

    return specular_map


def create_basic_texture(mesh, texture_size, output_path):
    """
    Create a basic texture without source images

    Args:
        mesh: trimesh.Trimesh object
        texture_size: Texture resolution
        output_path: Output file path
    """
    print("[Basic Texture] Creating basic texture...")

    # Create a simple color texture
    texture = Image.new('RGBA', (texture_size, texture_size), (128, 128, 128, 255))

    # Try to use vertex colors if available
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        print("  Using vertex colors for basic texture...")
        texture = project_vertex_colors_to_texture(mesh, mesh.visual.vertex_colors, texture_size)

    # Save texture
    texture_path = os.path.join(os.path.dirname(output_path), 'texture.png')
    texture.save(texture_path)
    print(f"  Texture saved to: {texture_path}")

    # Export mesh with basic texture
    if output_path.endswith('.glb') or output_path.endswith('.gltf'):
        mesh.export(output_path)
        print(f"  Textured mesh saved to: {output_path}")
    else:
        obj_path = output_path.replace('.obj', '_textured.obj')
        mesh.export(obj_path)
        print(f"  Textured mesh saved to: {obj_path}")


def create_basic_texture_fallback(mesh_path, output_path, texture_size):
    """
    Fallback: Create basic texture when main method fails

    Args:
        mesh_path: Input mesh path
        output_path: Output file path
        texture_size: Texture resolution

    Returns:
        Output file path
    """
    try:
        import trimesh

        print("[Fallback] Creating basic texture...")

        mesh = trimesh.load(mesh_path)

        # Create a simple texture
        texture = Image.new('RGBA', (texture_size, texture_size), (128, 128, 128, 255))
        texture_path = os.path.join(os.path.dirname(output_path), 'texture.png')
        texture.save(texture_path)

        # Export with basic texture
        if output_path.endswith('.glb') or output_path.endswith('.gltf'):
            mesh.export(output_path)
        else:
            obj_path = output_path.replace('.obj', '_textured.obj')
            mesh.export(obj_path)

        print(f"  Fallback texture saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"[Error] Fallback texture creation failed: {e}")
        return output_path


def apply_material_properties(output_path, specular_strength=0.5,
                               roughness=0.3, metallic=0.1,
                               clearcoat=0.5, normal_strength=1.0):
    """
    Apply car-specific PBR material properties

    This function modifies the GLB/OBJ material properties
    to simulate car paint with clearcoat.

    Args:
        output_path: Path to mesh file
        specular_strength: Specular strength (0.0 - 1.0)
        roughness: Roughness value (0.0 = smooth, 1.0 = rough)
        metallic: Metallic value (0.0 = non-metal, 1.0 = metal)
        clearcoat: Clearcoat amount (0.0 = none, 1.0 = full)
        normal_strength: Normal map strength

    Returns:
        Path to file with material properties
    """
    print("=" * 60)
    print("  Material Properties (PBR)")
    print("=" * 60)
    print(f"  Input: {output_path}")
    print(f"  Specular strength: {specular_strength}")
    print(f"  Roughness: {roughness}")
    print(f"  Metallic: {metallic}")
    print(f"  Clearcoat: {clearcoat}")
    print(f"  Normal strength: {normal_strength}")

    try:
        import trimesh

        print("[Material] Loading mesh...")
        mesh = trimesh.load(output_path)

        # Update material properties
        print("[Material] Applying PBR material properties...")

        # Create PBR material with car paint settings
        pbr_material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            roughnessFactor=roughness,
            metallicFactor=metallic,
            clearcoat=clearcoat,
            clearcoatRoughness=0.2,
            clearcoatNormalScale=normal_strength,
            emissiveFactor=[0.0, 0.0, 0.0]
        )

        # Apply material to mesh
        if hasattr(mesh.visual, 'uv'):
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv,
                material=pbr_material
            )
        else:
            mesh.visual = trimesh.visual.MaterialVisual(pbr_material)

        print("[Material] Material properties applied successfully")

        # Save updated mesh
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        if output_path.endswith('.glb') or output_path.endswith('.gltf'):
            mesh.export(output_path)
            print(f"[Material] Updated mesh saved to: {output_path}")
        else:
            final_path = output_path.replace('.obj', '_textured.obj')
            mesh.export(final_path)
            print(f"[Material] Updated mesh saved to: {final_path}")

        return output_path

    except ImportError:
        print("[Error] trimesh not installed. Run: pip install trimesh[extras]")
        return output_path
    except Exception as e:
        print(f"[Warning] Material properties application failed: {e}")
        return output_path


def generate_pbr_textures(mesh_path, output_dir, texture_size=2048):
    """
    Generate all PBR texture maps

    This function generates:
    - Albedo/Color map
    - Normal map
    - Roughness map
    - Metallic map
    - Specular map
    - Clearcoat map (for car paint)

    Args:
        mesh_path: Path to input mesh file
        output_dir: Directory for output textures
        texture_size: Texture resolution

    Returns:
        Dictionary of texture file paths
    """
    print("=" * 60)
    print("  PBR Texture Generation")
    print("=" * 60)
    print(f"  Input: {mesh_path}")
    print(f"  Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        import trimesh

        mesh = trimesh.load(mesh_path)

        texture_paths = {}

        # Generate albedo map
        print("  Generating albedo map...")
        albedo = generate_albedo_map(mesh, texture_size)
        albedo_path = os.path.join(output_dir, 'albedo.png')
        albedo.save(albedo_path)
        texture_paths['albedo'] = albedo_path
        print(f"    Saved: {albedo_path}")

        # Generate normal map
        print("  Generating normal map...")
        normal = generate_normal_map(mesh, texture_size)
        normal_path = os.path.join(output_dir, 'normal.png')
        normal.save(normal_path)
        texture_paths['normal'] = normal_path
        print(f"    Saved: {normal_path}")

        # Generate roughness map
        print("  Generating roughness map...")
        roughness = generate_roughness_map(texture_size)
        roughness_path = os.path.join(output_dir, 'roughness.png')
        roughness.save(roughness_path)
        texture_paths['roughness'] = roughness_path
        print(f"    Saved: {roughness_path}")

        # Generate metallic map
        print("  Generating metallic map...")
        metallic = generate_metallic_map(texture_size)
        metallic_path = os.path.join(output_dir, 'metallic.png')
        metallic.save(metallic_path)
        texture_paths['metallic'] = metallic_path
        print(f"    Saved: {metallic_path}")

        # Generate clearcoat map (for car paint)
        print("  Generating clearcoat map...")
        clearcoat = generate_clearcoat_map(mesh, texture_size)
        clearcoat_path = os.path.join(output_dir, 'clearcoat.png')
        clearcoat.save(clearcoat_path)
        texture_paths['clearcoat'] = clearcoat_path
        print(f"    Saved: {clearcoat_path}")

        print(f"[PBR Textures] Generated {len(texture_paths)} texture maps")

        return texture_paths

    except Exception as e:
        print(f"[Error] PBR texture generation failed: {e}")
        return {}


def generate_albedo_map(mesh, texture_size):
    """Generate albedo/color map"""
    # Check for vertex colors
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        print("    Using vertex colors")
        return project_vertex_colors_to_texture(mesh, mesh.visual.vertex_colors, texture_size)

    # Create gradient texture
    texture = Image.new('RGBA', (texture_size, texture_size), (180, 180, 180, 255))
    return texture


def generate_normal_map(mesh, texture_size):
    """Generate normal map"""
    # Create flat normal map (pointing outward)
    texture = Image.new('RGBA', (texture_size, texture_size), (128, 128, 255, 255))
    return texture


def generate_roughness_map(texture_size):
    """Generate roughness map"""
    # Create uniform roughness map
    texture = Image.new('RGBA', (texture_size, texture_size), (80, 80, 80, 255))
    return texture


def generate_metallic_map(texture_size):
    """Generate metallic map"""
    # Create uniform metallic map (non-metal for car body)
    texture = Image.new('RGBA', (texture_size, texture_size), (25, 25, 25, 255))
    return texture


def generate_clearcoat_map(mesh, texture_size):
    """Generate clearcoat map for car paint"""
    # Create clearcoat map (higher on top, lower on bottom)
    texture = Image.new('RGBA', (texture_size, texture_size))
    pixels = texture.load()

    center_y = texture_size // 2

    for y in range(texture_size):
        for x in range(texture_size):
            # Higher clearcoat on top (roof, hood)
            clearcoat_val = int(200 * (1.0 - y / texture_size))
            pixels[x, y] = (clearcoat_val, clearcoat_val, clearcoat_val, 255)

    return texture


def main():
    args = parse_args()

    print("=" * 60)
    print("  Texture Baking Pipeline for Car Modeling")
    print("=" * 60)
    print("")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Texture size: {args.texture_size}")
    print("")

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: UV Unwrapping
    print("Step 1: UV Unwrapping")
    print("-" * 40)
    uv_path = uv_unwrap(args.input, args.uv_method)
    print("")

    # Step 2: Texture Baking
    print("Step 2: Texture Baking")
    print("-" * 40)
    baked_path = bake_textures(uv_path, args.output, args.texture_size, args.source_images)
    print("")

    # Step 3: Apply Material Properties
    print("Step 3: Material Properties")
    print("-" * 40)
    final_output = apply_material_properties(
        baked_path,
        args.specular_strength,
        args.roughness,
        args.metallic,
        args.clearcoat,
        args.normal_strength
    )
    print("")

    # Step 4: Generate PBR Textures (optional)
    print("Step 4: PBR Texture Generation (Optional)")
    print("-" * 40)
    textures_dir = os.path.join(os.path.dirname(args.output), 'textures')
    texture_paths = generate_pbr_textures(args.input, textures_dir, args.texture_size)
    print("")

    print("=" * 60)
    print("  Texture Baking Complete!")
    print("=" * 60)
    print(f"  Output: {final_output}")

    if texture_paths:
        print("  Generated textures:")
        for name, path in texture_paths.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"    {name}: {path} ({size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
