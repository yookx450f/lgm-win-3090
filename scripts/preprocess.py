#!/usr/bin/env python3
"""
Preprocessing Script for Car 3D Modeling
- Image normalization
- Background masking (car segmentation)
- Image alignment
"""

import argparse
import os
import sys
import glob
import subprocess
from pathlib import Path
import numpy as np

try:
    from PIL import Image, ImageOps, ImageFilter
    import cv2
except ImportError:
    print("[Error] Please install required packages:")
    print("  pip install Pillow opencv-python numpy")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing for car 3D modeling')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing car images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed images')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Target image size (default: 1024)')
    parser.add_argument('--bg_color', type=str, default='white',
                        choices=['white', 'black', 'green', 'transparent'],
                        help='Background color (default: white)')
    return parser.parse_args()


def normalize_image(image: Image.Image, target_size: int) -> Image.Image:
    """Normalize image size and format"""
    # Handle special image modes (P, RGBA, L, 1, etc.)
    if image.mode == 'P':
        # Palette mode: convert using palette data
        image = image.convert('RGB')
    elif image.mode == 'RGBA':
        # Transparent background: composite onto white background
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image)
        image = bg
    elif image.mode == 'L':
        # Grayscale: convert to RGB
        image = image.convert('RGB')
    elif image.mode == '1':
        # 1-bit images: convert to RGB
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        # Fallback: try to convert to RGB
        try:
            image = image.convert('RGB')
        except Exception:
            print(f"    [Warning] Could not convert image from {image.mode} to RGB, skipping")
            return image
    
    # Resize maintaining aspect ratio
    original_size = image.size
    ratio = target_size / max(original_size)
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create canvas with target size
    canvas = Image.new('RGB', (target_size, target_size), 'white')
    
    # Center the image
    x_offset = (target_size - new_size[0]) // 2
    y_offset = (target_size - new_size[1]) // 2
    
    canvas.paste(image, (x_offset, y_offset))
    
    return canvas


def detect_background(image: Image.Image) -> Image.Image:
    """Detect and create background mask"""
    img_array = np.array(image)
    
    # Sample corners to determine background color
    h, w = img_array.shape[:2]
    corner_size = min(20, h // 10, w // 10)
    
    corners = [
        img_array[0:corner_size, 0:corner_size],
        img_array[0:corner_size, w-corner_size:w],
        img_array[h-corner_size:h, 0:corner_size],
        img_array[h-corner_size:h, w-corner_size:w]
    ]
    
    bg_color = np.mean(np.vstack([c.reshape(-1, 3) for c in corners]), axis=0)
    
    # Calculate difference from background
    diff = np.abs(img_array.astype(float) - bg_color)
    mask = np.any(diff > 30, axis=2)  # Threshold for foreground
    
    # Create mask image
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    
    return mask_image


def remove_background(image: Image.Image, bg_color: str = 'white') -> Image.Image:
    """Remove background and replace with specified color"""
    # Get mask
    mask = detect_background(image)
    
    # Convert to RGBA
    rgba_image = image.convert('RGBA')
    
    # Create alpha channel from mask (L mode)
    # The mask is 'L' mode (grayscale), use it directly as alpha
    alpha_channel = mask.convert('L')
    
    # Replace the alpha channel of the original image
    # split() on RGBA returns 4 channels (R, G, B, A)
    channels = rgba_image.split()
    # Use the original alpha or the mask alpha (whichever is more opaque)
    rgba_image = Image.merge('RGBA', (channels[0], channels[1], channels[2], alpha_channel))
    
    # Create background
    if bg_color == 'white':
        bg = Image.new('RGBA', rgba_image.size, (255, 255, 255, 255))
    elif bg_color == 'black':
        bg = Image.new('RGBA', rgba_image.size, (0, 0, 0, 255))
    elif bg_color == 'green':
        bg = Image.new('RGBA', rgba_image.size, (0, 255, 0, 255))
    else:
        bg = Image.new('RGBA', rgba_image.size, (255, 255, 255, 255))
    
    # Composite
    result = Image.alpha_composite(bg, rgba_image)
    
    return result.convert('RGB')


def align_image(image: Image.Image) -> Image.Image:
    """Align image to center and correct orientation"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find bounding box of foreground
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop to bounding box
        cropped = img_array[y:y+h, x:x+w]
        
        # Resize to square
        size = max(w, h)
        resized = cv2.resize(cropped, (size, size), cv2.INTER_LANCZOS4)
        
        # Create canvas
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)
        cx = (size - w) // 2
        cy = (size - h) // 2
        canvas[cy:cy+h, cx:cx+w] = cropped
        
        return Image.fromarray(canvas)
    
    return image


def enhance_car_features(image: Image.Image) -> Image.Image:
    """Enhance car features for better 3D reconstruction"""
    img_array = np.array(image).astype(float)
    
    # Apply subtle sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    img_array = np.clip(img_array, 0, 255)
    
    # Apply contrast enhancement
    gray = np.mean(img_array, axis=2)
    histogram, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    cdf = histogram.cumsum()
    
    # Normalize CDF
    cdf_normalized = cdf * 255.0 / cdf[-1]
    
    result = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def preprocess_image(input_path: str, output_path: str, target_size: int, bg_color: str):
    """Process a single image"""
    print(f"  Processing: {os.path.basename(input_path)}")
    
    # Load image
    image = Image.open(input_path)
    
    # Step 1: Normalize
    image = normalize_image(image, target_size)
    
    # Step 2: Remove background
    if bg_color != 'transparent':
        image = remove_background(image, bg_color)
    
    # Step 3: Align
    image = align_image(image)
    
    # Step 4: Enhance
    image = enhance_car_features(image)
    
    # Save
    image.save(output_path, 'JPEG', quality=95)
    print(f"    -> Saved: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Preprocessing Step")
    print("=" * 60)
    print(f"  Input: {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Image Size: {args.image_size}")
    print(f"  Background: {args.bg_color}")
    print("")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
    
    if not image_files:
        print("[Error] No image files found in input directory")
        sys.exit(1)
    
    print(f"  Found {len(image_files)} images")
    print("")
    
    # Process each image
    for i, input_path in enumerate(image_files, 1):
        filename = os.path.basename(input_path)
        output_filename = f"preprocessed_{i:03d}_{filename.rsplit('.', 1)[0]}.jpg"
        output_path = os.path.join(args.output_dir, output_filename)
        
        try:
            preprocess_image(input_path, output_path, args.image_size, args.bg_color)
        except Exception as e:
            print(f"  [Warning] Failed to process {filename}: {e}")
    
    print("")
    print(f"  Preprocessing complete! {len(image_files)} images processed.")
    print(f"  Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
