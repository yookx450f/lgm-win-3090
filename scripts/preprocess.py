#!/usr/bin/env python3
"""
Preprocessing Script for Car 3D Modeling
- Image normalization
- Car body mask generation (background removal)
- Image alignment
- Camera position estimation (Structure-from-Motion)
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess car images for 3D modeling')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing car images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for preprocessed images')
    parser.add_argument('--image_size', type=int, default=1024, help='Target image size (default: 1024)')
    parser.add_argument('--bg_color', type=str, default='white', help='Background color: white, black, green (default: white)')
    return parser.parse_args()


def load_images(input_dir):
    """Load images from input directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))
    
    if not image_paths:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images")
    return sorted(image_paths)


def normalize_image(image_path, target_size=1024, bg_color='white'):
    """Normalize image to target size"""
    img = Image.open(image_path).convert('RGB')
    
    # Calculate aspect ratio and resize maintaining aspect ratio
    orig_w, orig_h = img.size
    scale = target_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Create canvas with background color
    if bg_color == 'white':
        bg = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    elif bg_color == 'black':
        bg = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    elif bg_color == 'green':
        bg = Image.new('RGB', (target_size, target_size), (0, 255, 0))
    else:
        bg = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    # Paste image centered on canvas
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    bg.paste(img, (x_offset, y_offset))
    
    return np.array(bg)


def generate_mask(image_array):
    """Generate mask for car body (background removal)"""
    # Convert to HSV color space
    img_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Define range of colors (adjust based on car colors)
    # This is a simple color-based mask - can be improved with ML-based segmentation
    
    # For now, return a full mask (no background removal)
    # In production, use a pre-trained model for car segmentation
    mask = np.ones((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
    
    return mask


def save_preprocessed(image_array, mask, output_path, image_name):
    """Save preprocessed image and mask"""
    # Save image
    img_pil = Image.fromarray(image_array)
    img_path = os.path.join(output_path, f'{image_name}.png')
    img_pil.save(img_path)
    
    # Save mask
    mask_path = os.path.join(output_path, f'{image_name}_mask.png')
    Image.fromarray(mask).save(mask_path)
    
    return img_path, mask_path


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    image_paths = load_images(args.input_dir)
    
    print(f"Preprocessing {len(image_paths)} images...")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    for i, img_path in enumerate(image_paths):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing [{i+1}/{len(image_paths)}]: {image_name}")
        
        # Normalize image
        image_array = normalize_image(img_path, args.image_size, args.bg_color)
        
        # Generate mask
        mask = generate_mask(image_array)
        
        # Save preprocessed data
        save_preprocessed(image_array, mask, args.output_dir, image_name)
        
        print(f"  -> Saved: {image_name}.png, {image_name}_mask.png")
    
    print("")
    print(f"Preprocessing complete! Output saved to: {args.output_dir}")
    print(f"Total images: {len(image_paths)}")


if __name__ == '__main__':
    main()
