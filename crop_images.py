#!/usr/bin/env python3
"""Simple script to crop both input images"""

from PIL import Image
import os

# Input paths (load from input_data_full)
INPUT_DIR = "/home/tomasjurica/projects/MatchAttention/input_data_full"
IMG0_PATH = os.path.join(INPUT_DIR, "im0.jpg")
IMG1_PATH = os.path.join(INPUT_DIR, "im1.jpg")

# Crop parameters: left, top, width, height
CROP_LEFT = 100
CROP_TOP = 150
CROP_WIDTH = 416
CROP_HEIGHT = 416

RESIZE_TO = (256, 256)

# Output paths (save to input_data)
OUTPUT_DIR = "/home/tomasjurica/projects/MatchAttention/input_data"
OUTPUT_LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
OUTPUT_RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
OUTPUT_IMG0_CROPPED = os.path.join(OUTPUT_LEFT_DIR, "im0_cropped.jpg")
OUTPUT_IMG0_RESIZED = os.path.join(OUTPUT_LEFT_DIR, "im0_cropped_resized.jpg")
OUTPUT_IMG1_CROPPED = os.path.join(OUTPUT_RIGHT_DIR, "im1_cropped.jpg")
OUTPUT_IMG1_RESIZED = os.path.join(OUTPUT_RIGHT_DIR, "im1_cropped_resized.jpg")

def crop_image(input_path, output_cropped_path, output_resized_path, left, top, width, height):
    """Crop an image, resize it, and save both versions"""
    img = Image.open(input_path)
    
    # Calculate crop box: (left, top, right, bottom)
    right = left + width
    bottom = top + height
    
    # Crop the image
    cropped = img.crop((left, top, right, bottom))
    
    # Save the cropped image (before resizing)
    cropped.save(output_cropped_path)
    print(f"Cropped {input_path} -> {output_cropped_path}")
    print(f"  Crop region: ({left}, {top}) to ({right}, {bottom}), size: {width}x{height}")
    
    # Resize the cropped image
    resized = cropped.resize(RESIZE_TO, Image.Resampling.LANCZOS)
    
    # Save the resized image
    resized.save(output_resized_path)
    print(f"Resized -> {output_resized_path}")
    print(f"  Resized to: {RESIZE_TO}")

if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_LEFT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RIGHT_DIR, exist_ok=True)
    
    # Crop both images
    crop_image(IMG0_PATH, OUTPUT_IMG0_CROPPED, OUTPUT_IMG0_RESIZED, CROP_LEFT, CROP_TOP, CROP_WIDTH, CROP_HEIGHT)
    crop_image(IMG1_PATH, OUTPUT_IMG1_CROPPED, OUTPUT_IMG1_RESIZED, CROP_LEFT, CROP_TOP, CROP_WIDTH, CROP_HEIGHT)
    
    print("\nDone! Cropped and resized images saved.")

