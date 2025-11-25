#!/usr/bin/env python3
"""Simple script to crop both input images"""

from PIL import Image
import os

# Input paths
IMG0_PATH = "/Users/tomasjurica/projects/FingernailCalibration/StereoVision/MatchAttention/input_data/im0.jpg"
IMG1_PATH = "/Users/tomasjurica/projects/FingernailCalibration/StereoVision/MatchAttention/input_data/im1.jpg"

# Crop parameters: left, top, width, height
CROP_LEFT = 100
CROP_TOP = 150
CROP_WIDTH = 400
CROP_HEIGHT = 400

# Output paths
OUTPUT_DIR = "/Users/tomasjurica/projects/FingernailCalibration/StereoVision/MatchAttention/input_data"
OUTPUT_IMG0 = os.path.join(OUTPUT_DIR, "im0_cropped.jpg")
OUTPUT_IMG1 = os.path.join(OUTPUT_DIR, "im1_cropped.jpg")

def crop_image(input_path, output_path, left, top, width, height):
    """Crop an image and save it"""
    img = Image.open(input_path)
    
    # Calculate crop box: (left, top, right, bottom)
    right = left + width
    bottom = top + height
    
    # Crop the image
    cropped = img.crop((left, top, right, bottom))
    
    # Save the cropped image
    cropped.save(output_path)
    print(f"Cropped {input_path} -> {output_path}")
    print(f"  Crop region: ({left}, {top}) to ({right}, {bottom}), size: {width}x{height}")

if __name__ == "__main__":
    # Crop both images
    crop_image(IMG0_PATH, OUTPUT_IMG0, CROP_LEFT, CROP_TOP, CROP_WIDTH, CROP_HEIGHT)
    crop_image(IMG1_PATH, OUTPUT_IMG1, CROP_LEFT, CROP_TOP, CROP_WIDTH, CROP_HEIGHT)
    
    print("\nDone! Cropped images saved.")

