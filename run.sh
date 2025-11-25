#!/bin/bash

# Simple shell script to run MatchStereo inference

IMG0_DIR="/home/tomasjurica/projects/MatchAttention/input_data/left"
IMG1_DIR="/home/tomasjurica/projects/MatchAttention/input_data/right"
WEIGHTS="/home/tomasjurica/projects/MatchAttention/checkpoints/matchstereo_tiny_fsd.pth"
OUTPUT_PATH="outputs"

python3 run_img.py \
    --img0_dir "$IMG0_DIR" \
    --img1_dir "$IMG1_DIR" \
    --checkpoint_path "$WEIGHTS" \
    --output_path "$OUTPUT_PATH" \
    --save_right

