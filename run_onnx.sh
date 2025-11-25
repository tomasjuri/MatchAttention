#!/bin/bash

# Simple shell script to run MatchStereo inference with ONNX models
# Note: FP16 models may not work on CPU-only systems (like Raspberry Pi)
# Use FP32 or INT8 models for better CPU compatibility

IMG0_DIR="/home/tomasjurica/projects/MatchAttention/input_data_416/left"
IMG1_DIR="/home/tomasjurica/projects/MatchAttention/input_data_416/right"
ONNX_MODEL="/home/tomasjurica/projects/MatchAttention/exports/matchstereo_tiny_default.onnx"
OUTPUT_PATH="outputs_onnx"

python3 run_img.py \
    --img0_dir "$IMG0_DIR" \
    --img1_dir "$IMG1_DIR" \
    --checkpoint_path "$ONNX_MODEL" \
    --output_path "$OUTPUT_PATH" \
    --save_right \
    --device_id -1

